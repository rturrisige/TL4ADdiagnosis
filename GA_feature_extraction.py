import torch
from acsconv.converters import ACSConverter
import numpy as np
import sys
import os
import git
import argparse
from alive_progress import alive_bar
from glob import glob as gg

current_path = os.getcwd()
tonpy = lambda x: x.detach().cpu().numpy()


def extract_embedding(model, data_dir, saver_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    models_A = ['ADnet', 'ADnetEx']
    models_B = ['ResNet18', 'ResNet50', 'ResNet101']
    if model not in models_A + models_B:
        print('Error. Chosen model is not allowed.')
        print('Please choose among ADnet, ADnetEx, ResNet18, ResNet50, and ResNet101')
        sys.exit()

    if not os.path.exists(saver_dir + '/' + model + '/'):
        os.makedirs(saver_dir + '/' + model + '/')

    if model in models_A:
        ADnet_path = os.getcwd() + '/ADnet/'
        git.Repo.clone_from('https://github.com/rturrisige/3D_CNN_pretrained_model.git', ADnet_path)
        sys.path.append(ADnet_path)
        from extract_embeddings_AD_pretrained import extract_embedding
        from AD_pretrained_utilities import CNN_8CL_B, CNN
        net_config = CNN_8CL_B()
        net = CNN(net_config)
        w = torch.load(ADnet_path + 'AD_pretrained_weights.pt')
        net.load_state_dict(w)
        embedding = 6
        extract_embedding(model, data_dir, embedding, saver_dir, model, device, processing=True)
    else:
        if model == 'ResNet18':
            from torchvision.models import resnet18
            model_2d = resnet18(pretrained=True)
        elif model == 'ResNet50':
            from torchvision.models import resnet50, ResNet50_Weights
            model_2d = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif model == 'ResNet101':
            from torchvision.models import resnet101, ResNet101_Weights
            model_2d = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        model_3d = ACSConverter(model_2d).to(device)
        # remove last layer
        feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])

        files = gg(data_dir + '/*.npy')
        print('\nNumber of files to process:', len(files))
        with alive_bar(len(files), bar='classic', spinner='arrow') as bar:
            for f in files:
                name = os.path.basename(f)
                x = np.load(f, allow_pickle=True)
                x = torch.tensor(x)[None, None, :, :, :]
                x = torch.concat([x, x, x], 1)
                output = feature_extractor(x.to(device))
                representation = tonpy(output.view(-1))
                np.save(saver_dir + '/' + model + '/' + name, representation)
                bar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Extract embeddings from models pretrained on ImageNet.
    File from <data_dir> are loaded and used as input for the model. The embedding <embedding> is extracted and saved in
    <saver_dir>.""")
    parser.add_argument('--data_dir', required=True, type=str,
                        help='The directory that contains npy files to be processed')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='The pre-trained model adopted as feature extractor: Resnet18, Resnet50, or ResNet101')
    parser.add_argument('--saver_dir', default=current_path + 'embeddings_ResNet_pretrained/', type=str,
                        help='The directory where to save the extracted embeddings')
    args = parser.parse_args()
    extract_embedding(args.data_dir, args.model, args.saver_dir)
