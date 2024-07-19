"""
This code performs fine-tuning on AD diagnosis task, given a 3D MRI dataset.
The model is pre-trained and it can be chosen among ADnet, ADnetEx, ResNet18, ResNet50, and ResNet101.
Data augmentation can be applied.
"""

import numpy as np
import os
import argparse
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from DA_utilities import train_net, predict, evaluation, load_pretrained_model
from DA_data_loading import torch_norm, MyDataLoader, my_augmented_data_finder, my_data_list
from DA_finetuning_configuration import Configuration


def cross_fold(config, model, data_path, saver_path, augmentation_dir=None):
    if not os.path.exists(saver_path + '/'):
        os.makedirs(saver_path + '/')
        
    # Load the Network
    net, n_channels = load_pretrained_model(model, config)
    net = net.to(config.device)

    # Create the dataset
    CN, AD = my_data_list(data_path)
    CN_kf = KFold(n_splits=config.nsplits).split(CN)
    AD_kf = KFold(n_splits=config.nsplits).split(AD)
    for i, (train_CN_index, test_CN_index) in enumerate(CN_kf):
        train_AD_index, test_AD_index = next(AD_kf)

        print('Fold ', i)
        if not os.path.exists(saver_path + '/Folder' + str(i)):
            os.makedirs(saver_path + '/Folder' + str(i))

        # Extract training samples from CN and AD file list
        train_dataset = list(np.array(CN)[train_CN_index]) + list(np.array(AD)[train_AD_index])
        if augmentation_dir:
            # Look for augmented training samples
            augmented_train_dataset = my_augmented_data_finder(train_dataset, augmentation_dir)
            train_dataset += augmented_train_dataset
        # Extract test samples from CN and AD file list
        test_dataset = list(np.array(CN)[test_CN_index]) + list(np.array(AD)[test_AD_index])
        print('# Training samples:', len(train_dataset))
        print('# Testing samples:', len(test_dataset))

        # Data loaders
        train_data = MyDataLoader(train_dataset, transform=torch_norm, preprocessing=True, nchannels=n_channels)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True,
                                                   pin_memory=True)
        test_data = MyDataLoader(test_dataset, transform=torch_norm, preprocessing=True, nchannels=n_channels)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False, pin_memory=True)
        
        # Fine-tuning  
        loss = train_net(net, config, train_loader, train_acc=False)

        # Plot history
        plt.plot(loss, label='loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(saver_path + '/Folder' + str(i) + '/training.png')

        # Evaluation                                
        y_test, yp0, yp1, y_pred = predict(net, test_loader, config.device)
        if i == 0:
            labels, prob0, prob1, predictions = y_test, yp0, yp1, y_pred
        else:
            labels = np.concatenate([labels, y_test], 0)
            prob0 = np.concatenate([prob0, yp0], 0)
            prob1 = np.concatenate([prob1, yp1], 0)
            predictions = np.concatenate([predictions, y_pred], 0)
        evaluation(y_test, yp0, yp1, y_pred, saver_path + '/Folder' + str(i), labels=['CN', 'AD'])
    print('Final evaluation...')
    evaluation(labels, prob0, prob1, predictions, saver_path, labels=['CN', 'AD'])
    print('End.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Fine tuning of <model>.
    File from <data_dir> are loaded and used as input for the <model>. 
    Training and evaluation results are saved in
    <saver_dir>.""")
    parser.add_argument('--model', default='ResNet101', type=str,
                        help='The pre-trained model adopted for transfer learning: '
                             'ADnet, ADnetEx, Resnet18, Resnet50, or ResNet101.')
    parser.add_argument('--data_dir', default='/home/rosannaturrisi/storage/ADNI/MRI/ADNI_3T/dataset', type=str,
                        help='The directory that contains npy files to be processed')
    parser.add_argument('--saver_dir', default='./Results/fine-tuning/', type=str,
                        help='The directory where to save the fine-tuned model')
    parser.add_argument('--augmentation_dir', default=None, type=str,
                        help='The directory containing the augmented dataset. If None, augmentation is not applied.')
    args = parser.parse_args()
    config = Configuration()
    cross_fold(config, args.model, args.data_dir, args.saver_dir, args.augmentation_dir)
