import torch
import os
import pandas as pd
import sys
import git
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, precision_recall_curve, roc_curve
from sklearn.metrics import f1_score, accuracy_score
from alive_progress import alive_bar
from DA_plot_functions import plot_complete_report, plot_auc_curve, plot_confusion_matrix
torch.cuda.empty_cache()

# ######################
# PRE-TRAINED MODELS  ##
# ######################


def load_pretrained_model(model, config):
    models_A = ['ADnet', 'ADnetEx']
    models_B = ['ResNet18', 'ResNet50', 'ResNet101']
    if model not in models_A + models_B:
        print('Error. Chosen model is not allowed.')
        print('Please choose among ADnet, ADnetEx, ResNet18, ResNet50, and ResNet101')
        sys.exit()
    if model in models_A:
        ADnet_path = os.getcwd() + '/ADnet/'
        git.Repo.clone_from('https://github.com/rturrisige/3D_CNN_pretrained_model.git', ADnet_path)
        sys.path.append(ADnet_path)
        from AD_pretrained_utilities import CNN_8CL_B, CNN
        net_config = CNN_8CL_B()
        net = CNN(net_config)
        w = torch.load(ADnet_path + 'AD_pretrained_weights.pt')
        net.load_state_dict(w)
        nchannels = 1
        if model == 'ADnetEx':
            net.f[0] = torch.nn.Linear(256, config.n_classes)
    else:
        if model == 'ResNet18':
            from torchvision.models import resnet18, ResNet18_Weights
            model_2d = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model_2d.fc = torch.nn.Linear(512, config.n_classes)
        elif model == 'ResNet50':
            from torchvision.models import resnet50, ResNet50_Weights
            model_2d = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model_2d.fc = torch.nn.Linear(2048, config.n_classes)
        elif model == 'ResNet101':
            from torchvision.models import resnet101, ResNet101_Weights
            model_2d = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            model_2d.fc = torch.nn.Linear(2048, config.n_classes)
        from acsconv.converters import ACSConverter
        net = ACSConverter(model_2d)
        nchannels = 3
    next(net.parameters()).requires_grad = False
    return net, nchannels


# ######################
#    TRAINING         ##
# ######################


def train_net(net, config, train_loader, train_acc=False):
    # PARAMETER DEFINITION:
    epochs_loss, epochs_train_acc = [], []
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_reg)
    total_step = len(train_loader)
    print('Number of total step per epoch:', total_step)
    # STARTING TRAINING:
    for epoch_counter in range(config.num_epochs):
        epoch_loss = 0.0
        # EPOCH TRAINING:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            net = net.train()
            batch_x, batch_y = batch_x.to(config.device, dtype=torch.float), batch_y.to(config.device)
            prob = net(batch_x)
            loss = config.criterion(prob, batch_y.long())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0 and step != 0:
                print('Epoch {} step {} - Loss {:.4f}'.format(epoch_counter, step, loss.item()))

        # COMPUTE TRAINING ACCURACY:
        if train_acc:
            train_accuracy = test_model(net, config, train_loader).item()
            epochs_train_acc.append(train_accuracy)
            print('Epoch {} - Loss {:.4f} - Train Acc {:.4f}'.format(epoch_counter, epoch_loss / total_step,
                                                                       train_accuracy))
        else:
            print('Epoch {} - Loss {:.4f}'.format(epoch_counter, epoch_loss / total_step))
        epochs_loss.append(epoch_loss / total_step)

    if train_acc:
        return epochs_loss, epochs_train_acc
    else:
        return epochs_loss

# ######################
#     EVALUATION      ##
# ######################


def test_model(net, config, data_loader):
    net = net.eval()
    tot_acc = 0.0
    N = 0
    for _, (x, y) in enumerate(data_loader):
        x, y = x.to(config.device, dtype=torch.float), y.to(config.device)
        acc = torch.sum(torch.max(net(x), 1)[1] == y.long()).to(dtype=torch.float)
        tot_acc += acc
        N += x.shape[0]
    return tot_acc / N


def predict(net, data_loader, device, return_prob=True):
    predictions, labels = torch.tensor([]), torch.tensor([])
    all_prob0, all_prob1 = torch.tensor([]), torch.tensor([])
    print('\nModel prediction. Total number of steps:', len(data_loader))
    with alive_bar(len(data_loader), bar='classic', spinner='arrow') as bar:
        for _, (x, y) in enumerate(data_loader):
            x = x.to(device, dtype=torch.float)
            output = net(x).detach().cpu()
            if return_prob:
                prob0 = output[:, 0]
                prob1 = output[:, 1]
                all_prob0 = torch.cat((all_prob0, prob0), dim=0)
                all_prob1 = torch.cat((all_prob1, prob1), dim=0)
            y_pred = torch.argmax(output, dim=1)
            predictions = torch.cat((predictions, y_pred), dim=0)
            labels = torch.cat((labels, y), dim=0)
            bar()
        if return_prob:
            return labels.numpy(), all_prob0.numpy(), all_prob1.numpy(), predictions.numpy()
        else:
            return labels.numpy(), predictions.numpy()


def evaluation(y_test, yp0, yp1, y_pred, saver_path, labels=None):
    # Evaluate data
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    roc_auc_0 = roc_auc_score(y_test, yp0)
    roc_auc_1 = roc_auc_score(y_test, yp1)
    prec_0, rec_0, _ = precision_recall_curve(y_test, yp0, pos_label=0)
    prec_1, rec_1, _ = precision_recall_curve(y_test, yp1)
    auprc_0 = auc(rec_0, prec_0)
    auprc_1 = auc(rec_1, prec_1)
    fpr, tpr, _ = roc_curve(y_test, yp1)

    # Save data and results
    df_report = pd.DataFrame({'Precision': [precision_0, precision_1],
                              'Recall': [recall_0, recall_1], 'F1-score': [f1_0, f1_1],
                              'AUC': [roc_auc_0, roc_auc_1], 'AUPRC': [auprc_0, auprc_1]})
    df_report.to_csv(saver_path + '/complete_report.csv')
    df_results = pd.DataFrame({'Accuracy': [acc], 'F1-score': [f1], 'AUC': [roc_auc_1]})
    df_results.to_csv(saver_path + '/Results.csv')
    logfile = open(saver_path + '/results.txt', 'w')
    logfile.write('Evaluation.\n\n')
    logfile.write('F1-score: {:.4f}\n'.format(f1))
    logfile.write('Accuracy: {:.4f}\n'.format(acc))
    logfile.write('AUC: {:.4f}\n.'.format(roc_auc_1))
    logfile.flush()
    logfile.close()

    # PLOTS
    plot_complete_report(df_report, saver_path, labels=labels)
    plot_auc_curve(fpr, tpr, roc_auc_1, saver_path)
    plot_confusion_matrix(y_test, y_pred, saver_path, labels=labels)
