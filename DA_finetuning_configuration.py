import torch


class Configuration:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 2  # number of classes
    train_batch_size = 30  # batch size of training set
    test_batch_size = 20  # batch size of testing set
    num_epochs = 100  # number of epochs
    nsplits = 5  # n. of splits for K-fold cross-validation
    lr = 0.0005  # learning rate
    l2_reg = 0.1  # l2 regularization parameter
    criterion = torch.nn.CrossEntropyLoss()   # objective loss
