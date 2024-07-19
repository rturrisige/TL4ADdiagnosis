import numpy as np
from glob import glob as gg
import torch
import sys
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

torch.cuda.empty_cache()

##
# ######################
#     DATA LOADING    ##
# ######################


def feature_data_preparation(extractor, feature_dir):
    """
    This function load extracted features for AD and CN groups.
    It assumes that CN files are labelled with 0, and AD files are labelled with 2.
    :param extractor: model used as feature extractor, type=str
    :param feature_dir: path to directory with extracted features, type=str
    :return: numpy.array, numpy.array
    """
    models_A = ['ADnet', 'ADnetEx']
    models_B = ['ResNet18', 'ResNet50', 'ResNet101']
    if extractor not in models_A + models_B:
        print('Error. Chosen model is not allowed.')
        print('Please choose among ADnet, ADnetEx, ResNet18, ResNet50, and ResNet101')
        sys.exit()
    data_dir = feature_dir + '/' + extractor 
    CN = gg(data_dir + '/0_*.npy')
    AD = gg(data_dir + '/2_*.npy')
    X = np.load(CN[0], allow_pickle=True)[None, :]
    for name in CN[1:] + AD:
        f = np.load(name, allow_pickle=True)[None, :]
        X = np.concatenate([X, f], 0)
    n_CN = len(CN)
    n_AD = len(AD)
    y = np.array([0] * n_CN + [1] * n_AD)
    return X, y


# ##################
#     TRAINING    ##
# ##################


def apply_classifier(clf, X_train, y_train, X_test, y_test, train_score, test_score, test_acc, test_results, y_prob_0, y_prob_1):
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    y_prob_0.append(clf.predict_proba(X_test)[:, 0])
    y_prob_1.append(clf.predict_proba(X_test)[:, 1])
    train_f1 = f1_score(y_train, train_pred)
    test_results.append([y_test, test_pred])
    acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    print('F1 score. Train={:.4f}. Test={:.4f}.'.format(train_f1, test_f1))
    print('Acc score. Test={:.4f}.'.format(acc))
    print('')
    train_score.append(train_f1)
    test_score.append(test_f1)
    test_acc.append(acc)
    return train_score, test_score, test_acc, test_results, y_prob_0, y_prob_1


def training(X, y, classifier, skf, aug=True):
    train_score, test_score, test_acc, test_results, y_prob_0, y_prob_1 = [], [], [], [], [], []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if aug:
            oversample = SMOTE(random_state=0)
            X_train, y_train = oversample.fit_resample(X_train, y_train)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        train_score, test_score, test_acc, test_results, y_prob_0, y_prob_1 = \
            apply_classifier(classifier, X_train, y_train, X_test, y_test, train_score, test_score, test_acc,
                             test_results, y_prob_0, y_prob_1)
    return train_score, test_score, test_acc, test_results, y_prob_0, y_prob_1


