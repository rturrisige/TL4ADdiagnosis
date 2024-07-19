import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from GA_utilities import feature_data_preparation, training
from sklearn.preprocessing import StandardScaler
from GA_plot_utilities import compute_CV_confusion_matrix, complete_evaluation, cm_sum, cl_mean
import argparse
sys.path.append(os.getcwd())


def run_classifiers(extractor, feature_dir, saver_path, augmentation=False):
    X, y = feature_data_preparation(extractor, feature_dir)
    saver_path = saver_path + '/' + extractor
    if not os.path.exists(saver_path):
        os.makedirs(saver_path)

    labels = ['CN', 'AD']
    logfile = open(saver_path + '/gridsearch.txt', 'w')

    # K-fold cross validation
    n_splits = 5
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits)

    ##
    # SVM Parameters
    C_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    gamma_range = [0.001, 0.01, 0.1, 1, 10]

    logfile.write('SVM parameters\n')
    logfile.write('C: ' + str(C_range) + '\n')
    logfile.write('gamma: ' + str(gamma_range) + '\n')

    # Linear SVM
    svm_parameters = {'model__C': C_range}
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='linear', max_iter=100000))])
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    linear_best = svm_grid.best_params_
    logfile.write('Linear SVM\n')
    logfile.write('Best C:' + str(linear_best['model__C']) + '\n\n')

    clf = SVC(kernel='linear', C=linear_best['model__C'], max_iter=100000, probability=True)
    L_train_score, L_test_score, L_test_acc, L_test_results, L_y_prob_0, L_y_prob_1 = training(X, y, clf, skf, 
                                                                                               aug=augmentation)

    # SVM (P)

    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='poly'))])

    svm_parameters = {'model__C': C_range, 'model__degree': (2, 3)}
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    poly_best = svm_grid.best_params_
    logfile.write('Polynomial SVM\n')
    logfile.write('Best degree:' + str(poly_best['model__degree']) + '\n')
    logfile.write('Best C:' + str(poly_best['model__C']) + '\n\n')

    clf = SVC(kernel='poly', C=poly_best['model__C'], degree=poly_best['model__degree'], probability=True)
    P_train_score, P_test_score, P_test_acc, P_test_results, P_y_prob_0, P_y_prob_1 = training(X, y, clf, skf, 
                                                                                               aug=augmentation)


    # SVM (RBF)
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='rbf'))])
    svm_parameters = {'model__C': C_range, 'model__gamma': gamma_range}
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    rbf_best = svm_grid.best_params_
    logfile.write('RBF SVM\n')
    logfile.write('Best gamma:' + str(rbf_best['model__gamma']) + '\n')
    logfile.write('Best C:' + str(rbf_best['model__C']) + '\n\n')
    clf = SVC(kernel='rbf', C=rbf_best['model__C'], gamma=rbf_best['model__gamma'], probability=True)
    G_train_score, G_test_score, G_test_acc, G_test_results, G_y_prob_0, G_y_prob_1 = training(X, y, clf, skf,
                                                                                               aug=augmentation)

    # SVM (S)
    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', SVC(kernel='sigmoid'))])

    svm_parameters = {'model__C': C_range}
    svm_grid = GridSearchCV(pipe, param_grid=svm_parameters, cv=cv)
    svm_grid.fit(X, y)
    print(svm_grid.best_params_)
    sigmoid_best = svm_grid.best_params_

    logfile.write('Sigmoid SVM\n')
    logfile.write('Best C:' + str(sigmoid_best['model__C']) + '\n\n')
    clf = SVC(kernel='sigmoid', C=sigmoid_best['model__C'], probability=True)
    S_train_score, S_test_score, S_test_acc, S_test_results, S_y_prob_0, S_y_prob_1 = training(X, y, clf, skf, 
                                                                                               aug=augmentation)

    ##
    # RF

    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', RandomForestClassifier(random_state=0))])

    rf_parameters = {'model__n_estimators': [100, 200, 500]}
    rf_grid = GridSearchCV(pipe, param_grid=rf_parameters, cv=cv)
    rf_grid.fit(X, y)
    print(rf_grid.best_params_)
    rf_best = rf_grid.best_params_

    logfile.write('Random Forest\n')
    logfile.write('Best n. of trees:' + str(rf_best['model__n_estimators']) + '\n\n')
    clf = RandomForestClassifier(random_state=0, n_estimators=rf_best['model__n_estimators'])
    rf_train_score, rf_test_score, rf_test_acc, rf_test_results, rf_y_prob_0, rf_y_prob_1 = training(X, y, clf, skf, 
                                                                                                     aug=augmentation)

    ##
    # Knn

    pipe = Pipeline(steps=[('preprocessing', StandardScaler()), ('model', Knn())])

    knn_parameters = {'model__n_neighbors': [3, 5, 7]}
    knn_grid = GridSearchCV(pipe, param_grid=knn_parameters, cv=cv)
    knn_grid.fit(X, y)
    print(knn_grid.best_params_)
    knn_best = knn_grid.best_params_

    logfile.write('KnnN\n')
    logfile.write('Best K:' + str(knn_best['model__n_neighbors']) + '\n\n')

    clf = Knn(n_neighbors=knn_best['model__n_neighbors'])
    knn_train_score, knn_test_score, knn_test_acc, knn_test_results, \
    knn_y_prob_0, knn_y_prob_1 = training(X, y, clf, skf, aug=augmentation)

    # SAVE GRID SEARCH INFORMATION
    df = pd.DataFrame({'SVM (L)': linear_best, 'SVM (P)': poly_best,
                       'SVM (RBF)': rbf_best, 'KnnN': knn_best, 'RF:': rf_best})
    df.to_csv(saver_path + extractor + '_best_parameters_' + str(n_splits) + 'CV.csv')

    # ############
    #     PLOTS  #
    # ############

    # Confusion matrix

    L_df_cm = compute_CV_confusion_matrix(L_test_results)
    P_df_cm = compute_CV_confusion_matrix(P_test_results)
    G_df_cm = compute_CV_confusion_matrix(G_test_results)
    S_df_cm = compute_CV_confusion_matrix(S_test_results)
    knn_df_cm = compute_CV_confusion_matrix(knn_test_results)
    rf_df_cm = compute_CV_confusion_matrix(rf_test_results)

    cm_sum(L_df_cm, labels, saver_path, 'Linear_SVC')
    cm_sum(P_df_cm, labels, saver_path, 'Poly_SVC')
    cm_sum(G_df_cm, labels, saver_path, 'RBF_SVC')
    cm_sum(S_df_cm, labels, saver_path, 'Sigmoid_SVC')
    cm_sum(knn_df_cm, labels, saver_path, 'KnnN')
    cm_sum(rf_df_cm, labels, saver_path, 'RandomForest')

    # Complete evaluation

    L_df = complete_evaluation(L_test_results, L_y_prob_0, L_y_prob_1)
    P_df = complete_evaluation(P_test_results, P_y_prob_0, P_y_prob_1)
    G_df = complete_evaluation(G_test_results, G_y_prob_0, G_y_prob_1)
    S_df = complete_evaluation(S_test_results, S_y_prob_0, S_y_prob_1)
    knn_df = complete_evaluation(knn_test_results, knn_y_prob_0, knn_y_prob_1)
    rf_df = complete_evaluation(rf_test_results, rf_y_prob_0, rf_y_prob_1)

    cl_mean(L_df, labels, saver_path, 'Linear_SVC')
    cl_mean(P_df, labels, saver_path, 'Poly_SVC')
    cl_mean(G_df, labels, saver_path, 'RBF_SVC')
    cl_mean(S_df, labels, saver_path, 'Sigmoid_SVC')
    cl_mean(knn_df, labels, saver_path, 'KnnN')
    cl_mean(rf_df, labels, saver_path, 'RandomForest')

    # SAVE RESULTS

    logfile = open(saver_path + 'all_results.txt', 'w')
    logfile.write('SVM results\n\n')

    for i in range(n_splits):
        logfile.write('Fold ' + str(i) + '\n')
        logfile.write('Linear SVM: Train={:.4f}, Test={:.4f}\n'.format(L_train_score[i], L_test_score[i]))
        logfile.write('Polynomial SVM: Train={:.4f}, Test={:.4f}\n'.format(P_train_score[i], P_test_score[i]))
        logfile.write('RBF SVM: Train={:.4f}, Test={:.4f}\n'.format(G_train_score[i], G_test_score[i]))
        logfile.write('Sigmoid SVM: Train={:.4f}, Test={:.4f}\n'.format(S_train_score[i], S_test_score[i]))
        logfile.write('Knn: Train={:.4f}, Test={:.4f}\n'.format(knn_train_score[i], knn_test_score[i]))
        logfile.write('RandomForest: Train={:.4f}, Test={:.4f}\n\n'.format(rf_train_score[i], rf_test_score[i]))

    logfile.flush()

    logfile.write('Average results (F1-score)\n')
    logfile.write('Linear SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
                  .format(np.mean(L_train_score), np.std(L_train_score), np.mean(L_test_score), np.std(L_test_score),
                          np.mean(L_test_acc), np.std(L_test_acc)))
    logfile.write('Polynomial SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
                  .format(np.mean(P_train_score), np.std(P_train_score), np.mean(P_test_score), np.std(P_test_score),
                          np.mean(P_test_acc), np.std(P_test_acc)))
    logfile.write('RBF SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
                  .format(np.mean(G_train_score), np.std(G_train_score), np.mean(G_test_score), np.std(G_test_score),
                          np.mean(G_test_acc), np.std(G_test_acc)))
    logfile.write('Sigmoid SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
                  .format(np.mean(S_train_score), np.std(S_train_score), np.mean(S_test_score), np.std(S_test_score),
                          np.mean(S_test_acc), np.std(S_test_acc)))
    logfile.write('Knn (k=7): Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
                  .format(np.mean(knn_train_score), np.std(knn_train_score), np.mean(knn_test_score),
                          np.std(knn_test_score), np.mean(knn_test_acc), np.std(knn_test_acc)))
    logfile.write('RandomForest: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
                  .format(np.mean(rf_train_score), np.std(rf_train_score), np.mean(rf_test_score),
                          np.std(rf_test_score), np.mean(rf_test_acc), np.std(rf_test_acc)))
    logfile.flush()
    logfile.close()

    print('Average (F1 score)\n')
    print('Linear SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
          .format(np.mean(L_train_score), np.std(L_train_score), np.mean(L_test_score), np.std(L_test_score),
                  np.mean(L_test_acc), np.std(L_test_acc)))
    print('Polynomial SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
          .format(np.mean(P_train_score), np.std(P_train_score), np.mean(P_test_score), np.std(P_test_score),
                  np.mean(P_test_acc), np.std(P_test_acc)))
    print('RBF SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
          .format(np.mean(G_train_score), np.std(G_train_score), np.mean(G_test_score), np.std(G_test_score),
                  np.mean(G_test_acc), np.std(G_test_acc)))
    print('Sigmoid SVM: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
          .format(np.mean(S_train_score), np.std(S_train_score), np.mean(S_test_score), np.std(S_test_score),
                  np.mean(S_test_acc), np.std(S_test_acc)))
    print('Knn: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
          .format(np.mean(knn_train_score), np.std(knn_train_score), np.mean(knn_test_score),
                  np.std(knn_test_score), np.mean(knn_test_acc), np.std(knn_test_acc)))
    print('RandomForest: Train={:.4f}±{:.4f}, Test={:.4f}±{:.4f}, Test acc={:.4f}±{:.4f}\n'
          .format(np.mean(rf_train_score), np.std(rf_train_score), np.mean(rf_test_score),
                  np.std(rf_test_score), np.mean(rf_test_acc), np.std(rf_test_acc)))

    np.save(saver_path + 'LinearSVM_classification_f1_train.npy', L_train_score)
    np.save(saver_path + 'LinearSVM_classification_f1_test.npy', L_test_score)

    np.save(saver_path + 'PolySVM_classification_f1_train.npy', P_train_score)
    np.save(saver_path + 'PolySVM_classification_f1_test.npy', P_test_score)

    np.save(saver_path + 'RbfSVM_classification_f1_train.npy', G_train_score)
    np.save(saver_path + 'RbfSVM_classification_f1_test.npy', G_test_score)

    np.save(saver_path + 'SigmoidSVM_classification_f1_train.npy', S_train_score)
    np.save(saver_path + 'SigmoidSVM_classification_f1_test.npy', S_test_score)

    np.save(saver_path + 'Knn_classification_f1_train.npy', knn_train_score)
    np.save(saver_path + 'Knn_classification_f1_test.npy', knn_test_score)

    np.save(saver_path + 'RandomForest_classification_f1_train.npy', rf_train_score)
    np.save(saver_path + 'RandomForest_classification_f1_test.npy', rf_test_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Fine tuning of <model>.
    File from <data_dir> are loaded and used as input for the <model>. 
    Training and evaluation results are saved in
    <saver_dir>.""")
    parser.add_argument('--model', default='ResNet101', type=str,
                        help='The pre-trained model adopted as feature extractor: '
                             'ADnet, Resnet18, Resnet50, or ResNet101.')
    parser.add_argument('--feature_dir', required=True, type=str,
                        help='The directory that contains the folders with extracted features')
    parser.add_argument('--saver_dir', default=os.getcwd(), type=str,
                        help='The directory where to save the classification results')
    parser.add_argument('--augmentation', default=False, type=bool,
                        help='If True, SMOTE augmentation is applied to the training data.')
    args = parser.parse_args()
    run_classifiers(args.model, args.feature_dir, args.saver_dir, args.augmentation)
