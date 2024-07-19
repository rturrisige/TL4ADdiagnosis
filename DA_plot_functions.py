from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statistics import stdev
from numpy import average as avg
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, \
    precision_score, recall_score, roc_auc_score, auc, precision_recall_curve, confusion_matrix
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

orig_cmap = plt.cm.Blues
colors = orig_cmap(np.linspace(0.25, 1))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors)

# ######################
#     PLOTTING       ##
# ######################


def plot_complete_report(data, saver_path, labels=None):
    if labels:
        data['Class'] = labels
        data = data.set_index('Class')
    plt.figure(figsize=(10,3.5))
    ax = sns.heatmap(data, annot=data, fmt='.2f', square=True, cmap=cmap)
    ax.set(ylabel='Classes')
    plt.title('Classification Report')
    plt.savefig(f'{saver_path}/Evaluation.png')
    plt.close()


def plot_auc_curve(fpr, tpr, roc_auc_1, saver_path):
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label="AUC={:.4f}".format(roc_auc_1))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('AUC curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{saver_path}/AUC.png')
    plt.close()


def plot_confusion_matrix(y_test, y_pred, saver_path, labels=None):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'{saver_path}/confusion_matrix.png')
    plt.close()

