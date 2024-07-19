from statistics import mean
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, \
    precision_score, recall_score, roc_auc_score, auc, precision_recall_curve, confusion_matrix
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

orig_cmap = plt.cm.Blues
colors = orig_cmap(np.linspace(0.25, 1))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors)

# ######################
#     PLOTTING       ##
# ######################


def compute_CV_confusion_matrix(test_results):
    all_tn, all_fp, all_fn, all_tp = [], [], [], []
    for y_test, y_pred in test_results:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        all_tn.append(tn)
        all_fp.append(fp)
        all_tp.append(tp)
        all_fn.append(fn)
    df = pd.DataFrame({'True Positive': all_tp, 'True Negative': all_tn, 'False Positive': all_fp, 'False Negative': all_fn})
    return df


def complete_evaluation(test_results, y_prob_0, y_prob_1):
    precision_0, recall_0, precision_1, recall_1 = [], [], [], []
    f1_0, f1_1, roc_auc_0, roc_auc_1, auprc_0, auprc_1 = [], [], [], [], [], []
    for i in range(len(test_results)):
        y_test, y_pred = test_results[i]
        yp0 = y_prob_0[i]
        yp1 = y_prob_1[i]
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        recall_0.append(recall_score(y_test, y_pred, pos_label=0))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        recall_1.append(recall_score(y_test, y_pred, pos_label=1))
        f1_0.append(f1_score(y_test, y_pred, pos_label=0))
        f1_1.append(f1_score(y_test, y_pred, pos_label=1))
        roc_auc_0.append(roc_auc_score(y_test, yp0))
        roc_auc_1.append(roc_auc_score(y_test, yp1))
        prec_0, rec_0, _ = precision_recall_curve(y_test, yp0, pos_label=0)
        prec_1, rec_1, _ = precision_recall_curve(y_test, yp1)
        auprc_0.append(auc(rec_0, prec_0))
        auprc_1.append(auc(rec_1, prec_1))

    df = pd.DataFrame({'Precision Positive Class': precision_1, 'Precision Negative Class': precision_0,
                       'Recall Positive Class': recall_1, 'Recall Negative Class': recall_0,
                       'F1-score Negative Class': f1_0, 'F1-score Positive Class': f1_1,
                       'AUC Positive Class': roc_auc_1, 'AUC Negative Class': roc_auc_0,
                       'AUPRC Positive Class': auprc_1, 'AUPRC Negative Class': auprc_0})
    return df


def cl_mean(d, label, saver_path, name, outcome='save'):
    mp1 = round(mean(d['Precision Positive Class']), 2)
    p1 = f'{mp1}'
    mp0 = round(mean(d['Precision Negative Class']), 2)
    p0 = f'{mp0}'
    mr1 = round(mean(d['Recall Positive Class']), 2)
    r1 = f'{mr1}'
    mr0 = round(mean(d['Recall Negative Class']), 2)
    r0 = f'{mr0}'
    mf1 = round(mean(d['F1-score Positive Class']), 2)
    f1 = f'{mf1}'
    mf0 = round(mean(d['F1-score Negative Class']), 2)
    f0 = f'{mf0}'
    ma1 = round(mean(d['AUC Positive Class']), 2)
    a1 = f'{ma1}'
    ma0 = round(mean(d['AUC Negative Class']), 2)
    a0 = f'{ma0}'
    map1 = round(mean(d['AUPRC Positive Class']), 2)
    ap1 = f'{map1}'
    map0 = round(mean(d['AUPRC Negative Class']), 2)
    ap0 = f'{map0}'
    data = pd.DataFrame({'Class': label, 'Precision': [mp0, mp1],
                         'Recall': [mr0, mr1], 'F1-score': [mf0, mf1],
                         'AUC': [ma0, ma1], 'AUPRC': [map0, map1]})
    data = data.set_index('Class')
    lab = pd.DataFrame({'Precision': [p0, p1], 'Recall': [r0, r1],
                        'F1-score': [f0, f1], 'AUC': [a0, a1], 'AUPRC': [ap0, ap1]})
    plt.figure(figsize=(10, 3.5))
    ax = sns.heatmap(data, annot=lab, fmt='', square=True, cmap=cmap)
    ax.set(ylabel='')
    plt.title('Classification Report')
    if outcome == 'show':
        plt.show()
    else:
        plt.savefig(f'{saver_path}/{name}_cl.png', bbox_inches='tight')
        plt.close()


def cm_sum(d, label, save_path, name, title='Confusion Matrix', outcome='save'):
    plt.figure()
    mtp = sum(d['True Positive'])
    tp = f'{mtp}'
    mtn = sum(d['True Negative'])
    tn = f'{mtn}'
    mfp = sum(d['False Positive'])
    fp = f'{mfp}'
    mfn = sum(d['False Negative'])
    fn = f'{mfn}'
    data = pd.DataFrame({'True Labels': label, f'{label[0]}': [mtn, mfn],
                         f'{label[1]}': [mfp, mtp]})
    data = data.set_index('True Labels')
    lab = pd.DataFrame({'column1': [tn, fn], 'column2': [fp, tp]})
    ax = sns.heatmap(data, annot=lab, fmt='', square=True, cmap=cmap, cbar=False)
    sns.set(font_scale=2)
    ax.set(xlabel='Predicted Labels')
    plt.title(title)
    plt.subplots_adjust(left=0.1,
                        bottom=0.2,
                        right=0.9,
                        top=0.9)
    if outcome == 'show':
        plt.show()
    else:
        plt.savefig(f'{save_path}/{name}_cm_sum.png', bbox_inches='tight')
        plt.close()
