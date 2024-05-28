import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc, precision_recall_curve
import prettytable as pt
import torch


def eval_metrics(probs, targets, cal_AUC=True, cal_PRC=True):
    threshold_list = []
    for i in range(1, 50):
        threshold_list.append(i / 50.0)

    if cal_AUC:
        if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
            fpr, tpr, thresholds = roc_curve(y_true=targets.detach().cpu().numpy(),
                                             y_score=probs.detach().cpu().numpy())
        elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
            fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
        else:
            print('ERROR: probs or targets type is error.')
            raise TypeError
        roc_auc = auc(x=fpr, y=tpr)
    else:
        roc_auc = 0
    if cal_PRC:
        if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
            pre, rec, _ = precision_recall_curve(y_true=targets.detach().cpu().numpy(),
                                                 probas_pred=probs.detach().cpu().numpy())
        elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
            pre, rec, _ = precision_recall_curve(y_true=targets, probas_pred=probs)
        else:
            print('ERROR: probs or targets type is error.')
            raise TypeError
        prc_auc = auc(rec, pre)
    else:
        prc_auc = 0

    threshold_best, rec_best, pre_best, F1_best, spe_best, mcc_best, pred_bi_best = 0, 0, 0, 0, 0, -1, None
    for threshold in threshold_list:
        threshold, acc, rec, pre, F1, spe, mcc, _, _, pred_bi = th_eval_metrics(threshold, probs, targets,
                                                                                cal_AUC=False)
        if F1 > F1_best:
            threshold_best, acc_best, rec_best, pre_best, F1_best, spe_best, mcc_best, pred_bi_best = threshold, acc, rec, pre, F1, spe, mcc, pred_bi

    return threshold_best, acc_best, rec_best, pre_best, F1_best, spe_best, mcc_best, roc_auc, prc_auc, pred_bi_best


def th_eval_metrics(threshold, probs, targets, cal_AUC=True, cal_PRC=True):
    if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
        targets, probs = targets.detach().cpu().numpy(), probs.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
        auc_ = auc(x=fpr, y=tpr)

        pre, rec, _ = precision_recall_curve(y_true=targets, probas_pred=probs)
        prc_ = auc(rec, pre)

        pred_bi = np.abs(np.ceil(probs - threshold))

        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
        if tp > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if tp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 0
        if tn > 0:
            spe = tn / (tn + fp)
        else:
            spe = 1e-8

        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = matthews_corrcoef(targets, pred_bi)
        if rec + pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0

    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=probs)
        auc_ = auc(x=fpr, y=tpr)
        pre, rec, _ = precision_recall_curve(y_true=targets, probas_pred=probs)
        prc_ = auc(rec, pre)

        pred_bi = np.abs(np.ceil(probs - threshold))

        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
        if tp > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if tp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 0
        if tn > 0:
            spe = tn / (tn + fp)
        else:
            spe = 1e-8
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = matthews_corrcoef(targets, pred_bi)
        if rec + pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError

    return threshold, acc, rec, pre, F1, spe, mcc, auc_, prc_, pred_bi


def CFM_eval_metrics(CFM):
    CFM = CFM.astype(float)
    tn = CFM[0, 0]
    fp = CFM[0, 1]
    fn = CFM[1, 0]
    tp = CFM[1, 1]

    acc = (tp + tn) / (tn + tp + fp + fn)
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 0
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        mcc = -1
    return acc, rec, pre, F1, spe, mcc


def print_results(valid_matrices=None, test_matrices=None):
    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'th', 'Acc', 'Rec', 'Pre', 'F1', 'Spe', 'MCC', 'AUC', 'PRC']

    if valid_matrices is not None:
        row_list = ['valid']
        for i in range(len(tb.field_names) - 1):
            row_list.append('{:.3f}'.format(valid_matrices[i]))
        tb.add_row(row_list)

    if test_matrices is not None:
        row_list = ['test']
        for i in range(len(tb.field_names) - 1):
            row_list.append('{:.3f}'.format(test_matrices[i]))
        tb.add_row(row_list)
    print(tb)


def print_seq_results(seq_list, matrices_list):
    tb = pt.PrettyTable()
    tb.field_names = ['seq_ID', 'th', 'Rec', 'Pre', 'F1', 'Spe', 'MCC', 'AUC', 'PRC']
    tb.float_format = ".3"

    for seq, matrices in zip(seq_list, matrices_list):
        row_list = [seq]
        for i in range(8):
            row_list.append(matrices[i])
        tb.add_row(row_list)
    print(tb)
