import numpy as np
from scipy import ndimage

def correctness(TP, FP, eps=1e-12):
    return TP / (TP + FP + eps)

def completeness(TP, FN, eps=1e-12):
    return TP / (TP + FN + eps)

def quality(TP, FP, FN, eps=1e-12):
    return TP / (TP + FP + FN + eps)

def f1_score(correctness_val, completeness_val, eps=1e-12):
    return 2.0 / (1.0 / (correctness_val + eps) + 1.0 / (completeness_val + eps))

def relaxed_confusion_matrix(pred_mask, gt_mask, slack=5):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    gt_d = ndimage.distance_transform_edt(~gt_mask)
    pred_d = ndimage.distance_transform_edt(~pred_mask)

    tp_pred = np.logical_and(pred_mask, gt_d <= slack)
    tp_gt   = np.logical_and(gt_mask, pred_d <= slack)

    TP = np.logical_or(tp_pred, tp_gt).sum()
    FP = np.logical_and(pred_mask, gt_d > slack).sum()
    FN = np.logical_and(gt_mask, pred_d > slack).sum()

    return TP, FP, FN

def compute_ccq(pred_score, gt_mask, threshold=0.5, slack=5):
    pred_mask = pred_score >= threshold
    gt_mask = gt_mask >= 0.5

    TP, FP, FN = relaxed_confusion_matrix(pred_mask, gt_mask, slack)
    corr = correctness(TP, FP)
    comp = completeness(TP, FN)
    qual = quality(TP, FP, FN)
    f1 = f1_score(corr, comp)
    return corr, comp, qual, f1

def compute_ccq_normal(pred_score, gt_mask, threshold=0.5):
    """
    Đánh giá chuẩn (không cho phép lệch)
    """
    pred_mask = (pred_score >= threshold).astype(bool)
    gt_mask = (gt_mask >= 0.5).astype(bool)

    TP = np.logical_and(pred_mask, gt_mask).sum()
    FP = np.logical_and(pred_mask, ~gt_mask).sum()
    FN = np.logical_and(~pred_mask, gt_mask).sum()

    corr = correctness(TP, FP)
    comp = completeness(TP, FN)
    qual = quality(TP, FP, FN)
    f1 = f1_score(corr, comp)
    return corr, comp, qual, f1
