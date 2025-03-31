import numpy as np
from scipy import ndimage
import sklearn.metrics

def min_max_normalize(array: np.ndarray) -> np.ndarray:
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val == min_val:
        return np.zeros_like(array)  # Tránh chia cho 0, mọi phần tử giống nhau → trả về 0
    
    return (array - min_val) / (max_val - min_val)

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


def AULC(uncs, error):
    idxs = np.argsort(uncs)
    uncs_s = uncs[idxs]
    error_s = error[idxs]
    mean_error = error_s.mean()
    error_csum = np.cumsum(error_s)
    Fs = error_csum / np.arange(1, len(error_s) + 1)
    Fs = mean_error / (Fs)
    s = 1 / len(Fs)
    return -1 + s * Fs.sum()

def rAULC(uncs, error):
    perf_aulc = AULC(error, error)
    curr_aulc = AULC(uncs, error)
    return curr_aulc / perf_aulc

def corr(uncs, error): 
    matrix = np.corrcoef(np.array(uncs), np.array(error))
    return matrix[0][1]

def split_and_mean(array, num_rows=2, num_cols=2):
    """
    Chia mảng 2D thành nhiều phần bằng nhau và tính mean của từng phần.

    Parameters:
        array (np.ndarray): Mảng đầu vào 2D.
        num_rows (int): Số hàng muốn chia.
        num_cols (int): Số cột muốn chia.

    Returns:
        crops (List[np.ndarray]): Danh sách các mảng con.
        means (List[float]): Danh sách các giá trị mean của từng crop.
    """
    h, w = array.shape
    assert h % num_rows == 0 and w % num_cols == 0, "Kích thước không chia hết!"

    crop_h = h // num_rows
    crop_w = w // num_cols

    means = []

    for i in range(num_rows):
        for j in range(num_cols):
            crop = array[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            means.append(np.mean(crop))

    return means

def get_uncertainty_by_var(list, axis, num_rows, num_cols): 
    matrix = np.var(list, axis=axis)
    return split_and_mean(matrix, num_rows, num_cols)

def get_uncertainty_by_std(list, axis, num_rows, num_cols): 
    matrix = np.std(list, axis=axis)
    return split_and_mean(matrix, num_rows, num_cols)

def get_error_by_abs(pred, mask, num_rows, num_cols): 
    matrix = np.abs(pred - mask)
    return split_and_mean(matrix, num_rows, num_cols)

def get_error_by_mse(pred, mask, num_rows, num_cols): 
    matrix = (pred - mask) ** 2
    return split_and_mean(matrix, num_rows, num_cols)

def cal_roc_auc(labels, uncertainties): 
    roc_auc = sklearn.metrics.roc_auc_score(labels, uncertainties) 
    return roc_auc

def cal_pr_auc(labels, uncertainties):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, uncertainties)
    pr_auc = sklearn.metrics.auc(recall, precision)
    return pr_auc