import tensorflow as tf
import keras.backend as K
import numpy as np 
from scipy import ndimage
import sklearn.metrics

class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='iou', threshold=0.5, from_logits=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.math.sigmoid(y_pred)

        y_pred_bin = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        inter = tf.reduce_sum(y_true * y_pred_bin)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - inter

        self.intersection.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return tf.math.divide_no_nan(self.intersection, self.union + tf.keras.backend.epsilon())

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1', threshold=0.5, from_logits=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.math.sigmoid(y_pred)

        y_pred_bin = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred_bin)
        fp = tf.reduce_sum((1 - y_true) * y_pred_bin)
        fn = tf.reduce_sum(y_true * (1 - y_pred_bin))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp + tf.keras.backend.epsilon())
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn + tf.keras.backend.epsilon())
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

class AUCMetric(tf.keras.metrics.AUC):
    def __init__(self, from_logits=True, name='roc_auc', **kwargs):
        super().__init__(name=name, from_logits=from_logits, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = tf.math.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)

class PRAUCMetric(tf.keras.metrics.AUC):
    def __init__(self, from_logits=True, name='pr_auc', **kwargs):
        super().__init__(name=name, curve='PR', from_logits=from_logits, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            y_pred = tf.math.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    min_val = np.min(array)
    max_val = np.max(array)
    
    if max_val == min_val:
        return np.zeros_like(array)  
    
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


def AULC(uncs, error, eps=1e-12):
    idxs = np.argsort(uncs)
    error_s = error[idxs]
    mean_error = error_s.mean()

    if np.all(error_s < eps):
        return 1.0

    error_csum = np.cumsum(error_s)
    Fs = error_csum / (np.arange(1, len(error_s) + 1) + eps)
    Fs = mean_error / (Fs + eps)
    s = 1.0 / len(Fs)
    return -1 + s * Fs.sum()

def rAULC(uncs, error, eps=1e-12):
    perf_aulc = AULC(error, error, eps)
    curr_aulc = AULC(uncs, error, eps)
    return curr_aulc / (perf_aulc + eps)

def corr(uncs, error): 
    if np.std(uncs) == 0 or np.std(error) == 0:
        return 0.0
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
    labels = np.asarray(labels).astype(np.uint8)
    uncertainties = np.asarray(uncertainties).astype(np.float32)
    return sklearn.metrics.roc_auc_score(labels, uncertainties)


def cal_pr_auc(labels, uncertainties):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, uncertainties)
    pr_auc = sklearn.metrics.auc(recall, precision)
    return pr_auc

def compute_ece(unc, mse, n_bins=40):
    """
    Compute the Expected Calibration Error for a regression task.

    Args:
    unc (numpy.array): Array of predicted uncertainties.
    mse (numpy.array): Array of mean squared errors.
    n_bins (int): Number of bins to use for calibration.

    Returns:
    float: The ECE value.
    """

    # Ensure unc and mse are numpy arrays
    unc = np.array(unc)
    mse = np.array(mse)

    # Create bins based on the predicted uncertainties
    bin_edges = np.linspace(0, np.max(unc), n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices of samples in each bin
        in_bin = (unc >= bin_lower) & (unc < bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            # Average uncertainty in bin
            avg_uncertainty = np.mean(unc[in_bin])
            # Average error in bin
            avg_error = np.mean(mse[in_bin])
            # Weighted absolute difference
            ece += np.abs(avg_uncertainty - avg_error) * (bin_count / len(unc))

    return ece