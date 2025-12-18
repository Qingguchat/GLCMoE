
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    total = y_pred.size
    correct = w[row_ind, col_ind].sum()
    return correct / total


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return {'ACC': acc, 'NMI': nmi, 'ARI': ari}
