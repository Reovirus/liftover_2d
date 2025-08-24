import numpy as np
from typing import List, Literal

from tests.benchmark.dist_utils.compute_all_dist import (
    kl_divergence_vec,
    wasserstein_vec,
    mse_vec,
    euclidean_vec,
    frobenius_mat,
    wasserstein_mat,
    kl_divergence_mat,
    mse_mat
)

__DICT_METRICS_VEC = {
    'kl': kl_divergence_vec,
    'wasserstein': wasserstein_vec,
    'mse': mse_vec,
    'frobenius': euclidean_vec
}

__DICT_METRICS_MAT = {
    'frobenius': frobenius_mat,
    'wasserstein': wasserstein_mat,
    'kl': kl_divergence_mat,
    'mse': mse_mat
}

def compute_diags(
        old: np.ndarray, 
        new: np.ndarray, 
        metric_list: List[Literal['kl', 'wasserstein', 'mse', 'frobenius']] = ['kl', 'mse']
    ):
    n = old.shape[0]
    out = {metric: np.empty(n, dtype=np.float64) for metric in metric_list}

    for offset in range(n):
        old_diag = np.diagonal(old, offset)
        new_diag = np.diagonal(new, offset)
        if 'kl' in metric_list:
            out['kl'][offset] = kl_divergence_vec(old_diag, new_diag)
        if 'wasserstein' in metric_list:
            out['wasserstein'][offset] = wasserstein_vec(old_diag, new_diag)
        if 'mse' in metric_list:
            out['mse'][offset] = mse_vec(old_diag, new_diag)
        if 'frobenius' in metric_list:
            out['frobenius'][offset] = euclidean_vec(old_diag, new_diag)
    return out

def compute_cols(
        old: np.ndarray, 
        new: np.ndarray, 
        metric_list: List[Literal['kl', 'wasserstein', 'mse', 'frobenius']] = ['kl', 'mse']
    ):
    n = old.shape[0]
    out = {metric: np.empty(n, dtype=np.float64) for metric in metric_list}

    for offset in range(n):
        old_col = old[:, offset]
        new_col = new[:, offset]
        if 'kl' in metric_list:
            out['kl'][offset] = kl_divergence_vec(old_col, new_col)
        if 'wasserstein' in metric_list:
            out['wasserstein'][offset] = wasserstein_vec(old_col, new_col)
        if 'mse' in metric_list:
            out['mse'][offset] = mse_vec(old_col, new_col)
        if 'frobenius' in metric_list:
            out['frobenius'][offset] = euclidean_vec(old_col, new_col)
    return out

def compute_mats(
        old: np.ndarray, 
        new: np.ndarray, 
        metric_list: List[Literal['frobenius', 'wasserstein', 'kl', 'mse']] = ['frobenius', 'mse']
    ):
    out = {}
    for metric in metric_list:
        out[metric] = __DICT_METRICS_MAT[metric](old, new)
    return out
