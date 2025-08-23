import numpy as np
import cooler
from scipy.stats import wasserstein_distance, wasserstein_distance_nd

from typing import Iterator, Literal

def sample_counts(
        clr_file: cooler.Cooler, 
        keep_rate: float, 
        random_state=42,
        return_strategy: Literal['total', 'per_chromosome']='total'
    ) -> Iterator[np.ndarray]:
    if return_strategy == 'per_chromosome':
        for i in clr_file.chromnames:
            rng = np.random.default_rng(random_state)
            yield rng.binomial(n=clr_file.matrix(balance=False).fetch(i)[:], p=keep_rate), i
    elif return_strategy == 'total':
        rng = np.random.default_rng(random_state)
        yield rng.binomial(n=clr_file.matrix(balance=False).fetch()[:], p=keep_rate), '__total__'
    else:
        raise ValueError("return_strategy must be 'total' or 'per_chromosome'")

def dist_per_columns(orig: np.ndarray, sub: np.ndarray):
    distances_w = []
    distances_mse = []
    for j in range(orig.shape[1]):
        o = orig[:, j]
        s = sub[:, j]
        distances_w.append(wasserstein_distance(o, s))
        distances_mse.append(np.mean((o - s) ** 2))
    return distances_w, distances_mse

def dist_per_diagonals(orig: np.ndarray, sub: np.ndarray):
    distances = {}
    mses = {}
    n, m = orig.shape
    for offset in range(0, n):
        o_diag = np.diagonal(orig, offset)
        s_diag = np.diagonal(sub, offset)
        distances[offset] = wasserstein_distance(o_diag, s_diag)
        mses[offset] = np.mean((o_diag - s_diag) ** 2)
    return distances, mses


def compute_diff_data(
        cooler_orig,
        attemts=100,
        keep_rates=(_/100 for _ in range(1, 100)),
        mode: Literal['total', 'per_chromosome']='total',
        alias:str='some_alias'
):
    for attempt_number in range(attemts):
        for keep_rate in keep_rates:
            for matrix, name in sample_counts(cooler_orig, keep_rate, random_state=attempt_number, return_strategy=mode):
                if name == '__total__':
                    old_matrix = cooler_orig.matrix(balance=False).fetch()[:]
                else:
                    old_matrix = cooler_orig.matrix(balance=False).fetch(name)[:]
                metrics_dict = {
                    'attempt': attempt_number,
                    'keep_rate': keep_rate,
                    'name': name,
                    'alias': alias
                }
                metrics_dict['total_wass'] = wasserstein_distance_nd(matrix, old_matrix)
                metrics_dict['wass_diag'], metrics_dict['mse_diag'] = dist_per_diagonals(matrix, old_matrix)
                metrics_dict['wass_columns'], metrics_dict['mse_diag'] = dist_per_columns(matrix, old_matrix)
                yield metrics_dict