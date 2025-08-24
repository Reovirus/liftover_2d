import numpy as np
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.linalg import svdvals

def euclidean_vec(u: np.ndarray, v: np.ndarray) -> float:
    return np.linalg.norm(u - v)

def kl_divergence_vec(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def wasserstein_vec(u: np.ndarray, v: np.ndarray) -> float:
    return wasserstein_distance(u, v)

def mse_vec(u: np.ndarray, v: np.ndarray) -> float:
    return np.mean((u - v)**2)

def frobenius_mat(A: np.ndarray, B: np.ndarray) -> float:
    return np.linalg.norm(A - B, 'fro')

def spectral_mat(A: np.ndarray, B: np.ndarray) -> float:
    return np.max(svdvals(A - B))

def wasserstein_mat(A: np.ndarray, B: np.ndarray) -> float:
    return wasserstein_distance_nd(A, B)

def kl_divergence_mat(A: np.ndarray, B: np.ndarray) -> float:
    p = np.abs(A.flatten())
    q = np.abs(B.flatten())
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return kl_divergence_vec(p, q)

def mse_mat(A: np.ndarray, B: np.ndarray) -> float:
    return np.mean((A-B)**2)
