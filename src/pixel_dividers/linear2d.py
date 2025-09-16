import numpy as np
from numba import njit

from src.pixel_dividers.base_divider import BaseNeigborUsingDivider

@njit(nopython=True, parallel=True)
def build_design_matrix_plane(n, counts):
    A = np.empty((n * n, 3), dtype=np.float64)
    b = np.empty(n * n, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(n):
            x = i + 0.5
            y = j + 0.5
            A[k, 0] = x
            A[k, 1] = y
            A[k, 2] = 1.0
            b[k] = counts[i, j]
            k += 1
    return A, b

@njit(nopython=True, parallel=True)
def build_design_matrix_log(n, counts):
    A = np.empty((n * n, 3), dtype=np.float64)
    b = np.empty(n * n, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(n):
            x = i + 0.5
            y = j + 0.5
            A[k, 0] = np.log(x)
            A[k, 1] = np.log(y)
            b[k] = np.log(counts[i, j])
            k += 1
    return A, b


class Regression2D:
    def __init__(self, is_log: bool = False):
        self.is_log = is_log
        self.__coeffs = None

    def fit(self, counts: np.ndarray):
        n, m = counts.shape
        assert n == m and n % 2 == 1

        if self.is_log:
            A, b = build_design_matrix_log(n, counts)
        else:
            A, b = build_design_matrix_plane(n, counts)

        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.__coeffs = coeffs

    def integrate(self, region):
        if self.__coeffs is None:
            raise ValueError("Call .fit() before integrate().")

        if self.is_log:
            alpha, beta, gamma = self.__coeffs
            C = np.exp(gamma)
            x0, x1 = region[0]
            y0, y1 = region[1]
            return (
                C
                * (x1 ** (alpha + 1) - x0 ** (alpha + 1)) / (alpha + 1)
                * (y1 ** (beta + 1) - y0 ** (beta + 1)) / (beta + 1)
            )
        else:
            a, b, c = self.__coeffs
            x0, x1 = region[0]
            y0, y1 = region[1]
            return (
                a * (x1**2 - x0**2) / 2 * (y1 - y0)
                + b * (y1**2 - y0**2) / 2 * (x1 - x0)
                + c * (x1 - x0) * (y1 - y0)
            )

class Linear2DDivider(BaseNeigborUsingDivider):
    METRIC_NAME='LINEAR_2D'
    NEED_NORMALISATION=False
    NEED_SAMPLING=True
    COMPUTE_COUNTS=False

    def __init__(self, mode='resample', ident: int=3, scale_factor: int=20, use_log: bool = False):
        self.__use_log = Regression2D(is_log=use_log)
        super().__init__(ident=ident, scale_factor=scale_factor, mode=mode)


    def _process_one_window(self, counts_arr, location):
        zoomed = np.clip(counts_arr, 0, None)
        s = zoomed.sum()
        if s > 0:
            zoomed /= s
        self._regression.fit(zoomed)
        