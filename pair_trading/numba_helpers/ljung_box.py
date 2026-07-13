import numpy as np
from numba import njit

from pair_trading.numba_helpers.chi2 import chi2_cdf_scalar
from pair_trading.numba_helpers.statistical import numba_mean


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def ljung_box(x, lags):
    n = x.shape[0]
    r = np.empty(lags)
    q = np.empty(lags)
    p = np.empty(lags)

    x_mean = numba_mean(x)

    denom = 0
    for i in range(n):
        denom += (x[i] - x_mean) ** 2

    for k in range(1, lags + 1):
        idx = k - 1

        num = 0
        for i in range(k, n):
            num += (x[i] - x_mean) * (x[i - k] - x_mean)

        r_k = num / denom
        r[idx] = r_k

        s = 0.0
        for j in range(k):
            lag = j + 1
            rj = r[j]
            s += (rj * rj) / (n - lag)
        q[idx] = n * (n + 2) * s

        p[idx] = 1 - chi2_cdf_scalar(q[idx], df=k)

    return q, p
