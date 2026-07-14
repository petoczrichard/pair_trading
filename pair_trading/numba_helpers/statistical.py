import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def numba_sum(x):
    n = x.shape[0]
    sum_x = 0.0

    for i in range(n):
        sum_x += x[i]

    return sum_x


@njit(cache=True)
def numba_mean(x):
    n = x.shape[0]
    return numba_sum(x) / n


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def numba_std(x, ddof=0):
    n = x.shape[0]
    mean_x = numba_mean(x)

    var_x = 0.0

    for i in range(n):
        dx = x[i] - mean_x
        var_x += dx * dx

    return np.sqrt(var_x / (n - ddof))


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def numba_create_spread(x, y, hedge_ratio):
    n = x.shape[0]
    result = np.empty_like(x)

    for i in range(n):
        result[i] = x[i] - hedge_ratio * y[i]

    return result


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def numba_covariance(x, y):
    n = x.shape[0]
    inv_n = 1.0 / n

    mean_x = 0.0
    mean_y = 0.0

    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]

    mean_x *= inv_n
    mean_y *= inv_n

    cov = 0.0
    var_x = 0.0
    var_y = 0.0

    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y

        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    return cov * inv_n, var_x * inv_n, var_y * inv_n, mean_x, mean_y


def pearson_correlation(x, y):
    cov, var_x, var_y, _, _ = numba_covariance(x, y)
    return cov / np.sqrt(var_x * var_y)


def rma_regression(x, y):
    n = x.shape[0]

    cov, var_x, var_y, mean_x, mean_y = numba_covariance(x, y)

    std_x = np.sqrt(var_x / (n - 1))
    std_y = np.sqrt(var_y / (n - 1))

    slope = (1 if cov > 0 else -1) * std_y / std_x
    intercept = mean_y - slope * mean_x

    return slope, intercept
