from typing import Union, Literal
import math
import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import mackinnonp

from pair_trading.numba_helpers.linalg import solve_numba, invert_numba, diff_numba


TREND_MAP = {
    'n': 0,
    'c': 1,
    'ct': 2,
    'ctt': 3,
}


@njit(inline='always', fastmath=True, cache=True)
def _default_max_lags(n_obs):
    return math.ceil(np.ceil(12 * (n_obs / 100) ** 0.25))


def _ols(y, X):
    XtX = X.T @ X
    Xty = X.T @ y

    beta = solve_numba(XtX, Xty)
    resid = y - X @ beta

    n, k = X.shape
    sigma2 = (resid.T @ resid) / (n - k)

    XtX_inv = invert_numba(XtX)
    var_beta = sigma2 * XtX_inv

    return beta.ravel(), resid.ravel(), sigma2.item(), var_beta


@njit(boundscheck=False, nogil=True, fastmath=True, cache=True)
def sum_of_squares(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i] ** 2
    return total


@njit(boundscheck=False, nogil=True, fastmath=True, cache=True)
def build_adf_matrix(y, lags, trend_index):
    n = y.shape[0] - 1
    delta_y = diff_numba(y)

    x_parts = np.empty((n, trend_index + lags + 1), dtype=y.dtype)

    if trend_index > 0:
        for i in range(n):
            x_parts[i, 0] = 1.0

    if trend_index > 1:
        for i in range(n):
            x_parts[i, 1] = i

    if trend_index > 2:
        for i in range(n):
            x_parts[i, 2] = i * i

    for i in range(n):
        x_parts[i, trend_index] = y[i]

    for L in range(1, lags + 1):
        for i in range(n):
            x_parts[i, trend_index + L] = delta_y[i - L] if i - L >= 0 else 0.0

    return x_parts, delta_y


def _information_criteria(rss, nobs, k, method="aic"):
    llf = -0.5 * nobs * (
        np.log(2 * np.pi)
        + np.log(rss / nobs)
        + 1
    )

    if method == "aic":
        ic = -2 * llf + 2 * k
    elif method == "bic":
        ic = -2 * llf + np.log(nobs) * k
    else:
        raise ValueError(
            "Invalid method for information criteria. Use 'aic' or 'bic'."
        )
    return ic


def adf(
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    lags: int = None,
    trend: Literal['n', 'c', 'ct', 'ctt'] = 'c',
    max_lags: int = None,
    method: Literal['aic', 'bic'] = 'aic',
):
    """
    Augmented Dickey-Fuller test (fixed lag version, from scratch).
    """
    gamma_idx = TREND_MAP[trend]

    if lags is not None:
        X, dy_t = build_adf_matrix(y, lags, gamma_idx)
        X, dy_t = X[lags:, :], dy_t[lags:]
        beta, resid, sigma2, var_beta = _ols(dy_t, X)

        gamma = beta[gamma_idx]
        se_gamma = np.sqrt(var_beta[gamma_idx, gamma_idx])
        t_stat = gamma / se_gamma

        ic = _information_criteria(
            sum_of_squares(resid), len(dy_t), X.shape[1], method=method
        )

        return t_stat, mackinnonp(t_stat, regression=trend), lags, ic

    if max_lags is None and lags is None:
        max_lags = _default_max_lags(len(y))

    X, dy_t = build_adf_matrix(y, max_lags, gamma_idx)
    best_ic, best_stat, best_lags = np.inf, None, None

    for lags in range(max_lags):
        dy_t_lags = dy_t[lags:]
        X_lags = X[lags:, :gamma_idx + 1 + lags]
        beta, resid, sigma2, var_beta = _ols(dy_t_lags, X_lags)

        gamma = beta[gamma_idx]
        se_gamma = np.sqrt(var_beta[gamma_idx, gamma_idx])

        t_stat = gamma / se_gamma

        nobs = len(dy_t_lags)
        k = X_lags.shape[1]
        ic = _information_criteria(
            sum_of_squares(resid), nobs, k, method=method,
        )

        if ic < best_ic:
            best_ic = ic
            best_stat = t_stat
            best_lags = lags

    return (
        best_stat,
        mackinnonp(best_stat, regression=trend),
        best_lags,
        best_ic,
    )
