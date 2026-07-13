from typing import Literal
import math
import numpy as np
from numba import njit

from pair_trading.numba_helpers.linalg import slogdet_numba, lstsq_numba


TREND_MAP = {
    'n': 0,
    'c': 1,
    'ct': 2,
    'ctt': 3,
}

IC_MAP = {
    'aic': 0,
    'bic': 1,
    'hqic': 2,
}


@njit(inline='always', fastmath=True, cache=True)
def _default_max_lags(n_obs):
    return math.ceil(np.ceil(12 * (n_obs / 100) ** 0.25))


@njit(inline='always', fastmath=True, cache=True)
def _compute_ic(T, n_params, log_det, ic_type):
    inv_T = 1.0 / T

    if ic_type == 0:
        return log_det + 2.0 * n_params * inv_T

    elif ic_type == 1:
        return log_det + np.log(T) * n_params * inv_T

    elif ic_type == 2:
        return log_det + 2.0 * np.log(np.log(T)) * n_params * inv_T

    else:
        raise ValueError("Invalid IC type.")


@njit(boundscheck=False, nogil=True, fastmath=True, cache=True)
def _create_var_matrix(
    data: np.ndarray,
    trend_index: int,
    max_lags: int | None = None,
):
    n_obs, n_vars = data.shape

    out = np.empty((n_obs, trend_index + n_vars * max_lags))

    if trend_index > 0:
        for i in range(n_obs):
            out[i, 0] = 1

    if trend_index > 1:
        for i in range(n_obs):
            out[i, 1] = i

    if trend_index > 2:
        for i in range(n_obs):
            out[i, 2] = i * i

    for lag in range(1, max_lags + 1):
        for n_var in range(n_vars):
            index_col = trend_index + n_var + n_vars * (lag - 1)

            for i in range(lag):
                out[i, index_col] = 0.0

            for i in range(lag, n_obs):
                out[i, index_col] = data[i - lag, n_var]

    return out


def var(
    data: np.ndarray,
    lags: int = None,
    trend: Literal['n', 'c', 'ct', 'ctt'] = 'c',
    max_lags: int = None,
    ic_method: Literal['aic', 'bic', 'hqic'] = 'aic',
):
    trend_index = TREND_MAP[trend]
    ic_index = IC_MAP[ic_method]

    if lags is not None:
        matrix = _create_var_matrix(data, trend_index, lags)
        lagged_matrix = matrix[lags:]
        lagged_data = data[lags:]

        beta = lstsq_numba(lagged_matrix, lagged_data)
        resid = lagged_data - lagged_matrix @ beta

        return resid, lags, None

    if max_lags is None and lags is None:
        max_lags = _default_max_lags(data.shape[0])

    lagged_matrix = _create_var_matrix(data, trend_index, max_lags)

    best_ic = np.inf
    best_resid = None
    best_lag = None

    for lag in range(max_lags):
        current_matrix = lagged_matrix[
            lag:,
            :trend_index + lag * data.shape[1],
        ]
        current_data = data[lag:, :]

        T = current_matrix.shape[0]
        K = current_data.shape[1]

        beta = lstsq_numba(current_matrix, current_data)
        resid = current_data - current_matrix @ beta

        sigma_mle = (resid.T @ resid) / T

        n_params = K * (lag * K + trend_index)
        log_det = slogdet_numba(sigma_mle)[1]
        ic = _compute_ic(T, n_params, log_det, ic_index)

        if ic < best_ic:
            best_ic = ic
            best_resid = resid
            best_lag = lag

    return best_resid, best_lag, best_ic
