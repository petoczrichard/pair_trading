import numpy as np
from numba import njit


NEGATIVE_LOG_2 = -np.log(2.0)


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def half_life(ts):
    n = ts.shape[0] - 1

    sum_lag2 = 0.0
    sum_lag = 0.0
    sum_lag_delta = 0.0
    sum_delta = 0.0

    for i in range(n):
        lag_i = ts[i]
        delta_i = ts[i + 1] - lag_i

        sum_lag2 += lag_i * lag_i
        sum_lag += lag_i
        sum_lag_delta += lag_i * delta_i
        sum_delta += delta_i

    denom = n * sum_lag2 - sum_lag * sum_lag

    if denom == 0.0:
        return np.inf

    beta = (n * sum_lag_delta - sum_lag * sum_delta) / denom

    if beta == 0.0:
        return np.inf

    return NEGATIVE_LOG_2 / beta


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def number_of_zero_crossings(spread, mean):
    n = spread.shape[0]

    count = 0
    for i in range(1, n):
        prev = spread[i - 1] > mean
        current = spread[i] > mean
        count += current ^ prev

    return count


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def number_of_trades(positions):
    trade_count = 0
    n = positions.shape[0]

    for i in range(1, n):
        prev_pos = positions[i-1]
        trade_count += (prev_pos != 0.0) & (positions[i] != prev_pos)

    return trade_count


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def average_round_trip_length(positions):
    n = positions.shape[0]

    trade_count = 0
    total_trade_length = 0

    for i in range(1, n):
        prev = positions[i - 1]
        pos = positions[i]

        if pos != 0:
            total_trade_length += 1
        elif prev != 0:
            trade_count += 1

    if trade_count == 0:
        return 0.0

    for i in range(n - 1, 0, -1):
        if positions[i] != 0:
            total_trade_length -= 1
        else:
            break

    return total_trade_length / trade_count


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def ssd(price1, price2):
    n = price1.shape[0]

    inv1 = 1.0 / price1[0]
    inv2 = 1.0 / price2[0]

    ssd_ = 0.0

    for i in range(1, n):
        diff = price1[i] * inv1 - price2[i] * inv2
        ssd_ += diff * diff

    return ssd_
