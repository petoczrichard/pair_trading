"""
trading engine
transonction cost model interfaces
portfolio analyser calc
"""

import numpy as np
from numba import njit


@njit(boundscheck=False, cache=True)
def compute_positions(
    spread,
    mean,
    std,
    long_entry: float = -2,
    long_exit: float = 0,
    long_stoploss: float = -5,
    short_entry: float = 2,
    short_exit: float = 0,
    short_stoploss: float = 5,
):

    long_entry = mean + long_entry * std
    long_exit = mean + long_exit * std
    long_stoploss = mean + long_stoploss * std

    short_entry = mean + short_entry * std
    short_exit = mean + short_exit * std
    short_stoploss = mean + short_stoploss * std

    n = spread.shape[0]
    positions = np.empty(n, dtype=np.int8)

    pos = 0

    for i in range(n):
        s = spread[i]

        if s < long_stoploss or s > short_stoploss:
            pos = 0
            positions[i] = 0
            break

        elif pos == 1 and s > long_exit:
            pos = 0
        elif pos == -1 and s < short_exit:
            pos = 0

        if pos == 0:
            if s < long_entry:
                pos = 1
            elif s > short_entry:
                pos = -1

        positions[i] = pos

    for i in range(i + 1, n):
        positions[i] = 0

    return positions


@njit(fastmath=True, boundscheck=False, nogil=True, cache=True)
def get_trade_dates(positions, dates):
    n = positions.shape[0]

    exits, entries = [], []

    for i in range(1, n):
        prev_pos = positions[i - 1]
        curr_pos = positions[i]

        if curr_pos != prev_pos and prev_pos != 0:
            exits.append(dates[i])

        if curr_pos != prev_pos and curr_pos != 0:
            entries.append(dates[i])

    return entries, exits
