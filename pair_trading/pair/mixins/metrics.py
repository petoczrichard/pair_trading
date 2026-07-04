import numpy as np

from pair_trading.numba.metrics import (
    half_life as numba_half_life,
    number_of_trades as numba_number_of_trades,
    number_of_zero_crossings as numba_number_of_zero_crossings,
    average_round_trip_length as numba_average_round_trip_length,
)
from pair_trading.numba.statistical import pearson_correlation
from pair_trading.numba.backtest import compute_positions


class MetricsMixin:
    def half_life(self):
        return numba_half_life(
            ts=self.spread_values[self.formation_period_index],
        )

    def hurst_exponent(self, max_lag: int = 100):
        lags = np.arange(2, max_lag)
        formation_spread = self.spread_values[self.formation_period_index]

        diffs = np.array(
            [formation_spread[lag:] - formation_spread[:-lag] for lag in lags]
        )
        tau = np.sqrt(np.var(diffs, axis=1))
        slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)

        return 2.0 * slope

    def average_round_trip_length(
        self,
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
    ):
        positions = compute_positions(
            spread=self.spread_values[self.formation_period_index],
            mean=self.spread_mean,
            std=self.spread_std,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )
        return numba_average_round_trip_length(positions)

    def number_of_trades(
        self,
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
    ):
        positions = compute_positions(
            spread=self.spread_values[self.formation_period_index],
            mean=self.spread_mean,
            std=self.spread_std,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )
        return numba_number_of_trades(positions)

    def number_of_zero_crossings(self):
        return numba_number_of_zero_crossings(
            spread=self.spread_values[self.formation_period_index],
            mean=self.spread_mean,
        )

    def price_correlation(self):
        return pearson_correlation(
            x=self.price1_values[self.formation_period_index],
            y=self.price2_values[self.formation_period_index],
        )

    def return_correlation(self):
        prices1 = self.price1_values[self.formation_period_index]
        prices2 = self.price2_values[self.formation_period_index]
        returns1 = prices1[1:] / prices1[:-1] - 1
        returns2 = prices2[1:] / prices2[:-1] - 1

        return pearson_correlation(
            x=returns1,
            y=returns2,
        )
