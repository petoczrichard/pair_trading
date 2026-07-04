from abc import ABC, abstractmethod

from pair_trading.numba.adf import adf
from pair_trading.catalog import PairTradingMeta


class AbstractDataCleaner(ABC, metaclass=PairTradingMeta):

    alias = 'data_cleaner'

    @abstractmethod
    def clean_tickers(self, metadata, **kwargs):
        pass

    @abstractmethod
    def clean_prices(self, prices, **kwargs):
        pass

    def _validate_no_missing_prices(self, prices):
        if prices.isna().any().any():
            raise ValueError("Prices contain missing values after cleaning.")

    def remove_negative_prices(
        self,
        prices,
        formation_start,
        formation_end,
    ):
        formation_prices = prices[formation_start:formation_end]
        negative_prices = (formation_prices < 0).sum() > 0
        return prices[negative_prices[~negative_prices].index]

    def remove_too_many_zero_returns(
        self,
        prices,
        formation_start,
        formation_end,
        threshold=0.2,
    ):
        formation_prices = prices[formation_start:formation_end]
        number_of_zero_returns = (formation_prices.pct_change() == 0).sum()
        zero_returns_ratio = (
            number_of_zero_returns
            / (formation_prices.shape[0] - 1)
        ) > threshold

        return prices[zero_returns_ratio[~zero_returns_ratio].index]

    def remove_stationary_prices(
        self,
        prices,
        formation_start,
        formation_end,
        significance_level=0.05,
    ):
        formation_prices = prices[formation_start:formation_end]
        non_stationary_tickers = [
            ticker
            for ticker
            in formation_prices
            if adf(formation_prices[ticker].values)[1] > significance_level
        ]

        return prices[non_stationary_tickers]

    def minimum_liquidity(
        self,
        prices,
        currency_volume,
        formation_start,
        formation_end,
        minimum_average_daily_volume=float("-inf"),
    ):
        formation_cv = currency_volume[formation_start:formation_end]
        liquid_tickers = formation_cv.mean() > minimum_average_daily_volume

        return prices[liquid_tickers[liquid_tickers].index]
