import numpy as np

from pair_trading.catalog import PairTradingCatalog

from trading_core import TransactionCostEngine


class TransactionCostModel(metaclass=PairTradingCatalog):

    alias = 'transaction_cost_model'

    def __init__(
        self,
        open_prices,
        high_prices,
        low_prices,
        close_prices,
        currency_volume,
    ):
        self.open = open_prices
        self.high = high_prices
        self.low = low_prices
        self.close = close_prices
        self.currency_volume = currency_volume

    def adv(self, window=21):
        return self.currency_volume.rolling(window).median()

    def volatility(self, window=21):
        parkinson_daily = (np.log(self.high / self.low) ** 2) / (4 * np.log(2))
        return parkinson_daily.rolling(window).mean() ** 0.5

    def spread_proxy(self):
        log_hl = np.log(self.high / self.low)
        beta = (log_hl ** 2).rolling(2).sum()

        high_2d = self.high.rolling(2).max()
        low_2d = self.low.rolling(2).min()
        gamma = np.log(high_2d / low_2d) ** 2

        denom = 3 - 2 * np.sqrt(2)

        alpha = (
            (np.sqrt(2 * beta) - np.sqrt(beta)) / denom
            - np.sqrt(gamma / denom)
        )

        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        return spread.clip(lower=0)

    def calibrate_spread_cost(self, average=0.0005):
        proxy = self.spread_proxy()
        flat_proxy = proxy.values.flatten()
        flat_proxy = flat_proxy[~np.isnan(flat_proxy)]

        return proxy / np.mean(flat_proxy) * average

    def calibrate_slippage_cost(self, window=10, average=0.00075):
        volatility = self.volatility(window=window)
        flat_volatility = volatility.values.flatten()
        flat_volatility = flat_volatility[~np.isnan(flat_volatility)]

        return volatility / np.mean(flat_volatility) * average

    def calibrate_impact_cost(
        self,
        window=63,
        average=0.0015,
        average_participation_ratio=0.01,
    ):
        volatility = self.volatility(window=window)
        flat_volatility = volatility.values.flatten()
        flat_volatility = flat_volatility[~np.isnan(flat_volatility)]

        return (
            volatility
            / np.mean(flat_volatility)
            / np.sqrt(average_participation_ratio)
            * average
        )

    @staticmethod
    def make_engine(
        commision,
        borrowing_cost,
        spread,
        slippage,
        market_impact,
        adv,
    ) -> TransactionCostEngine:
        return TransactionCostEngine(
            commission=commision,
            borrowing_cost=borrowing_cost,
            spread=spread,
            slippage=slippage,
            market_impact=market_impact,
            adv=adv,
        )
