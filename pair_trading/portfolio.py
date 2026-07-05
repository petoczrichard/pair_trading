from typing import Literal
import pandas as pd


class Portfolio:
    def __init__(self, backtester, index, columns):
        self.holdings = pd.DataFrame(
            backtester.holdings(),
            index=index,
            columns=columns,
        )
        self.prices = pd.DataFrame(
            backtester.prices(),
            index=index,
            columns=columns,
        )
        self.values = (
            (self.holdings * self.prices)
                .sum(axis=1)
                .rename('portfolio')
        )

    def contributions(
        self,
        level: Literal['pair', 'asset', 'sector', 'industry', 'country'],
    ):
        pass

    def performance(self):
        # ret, vol, sharpe, drawdown, etc.
        # rolling metrics as well
        # exposure, sector, industry, country, factors?
        pass
