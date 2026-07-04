from typing import Literal


class Portfolio:
    def __init__(self, pairs):
        pass

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
