import numpy as np
import pandas as pd

from pair_trading.portfolio.mixins.metrics import PortfolioMetricsMixin
from pair_trading.portfolio.mixins.plotting import PortfolioPlottingMixin


class Portfolio(
    PortfolioMetricsMixin,
    PortfolioPlottingMixin,
):
    def __init__(
        self,
        backtester,
        index,
        columns,
        # metadata,
        source_id_to_index,
        ticker_to_index,
        date_to_index,
    ):
        self.backtester = backtester
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
        if not self.values.empty and (
            not np.isfinite(self.values.to_numpy()).all()
            or self.values.iloc[0] <= 0
        ):
            raise ValueError(
                'Portfolio values must be finite and start positive',
            )

        # self.metadata = metadata
        self.source_id_to_index=source_id_to_index
        self.ticker_to_index=ticker_to_index
        self.date_to_index=date_to_index
