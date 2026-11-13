import numpy as np
import pandas as pd

from pair_trading.portfolio.mixins.metrics import PortfolioMetricsMixin
from pair_trading.portfolio.mixins.plotting import PortfolioPlottingMixin


class Portfolio(
    PortfolioMetricsMixin,
    PortfolioPlottingMixin,
):
    """A backtested portfolio or a values-only benchmark via ``values=Series``."""

    def __init__(
        self,
        backtester=None,
        index=None,
        columns=None,
        # metadata,
        source_id_to_index=None,
        ticker_to_index=None,
        date_to_index=None,
        *,
        values=None,
    ):
        if backtester is None and values is None:
            raise ValueError("Provide either backtester or values.")
        if backtester is not None and values is not None:
            raise ValueError("Provide either backtester or values, not both.")
        if values is not None and not isinstance(values, pd.Series):
            raise TypeError("values must be a pandas Series with its own index.")

        self.backtester = backtester
        self.holdings = None
        self.prices = None
        if backtester is not None:
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
        else:
            self.values = values.copy()
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
