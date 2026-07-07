import pandas as pd

from pair_trading.plotting import (
    prices_plot as _prices_plot,
    scatter_plot as _scatter_plot,
    spread_plot as _spread_plot,
    equity_curve_plot as _equity_curve_plot,
    figures_to_subplot,
)
from pair_trading.utils import filter_allowed_kwargs


ALIAS_TO_PAIR_TYPE = {
    'distance': 'Distance',
    'engle_granger': 'Engle-Granger',
    'johansen': 'Johansen',
}


class PlottingMixin:
    def plot(self, **kwargs):
        prices_plot = self.prices_plot(title="Prices")
        spread_plot = self.spread_plot(title="Spread")
        scatter_plot = self.scatter_plot(title="Scatter")
        equity_curve_plot = self.equity_curve_plot(title="Portfolio")

        default_kwargs = {
            'height': 750,
            'width': 1250,
            'showlegend': False,
            'vertical_spacing': 0.1,
            'horizontal_spacing': 0.1,
            'title': {
                'x': 0.5,
                'text': f'{self.name} {ALIAS_TO_PAIR_TYPE[self.alias]} Pair',
            }
        }
        kwargs = default_kwargs | kwargs

        return figures_to_subplot(
            [
                prices_plot,
                spread_plot,
                scatter_plot,
                equity_curve_plot,
            ],
            rows=2,
            columns=2,
            **kwargs,
        )

    def prices_plot(self, **kwargs):
        return _prices_plot(
            self.price1,
            self.price2,
            self.formation_start,
            self.formation_end,
            self.trading_start,
            self.trading_end,
            **kwargs,
        )

    def spread_plot(self, **kwargs):
        return _spread_plot(
            self.spread,
            self.formation_start,
            self.formation_end,
            self.trading_start,
            self.trading_end,
            self.spread_mean,
            self.spread_std,
            **kwargs,
        )

    def scatter_plot(self, **kwargs):
        return _scatter_plot(
            self.price1,
            self.price2,
            self.formation_start,
            self.formation_end,
            self.trading_start,
            self.trading_end,
            self.hedge_ratio,
            self.spread_mean,
            self.spread_std,
            **kwargs,
        )

    def equity_curve_plot(self, **kwargs):
        equity_curve_kwargs = filter_allowed_kwargs(self.equity_curve, kwargs)
        figure_kwargs = {
            k: v for k, v in kwargs.items() if k not in equity_curve_kwargs
        }

        backtester = self.equity_curve(period='all', **equity_curve_kwargs)
        portfolio_value = pd.Series(
            (backtester.holdings() * backtester.prices()).sum(axis=1),
            index=self.price1.index,
        )

        return _equity_curve_plot(
            portfolio_value,
            self.formation_start,
            self.formation_end,
            self.trading_start,
            self.trading_end,
            **figure_kwargs,
        )
