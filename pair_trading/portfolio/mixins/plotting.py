import plotly.graph_objects as go


class PortfolioPlottingMixin:
    """Plot portfolio metrics; keyword arguments override the figure layout."""

    @staticmethod
    def _metric_series_plot(values, title, yaxis_title, /, percent=False, **kwargs):
        figure = go.Figure()
        series = values.items() if values.ndim == 2 else [(title, values)]
        for name, data in series:
            figure.add_trace(go.Scatter(
                x=data.index, y=data, mode="lines", name=name,
            ))
        figure.update_layout(
            title=title, xaxis_title="Date", yaxis_title=yaxis_title,
        )
        if percent:
            figure.update_yaxes(tickformat=".1%")
        figure.update_layout(**kwargs)
        return figure

    @staticmethod
    def _metric_scalar_plot(value, title, /, **kwargs):
        figure = go.Figure(go.Bar(x=[title], y=[value], name=title))
        figure.update_layout(title=title, yaxis_tickformat=".1%")
        figure.update_layout(**kwargs)
        return figure

    def drawdown_plot(self, **kwargs):
        """Plot percentage losses from the running portfolio peak."""
        return self._metric_series_plot(
            self.drawdown() - 1, "Drawdown", "Drawdown", percent=True,
            **kwargs,
        )

    def rolling_return_plot(self, period, **kwargs):
        return self._metric_series_plot(
            self.rolling_return(period), f"Rolling Return ({period} periods)",
            "Return", percent=True, **kwargs,
        )

    def rolling_annualized_volatility_plot(self, period, **kwargs):
        return self._metric_series_plot(
            self.rolling_annualized_volatility(period),
            f"Rolling Annualized Volatility ({period} periods)",
            "Annualized Volatility", percent=True, **kwargs,
        )

    def annualized_return_plot(self, periods_per_year=252, **kwargs):
        return self._metric_scalar_plot(
            self.annualized_return(periods_per_year), "Annualized Return",
            **kwargs,
        )

    def annualized_volatility_plot(self, periods_per_year=252, **kwargs):
        return self._metric_scalar_plot(
            self.annualized_volatility(periods_per_year),
            "Annualized Volatility", **kwargs,
        )

    def sharpe_ratio_plot(self, periods_per_year=252, **kwargs):
        return self._metric_scalar_plot(
            self.sharpe_ratio(periods_per_year), "Sharpe Ratio",
            **({"yaxis_tickformat": ".2f"} | kwargs),
        )

    def rolling_sharpe_ratio_plot(self, period, **kwargs):
        return self._metric_series_plot(
            self.rolling_sharpe_ratio(period),
            f"Rolling Sharpe Ratio ({period} periods)",
            "Sharpe Ratio", **kwargs,
        )

    def currency_turnover_plot(self, **kwargs):
        return self._metric_series_plot(
            self.currency_turnover(), "Currency Turnover", "Turnover (currency)",
            **kwargs,
        )

    def relative_turnover_plot(self, **kwargs):
        return self._metric_series_plot(
            self.relative_turnover(), "Relative Turnover", "Turnover",
            percent=True, **kwargs,
        )

    def trade_durations_plot(self, nbins=None, **kwargs):
        """Plot round-trip durations measured in backtest periods."""
        figure = go.Figure(go.Histogram(
            x=self.trade_durations(), nbinsx=nbins, name="Trade Durations",
        ))
        figure.update_layout(
            title="Trade Durations", xaxis_title="Duration (periods)",
            yaxis_title="Number of Trades",
        )
        figure.update_layout(**kwargs)
        return figure

    def drawdown_metrics_plot(self, **kwargs):
        """Display drawdown episodes, including recovery status, as a table."""
        metrics = self.drawdown_metrics()
        figure = go.Figure(go.Table(
            header=dict(values=[
                "ID", "Start", "Trough", "End", "Duration (days)",
                "Minimum Drawdown", "Recovered",
            ]),
            cells=dict(
                values=[metrics[column] for column in metrics.columns],
                format=[None, None, None, None, ".0f", ".1%", None],
            ),
        ))
        figure.update_layout(title="Drawdown Metrics")
        figure.update_layout(**kwargs)
        return figure

    def exposure_plot(self, kind="all", relative=False, **kwargs):
        return self._metric_series_plot(
            self.exposure(kind=kind, relative=relative),
            f"{kind.title()} Exposure",
            "Relative Exposure" if relative else "Exposure (currency)",
            percent=relative, **kwargs,
        )

    def number_of_assets_plot(self, kind="all", threshold=1e-2, **kwargs):
        return self._metric_series_plot(
            self.number_of_assets(kind=kind, threshold=threshold),
            f"Number of Assets ({kind.title()})", "Number of Assets", **kwargs,
        )

    def effective_asset_ratio_plot(self, **kwargs):
        return self._metric_series_plot(
            self.effective_asset_ratio(), "Effective Asset Ratio",
            "Effective Asset Ratio", percent=True, **kwargs,
        )
