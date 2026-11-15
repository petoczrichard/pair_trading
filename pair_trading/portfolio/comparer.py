import pandas as pd
import plotly.graph_objects as go

from pair_trading.portfolio.portfolio import Portfolio


class PortfolioComparer:
    """Compare a collection of uniquely named portfolios.

    Pass an iterable of Portfolio objects. Every metric returns a DataFrame;
    matching *_plot methods return Plotly figures. Use names=None for all
    portfolios, a single name, or a sequence of names to select and order them.

    Series are aligned on the union of their indexes without filling gaps.
    Scalars use portfolio names as rows. Exposure tables have (portfolio,
    exposure) columns; drawdown episodes have (portfolio, episode) rows.
    Trade durations align by trade ordinal, not by date or matched trades.
    Each metric uses the portfolio's full history, not a common date range.
    """

    def __init__(self, portfolios):
        self._portfolios = tuple(portfolios)
        self._by_name()

    def _by_name(self):
        # Revalidate because Portfolio.name can be changed after construction.
        if not self._portfolios:
            raise ValueError("Provide at least one portfolio.")
        result = {}
        for portfolio in self._portfolios:
            if not isinstance(portfolio, Portfolio):
                raise TypeError("All entries must be Portfolio objects.")
            name = portfolio.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Every portfolio must have a nonempty string name.")
            if name in result:
                raise ValueError(f"Duplicate portfolio name: {name!r}.")
            result[name] = portfolio
        return result

    def _select(self, names):
        portfolios = self._by_name()
        if names is None:
            return portfolios
        names = [names] if isinstance(names, str) else list(names)
        if not names:
            raise ValueError("Select at least one portfolio name.")
        if any(not isinstance(name, str) for name in names):
            raise TypeError("Portfolio selections must be names.")
        if len(names) != len(set(names)):
            raise ValueError("Portfolio selections must not contain duplicate names.")
        unknown = [name for name in names if name not in portfolios]
        if unknown:
            raise ValueError(f"Unknown portfolio names: {unknown}.")
        return {name: portfolios[name] for name in names}

    @staticmethod
    def _call(portfolio, method, **kwargs):
        try:
            return getattr(portfolio, method)(**kwargs)
        except (ValueError, TypeError) as error:
            raise type(error)(
                f"Portfolio {portfolio.name!r}, {method}: {error}"
            ) from error

    def _table(self, metric, names, **kwargs):
        results = {
            name: self._call(portfolio, metric, **kwargs)
            for name, portfolio in self._select(names).items()
        }
        first = next(iter(results.values()))
        if metric == "drawdown_metrics":
            return pd.concat(results, names=["portfolio", "episode"])
        if metric == "trade_durations":
            results = {
                name: pd.Series(values, name=name)
                for name, values in results.items()
            }
            return pd.DataFrame(results).rename_axis(
                index="trade", columns="portfolio",
            )
        if isinstance(first, pd.DataFrame):
            return pd.concat(results, axis=1, names=["portfolio", "exposure"])
        if isinstance(first, pd.Series):
            return pd.concat(results, axis=1).rename_axis(columns="portfolio")
        return pd.DataFrame.from_dict(
            results, orient="index", columns=[metric],
        ).rename_axis("portfolio")

    def _plot(self, metric, names, metric_kwargs, **layout):
        selected = self._select(names)
        if metric == "drawdown_metrics":
            table = self._table(metric, names=list(selected)).reset_index()
            table = table.drop(columns="episode")
            figure = go.Figure(go.Table(
                header=dict(values=list(table.columns)),
                cells=dict(
                    values=[table[column] for column in table.columns],
                    format=[
                        ".1%" if column == "min" else None
                        for column in table.columns
                    ],
                ),
            ))
            figure.update_layout(title="Drawdown Metrics Comparison")
        else:
            figure = go.Figure()
            for name, portfolio in selected.items():
                individual = self._call(
                    portfolio, metric + "_plot", **metric_kwargs,
                )
                if not figure.data:
                    figure.update_layout(individual.layout)
                for trace in individual.data:
                    trace.name = (
                        f"{name} — {trace.name}"
                        if len(individual.data) > 1 else name
                    )
                    trace.legendgroup = name
                    trace.showlegend = True
                    if trace.type == "histogram":
                        trace.opacity = 0.6
                        trace.bingroup = metric
                    figure.add_trace(trace)
            figure.update_layout(
                showlegend=True,
                barmode="overlay" if metric == "trade_durations" else "group",
            )
        figure.update_layout(**layout)
        return figure

    def values_plot(self, *, names=None, **kwargs):
        """Compare portfolio values over time for the selected names."""
        return self._plot("values", names, {}, **kwargs)

    def summary(self, periods_per_year=252, *, names=None):
        """Return annualized return, volatility, and Sharpe ratio by portfolio."""
        selected = list(self._select(names))
        return pd.concat([
            self._table(metric, selected, periods_per_year=periods_per_year)
            for metric in (
                "annualized_return", "annualized_volatility", "sharpe_ratio",
            )
        ], axis=1)

    def drawdown(self, *, names=None):
        return self._table("drawdown", names)

    def drawdown_plot(self, *, names=None, **kwargs):
        return self._plot(
            "drawdown", names, {}, **kwargs,
        )

    def rolling_return(self, period, *, names=None):
        return self._table("rolling_return", names, period=period)

    def rolling_return_plot(self, period, *, names=None, **kwargs):
        return self._plot(
            "rolling_return", names, dict(period=period), **kwargs,
        )

    def rolling_annualized_volatility(self, period, *, names=None):
        return self._table("rolling_annualized_volatility", names, period=period)

    def rolling_annualized_volatility_plot(self, period, *, names=None, **kwargs):
        return self._plot(
            "rolling_annualized_volatility", names, dict(period=period), **kwargs,
        )

    def annualized_return(self, periods_per_year=252, *, names=None):
        return self._table("annualized_return", names, periods_per_year=periods_per_year)

    def annualized_return_plot(self, periods_per_year=252, *, names=None, **kwargs):
        return self._plot(
            "annualized_return", names, dict(periods_per_year=periods_per_year), **kwargs,
        )

    def annualized_volatility(self, periods_per_year=252, *, names=None):
        return self._table("annualized_volatility", names, periods_per_year=periods_per_year)

    def annualized_volatility_plot(self, periods_per_year=252, *, names=None, **kwargs):
        return self._plot(
            "annualized_volatility", names, dict(periods_per_year=periods_per_year), **kwargs,
        )

    def sharpe_ratio(self, periods_per_year=252, *, names=None):
        return self._table("sharpe_ratio", names, periods_per_year=periods_per_year)

    def sharpe_ratio_plot(self, periods_per_year=252, *, names=None, **kwargs):
        return self._plot(
            "sharpe_ratio", names, dict(periods_per_year=periods_per_year), **kwargs,
        )

    def rolling_sharpe_ratio(self, period, *, names=None):
        return self._table("rolling_sharpe_ratio", names, period=period)

    def rolling_sharpe_ratio_plot(self, period, *, names=None, **kwargs):
        return self._plot(
            "rolling_sharpe_ratio", names, dict(period=period), **kwargs,
        )

    def currency_turnover(self, *, names=None):
        return self._table("currency_turnover", names)

    def currency_turnover_plot(self, *, names=None, **kwargs):
        return self._plot(
            "currency_turnover", names, {}, **kwargs,
        )

    def relative_turnover(self, *, names=None):
        return self._table("relative_turnover", names)

    def relative_turnover_plot(self, *, names=None, **kwargs):
        return self._plot(
            "relative_turnover", names, {}, **kwargs,
        )

    def trade_durations(self, *, names=None):
        return self._table("trade_durations", names)

    def trade_durations_plot(self, nbins=None, *, names=None, **kwargs):
        return self._plot(
            "trade_durations", names, dict(nbins=nbins), **kwargs,
        )

    def drawdown_metrics(self, *, names=None):
        return self._table("drawdown_metrics", names)

    def drawdown_metrics_plot(self, *, names=None, **kwargs):
        return self._plot(
            "drawdown_metrics", names, {}, **kwargs,
        )

    def exposure(self, kind="all", relative=False, *, names=None):
        return self._table("exposure", names, kind=kind, relative=relative)

    def exposure_plot(self, kind="all", relative=False, *, names=None, **kwargs):
        return self._plot(
            "exposure", names, dict(kind=kind, relative=relative), **kwargs,
        )

    def number_of_assets(self, kind="all", threshold=1e-2, *, names=None):
        return self._table("number_of_assets", names, kind=kind, threshold=threshold)

    def number_of_assets_plot(self, kind="all", threshold=1e-2, *, names=None, **kwargs):
        return self._plot(
            "number_of_assets", names, dict(kind=kind, threshold=threshold), **kwargs,
        )

    def effective_asset_ratio(self, *, names=None):
        return self._table("effective_asset_ratio", names)

    def effective_asset_ratio_plot(self, *, names=None, **kwargs):
        return self._plot(
            "effective_asset_ratio", names, {}, **kwargs,
        )
