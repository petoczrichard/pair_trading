from itertools import groupby
import numpy as np
import pandas as pd


class PortfolioMetricsMixin:
    def _require_metric_data(self, metric, *attributes):
        missing = [name for name in attributes if getattr(self, name, None) is None]
        if missing:
            raise ValueError(
                f"{metric} requires {', '.join(missing)}; "
                "portfolio values alone are insufficient."
            )

    def drawdown(self):
        portfolio_values = self.values
        return portfolio_values / portfolio_values.cummax()

    def rolling_return(self, period):
        return self.values.pct_change(periods=period)

    def rolling_annualized_volatility(self, period):
        return (
            self.values
            .pct_change()
            .dropna()
            .rolling(window=period)
            .std()
        ) * 252 ** 0.5

    def annualized_return(self, periods_per_year=252):
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and positive.")
        values = self.values
        if len(values) < 2:
            return np.nan
        if (values <= 0).any():
            raise ValueError("Portfolio values must be positive.")

        return (values.iloc[-1] / values.iloc[0]) ** (
            periods_per_year / (len(values) - 1)
        ) - 1

    def annualized_volatility(self, periods_per_year=252):
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and positive.")
        returns = self.values.pct_change(fill_method=None)
        return returns.std() * periods_per_year ** 0.5

    def sharpe_ratio(self, periods_per_year=252):
        """Return annualized return / annualized volatility, with no risk-free adjustment."""
        annualized_return = self.annualized_return(periods_per_year)
        volatility = self.annualized_volatility(periods_per_year)
        return annualized_return / volatility if volatility != 0 else np.nan

    def rolling_sharpe_ratio(self, period):
        """Return rolling period return / rolling annualized volatility.

        Uses the existing rolling metrics directly, with no risk-free
        adjustment. Zero volatility produces NaN.
        """
        volatility = self.rolling_annualized_volatility(period)
        return (
            self.rolling_return(period) / volatility.replace(0, np.nan)
        ).rename("rolling_sharpe_ratio")

    def currency_turnover(self):
        self._require_metric_data("currency_turnover", "backtester")
        backtest_prices = self.backtester.prices()
        turnover = np.zeros(backtest_prices.shape[0])

        for trade in self.backtester.blotter_trades():
            trade_date = trade.date
            turnover[trade_date] += abs(
                trade.amount * backtest_prices[trade_date, trade.ticker]
            )

        return pd.Series(turnover, index=self.values.index)

    def relative_turnover(self):
        self._require_metric_data("relative_turnover", "backtester")
        return self.currency_turnover() / self.values

    def trade_durations(self):
        self._require_metric_data("trade_durations", "backtester")
        trades = self.backtester.blotter_trades()

        groups = groupby(
            sorted(
                trades,
                key=lambda t: (t.source_id, t.ticker, t.date)
            ),
            key=lambda t: (t.source_id, t.ticker),
        )

        trade_duration = np.zeros(len(trades) // 2)
        ind = 0

        for group in groups:
            group_trades = list(group[1])
            round_trip_trade_number = len(group_trades) // 2

            for i in range(round_trip_trade_number):
                trade_duration[ind] = (
                    group_trades[2 * i + 1].date - group_trades[2 * i].date
                )
                ind += 1

        return trade_duration

    def drawdown_metrics(self) -> pd.DataFrame:
        """Summarize drawdowns from positive portfolio values with a DatetimeIndex."""
        values = self.values

        columns = [
            "id", "start", "trough", "end",
            "duration", "min", "recovered",
        ]

        if values.empty:
            return pd.DataFrame(columns=columns)

        if not isinstance(values.index, pd.DatetimeIndex):
            raise TypeError("values must have a DatetimeIndex.")
        if not values.index.is_unique:
            raise ValueError("values must have unique timestamps.")
        if (values <= 0).any():
            raise ValueError("Portfolio values must be positive.")

        drawdown = values / values.cummax() - 1

        episodes = []
        peak_date = values.index[0]
        start = None
        trough = None
        minimum = 0.0

        for date, dd in drawdown.items():
            if dd < 0:
                if start is None:
                    start = peak_date
                    trough = date
                    minimum = float(dd)
                elif dd < minimum:
                    trough = date
                    minimum = float(dd)

            else:
                if start is not None:
                    episodes.append({
                        "id": len(episodes) + 1,
                        "start": start,
                        "trough": trough,
                        "end": date,
                        "duration": (date - start).days,
                        "min": minimum,
                        "recovered": True,
                    })
                    start = None

                peak_date = date

        # Include an ongoing drawdown through the last observation.
        if start is not None:
            end = values.index[-1]
            episodes.append({
                "id": len(episodes) + 1,
                "start": start,
                "trough": trough,
                "end": end,
                "duration": (end - start).days,
                "min": minimum,
                "recovered": False,
            })

        return pd.DataFrame(episodes, columns=columns)

    def exposure(self, kind="all", relative=False):
        """Return exposure in currency amounts or relative to portfolio value.

        Select 'long', 'short', 'net', or 'gross' for a Series, or 'all'
        for a DataFrame. Short exposure is negative. Cash and transaction
        costs are excluded. Set relative=True to divide by portfolio value.
        """
        if kind not in ("all", "long", "short", "net", "gross"):
            raise ValueError(
                "kind must be 'all', 'long', 'short', 'net', or 'gross'."
            )

        self._require_metric_data("exposure", "prices", "holdings")
        # The final six columns contain cash and transaction costs.
        exposures = (self.prices * self.holdings).iloc[:, :-6]
        long_exposure = exposures[exposures > 0].sum(axis=1)
        short_exposure = exposures[exposures < 0].sum(axis=1)
        result = pd.DataFrame({
            "long": long_exposure,
            "short": short_exposure,
            "net": long_exposure + short_exposure,
            "gross": long_exposure.abs() + short_exposure.abs(),
        })

        if relative:
            result = result.div(self.values, axis=0)

        return result if kind == "all" else result[kind]

    def number_of_assets(self, kind="all", threshold=1e-2):
        """Count positions per date with absolute currency exposure > threshold.

        Select 'long', 'short', or 'all' (both sides). The threshold must be
        finite and nonnegative. Cash and transaction costs are excluded.
        """
        if kind not in ("all", "long", "short"):
            raise ValueError("kind must be 'all', 'long', or 'short'.")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative.")

        self._require_metric_data("number_of_assets", "prices", "holdings")
        exposures = (self.prices * self.holdings).iloc[:, :-6]
        if kind == "long":
            positions = exposures > threshold
        elif kind == "short":
            positions = exposures < -threshold
        else:
            positions = exposures.abs() > threshold

        return positions.sum(axis=1).rename("number_of_assets")

    def effective_asset_ratio(self):
        """Return the effective asset count divided by the active asset count.

        The effective count is the inverse sum of squared exposures normalized
        by gross exposure. The threshold applies only to the active count,
        using absolute currency exposure > threshold. Cash and costs are
        excluded. Dates with zero gross exposure or no active assets return NaN.
        """
        self._require_metric_data("effective_asset_ratio", "prices", "holdings")
        asset_count = self.number_of_assets()
        exposures = (self.prices * self.holdings).iloc[:, :-6]
        gross_exposure = exposures.abs().sum(axis=1)
        weights = exposures.div(gross_exposure, axis=0)
        concentration = weights.pow(2).sum(axis=1)

        return (
            1 / concentration / asset_count
        ).rename("effective_asset_ratio")
