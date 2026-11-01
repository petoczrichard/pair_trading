from itertools import groupby
import numpy as np
import pandas as pd


class PortfolioMetricsMixin:
    def drawdown(self):
        portfolio_values = self.values
        return portfolio_values / portfolio_values.cummax()

    def rolling_return(self, period):
        return self.values.pct_change(period=period)

    def annualized_volatility(period):
        return (
            self.values
            .pct_change()
            .dropna()
            .rolling(period=period)
            .std()
        ) * 252 ** 0.5

    def currency_turnover(self):
        backtest_prices = self.backtester.prices()
        turnover = np.zeros(backtest_prices.shape[0])

        for trade in self.backtester.blotter_trades():
            trade_date = trade.date
            turnover[trade_date] += abs(
                trade.amount * backtest_prices[trade_date, trade.ticker]
            )

        return pd.Series(turnover, index=self.values.index)

    def relative_turnover(self):
        return self.currency_turnover() / self.values

    def trade_durations(self):
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
