from abc import ABC, abstractmethod
from typing import Literal
import pandas as pd
import numpy as np

from pair_trading.pair.mixins.metrics import MetricsMixin
from pair_trading.pair.mixins.plotting import PlottingMixin
from pair_trading.catalog import PairTradingMeta
from pair_trading.transaction_cost_model import TransactionCostModel
from pair_trading.numba.backtest import (
    compute_positions as numba_compute_positions,
    get_trade_dates as get_trade_dates_numba,
)
from pair_trading.numba.metrics import (
    ssd as numba_ssd,
)
from pair_trading.numba.statistical import (
    numba_mean,
    numba_std,
    numba_create_spread,
)
from pair_trading.numba.linalg import roll_numba

from trading_core import (
    TradeType,
    TradeSource,
    Basket,
    BasketWeightType,
    BacktesterEngine,
)


class AbstractPair(
    ABC,
    MetricsMixin,
    PlottingMixin,
    metaclass=PairTradingMeta,
):

    alias = 'pair'

    def __init__(
        self,
        price1,
        price2,
        formation_start,
        formation_end,
        trading_start,
        trading_end,
    ):
        self.price1 = price1
        self.price2 = price2
        self.price1_values = price1.values
        self.price2_values = price2.values

        self.name1 = price1.name
        self.name2 = price2.name
        self.name = self.name1 + '-' + self.name2

        self.formation_start = pd.Timestamp(formation_start)
        self.formation_end = pd.Timestamp(formation_end)
        self.trading_start = pd.Timestamp(trading_start)
        self.trading_end = pd.Timestamp(trading_end)

        self.id = self.name + '_' + str(self.formation_start)

        price_index = price1.index
        self.formation_start_index = price_index.searchsorted(formation_start)
        self.formation_end_index = price_index.searchsorted(formation_end) + 1
        self.trading_start_index = price_index.searchsorted(trading_start)
        self.trading_end_index = price_index.searchsorted(trading_end) + 1

        self.formation_period = slice(self.formation_start, self.formation_end)
        self.trading_period = slice(self.trading_start, self.trading_end)

        self.formation_period_index = slice(
            self.formation_start_index,
            self.formation_end_index,
        )
        self.trading_period_index = slice(
            self.trading_start_index,
            self.trading_end_index,
        )

        self.hedge_ratio = None
        self.spread = None
        self.spread_mean = None
        self.spread_std = None

    @abstractmethod
    def calculate(self):
        pass

    def is_long_short(self):
        return self.hedge_ratio > 0

    def is_significant(self, **kwargs):
        return None

    def create_spread(self):
        if self.hedge_ratio is None:
            raise ValueError("Hedge ratio not calculated yet.")

        self.spread = pd.Series(
            numba_create_spread(
                self.price1_values,
                self.price2_values,
                self.hedge_ratio
            ),
            index=self.price1.index,
        )
        self.spread_values = self.spread.values

        formation_spread = self.spread_values[self.formation_period_index]
        self.spread_mean = numba_mean(formation_spread)
        self.spread_std = numba_std(formation_spread, ddof=1)

        return self.spread

    def calculate_ssd(self):
        return numba_ssd(
            self.price1_values[self.formation_period_index],
            self.price2_values[self.formation_period_index],
        )

    def compute_positions(
        self,
        period: Literal['formation', 'trading', 'all'] = 'formation',
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
    ):
        if period == 'formation':
            period_slice = self.formation_period_index
        elif period == 'trading':
            period_slice = self.trading_period_index
        elif period == 'all':
            period_slice = slice(
                self.formation_start_index,
                self.trading_end_index,
            )
        else:
            raise ValueError(
                "Invalid period. Must be 'formation', 'trading', or 'all'."
            )

        values = numba_compute_positions(
            spread=self.spread_values[period_slice],
            mean=self.spread_mean,
            std=self.spread_std,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )

        return pd.Series(values, index=self.spread.index[period_slice])

    def get_trades(
        self,
        period: Literal['formation', 'trading', 'all'] = 'formation',
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
    ):
        positions = self.compute_positions(
            period=period,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )
        dates_ints = np.array(positions.index.map(pd.Timestamp.timestamp))
        positions_values = positions.values
        positions_values = roll_numba(positions_values, 1)
        positions_values[0] = 0
        positions_values[-1] = 0

        entries, exits = get_trade_dates_numba(positions_values, dates_ints)

        trade_list = []
        default_pair_info = {
            "pair_id": self.id,
            "name1": self.name1,
            "name2": self.name2,
            "hedge_ratio": self.hedge_ratio,
        }
        timestamp_multiplier = 1_000_000_000

        for date in entries:
            entry_date = pd.Timestamp(date * timestamp_multiplier)
            trade_list.append({
                "type": "entry",
                "direction": (
                    "long" if positions.loc[entry_date] == -1 else "short"
                ),
                "trade_date": entry_date,
                **default_pair_info,
            })
        for date in exits:
            trade_list.append({
                "type": "exit",
                "trade_date": pd.Timestamp(date * timestamp_multiplier),
                **default_pair_info,
            })

        return trade_list

    def get_trades_cpp(
        self,
        period: Literal['formation', 'trading', 'all'] = 'formation',
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
    ):
        positions = self.compute_positions(
            period=period,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )
        dates_ints = np.array(positions.index.map(pd.Timestamp.timestamp))
        positions_values = positions.values
        positions_values = roll_numba(positions_values, 1)
        positions_values[0] = 0
        positions_values[-1] = 0

        entries, exits = get_trade_dates_numba(positions_values, dates_ints)

        trade_source_list = []

        weights = np.array([1, -self.hedge_ratio])
        tickers = [self.name1, self.name2]
        timestamp_multiplier = 1_000_000_000

        for date in entries:
            entry_date = pd.Timestamp(date * timestamp_multiplier)
            trade_source_list.append({
                "source_id": self.id,
                "type": TradeType.Entry,
                "date": entry_date,
                "basket": {
                    "asset_ids": tickers,
                    "weights": weights * positions_values[
                        positions.index.get_loc(entry_date)
                    ],
                    "weight_type": BasketWeightType.Shares,
                }
            })
        for date in exits:
            trade_source_list.append({
                "source_id": self.id,
                "type": TradeType.Exit,
                "date": pd.Timestamp(date * timestamp_multiplier),
            })

        return trade_source_list

    def equity_curve_cpp(
        self,
        period: Literal['formation', 'trading', 'all'] = 'formation',
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
        initial_cash: float = 100.0,
        leverage: float = 1.0,
    ):
        prices = pd.concat([self.price1, self.price2], axis=1)
        adv = np.ones_like(prices) * 1e12
        tc_engine = TransactionCostModel.make_engine(
            commision=0.0,
            borrowing_cost=0.0,
            spread=np.zeros_like(adv),
            slippage=np.zeros_like(adv),
            market_impact=np.zeros_like(adv),
            adv=adv,
        )
        trade_sources = self.get_trades_cpp(
            period=period,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )

        date_to_index = {
            date: index for index, date in enumerate(prices.index)
        }
        trade_sources_cpp = []

        for trade_source in trade_sources:
            trade_sources_cpp.append(
                TradeSource(
                    source_id=0,
                    type=trade_source['type'],
                    date=date_to_index[trade_source['date']],
                    basket=(
                        Basket(
                            **(trade_source['basket'] | {'asset_ids': [0, 1]})
                        )
                        if trade_source.get('basket') is not None
                        else None
                    ),
                )
            )

        backtester = BacktesterEngine(
            initial_portfolio_value=initial_cash,
            leverage=leverage,
            prices=prices.values,
            adv=adv,
            trade_sources=trade_sources_cpp,
            tc_engine=tc_engine,
        )
        backtester.run()

        # return pd.Series(
        #     (backtester.holdings() * backtester.prices()).sum(axis=1),
        #     index=prices.index,
        # )
        return backtester

    def equity_curve(
        self,
        period: Literal['formation', 'trading', 'all'] = 'formation',
        long_entry: float = -2,
        long_exit: float = 0,
        long_stoploss: float = -5,
        short_entry: float = 2,
        short_exit: float = 0,
        short_stoploss: float = 5,
        initial_cash: float = 100.0,
        leverage: float = 1.0,
    ):
        trades = self.get_trades(
            period=period,
            long_entry=long_entry,
            long_exit=long_exit,
            long_stoploss=long_stoploss,
            short_entry=short_entry,
            short_exit=short_exit,
            short_stoploss=short_stoploss,
        )
        sorted_trades = sorted(
            trades,
            key=lambda trade: (
                trade['trade_date'],
                0 if trade['type'] == 'exit' else 1,
            ),
        )
        trade_blotter = []
        trade_mapping = {}
        trade_counter = 0

        portfolio_holdings = pd.DataFrame(
            index=self.spread.index,
            columns=['USD_CASH', self.name1, self.name2],
        )
        portfolio_holdings.iloc[0] = [initial_cash, 0, 0]

        for i, date in enumerate(portfolio_holdings.index[1:], start=1):
            portfolio_holdings.iloc[i] = portfolio_holdings.iloc[i - 1]

            while sorted_trades and sorted_trades[0]['trade_date'] == date:
                current_trade = sorted_trades.pop(0)

                if current_trade['type'] == 'exit':
                    pair_id = current_trade['pair_id']
                    pair_entry = trade_mapping[pair_id]

                    while pair_entry:
                        trade_id = pair_entry.pop(0)
                        trade = trade_blotter[trade_id]
                        ticker = trade['ticker']
                        amount = trade['amount']

                        current_price = (
                            self.price1
                            if ticker == self.name1
                            else self.price2
                        )

                        portfolio_holdings.loc[date, 'USD_CASH'] += (
                            amount * current_price.loc[date]
                        )
                        portfolio_holdings.loc[date, ticker] -= amount

                        trade_blotter.append({
                            'trade_id': trade_counter,
                            'date': date,
                            'ticker': ticker,
                            'amount': -amount,
                            'pair': self.name,
                            'formation_start': self.formation_start,
                        })
                        trade_counter += 1

                    if len(trade_mapping[pair_id]) == 0:
                        del trade_mapping[pair_id]

                if current_trade['type'] == 'entry':
                    pair_id = current_trade['pair_id']

                    direction = (
                        -1 if current_trade['direction'] == 'long' else 1
                    )
                    pair_amount = (
                        (portfolio_holdings.loc[date, 'USD_CASH'] * leverage)
                        / (
                            self.price1.loc[date]
                            + abs(self.hedge_ratio) * self.price2.loc[date]
                        )
                    )

                    portfolio_holdings.loc[date, self.name1] = (
                        direction * pair_amount
                    )
                    portfolio_holdings.loc[date, 'USD_CASH'] += (
                        - direction * self.price1.loc[date] * pair_amount
                    )

                    trade_blotter.append({
                        'trade_id': trade_counter,
                        'date': date,
                        'ticker': self.name1,
                        'amount': direction * pair_amount,
                        'pair': self.name,
                        'formation_start': self.formation_start,
                    })
                    trade_mapping.setdefault(pair_id, []).append(trade_counter)
                    trade_counter += 1

                    portfolio_holdings.loc[date, self.name2] = (
                        -direction * self.hedge_ratio * pair_amount
                    )
                    portfolio_holdings.loc[date, 'USD_CASH'] += (
                        direction * self.price2.loc[date]
                        * self.hedge_ratio * pair_amount
                    )

                    trade_blotter.append({
                        'trade_id': trade_counter,
                        'date': date,
                        'ticker': self.name2,
                        'amount': -direction * self.hedge_ratio * pair_amount,
                        'pair': self.name,
                        'formation_start': self.formation_start,
                    })
                    trade_mapping.setdefault(pair_id, []).append(trade_counter)
                    trade_counter += 1

        portfolio_prices = pd.concat(
            [
                pd.Series(1, index=self.price1.index, name='USD_CASH'),
                self.price1,
                self.price2,
            ],
            axis=1,
        )
        portfolio_value = (portfolio_holdings * portfolio_prices).sum(axis=1)

        return portfolio_value
