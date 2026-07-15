from abc import ABC, abstractmethod
from typing import Literal, overload
import pandas as pd
import numpy as np

from pair_trading.pair.mixins.metrics import MetricsMixin
from pair_trading.pair.mixins.plotting import PlottingMixin
from pair_trading.catalog import PairTradingMeta
from pair_trading.transaction_cost_model import TransactionCostModel
from pair_trading.numba_helpers.backtest import (
    compute_positions as numba_compute_positions,
    get_trade_dates as get_trade_dates_numba,
)
from pair_trading.numba_helpers.metrics import (
    ssd as numba_ssd,
)
from pair_trading.numba_helpers.statistical import (
    numba_mean,
    numba_std,
    numba_create_spread,
)
from pair_trading.numba_helpers.linalg import roll_numba

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

    @overload
    def __init__(
        self,
        price1,
        price2,
        formation_start,
        formation_end,
        trading_start,
        trading_end,
    ):
        ...

    @overload
    def __init__(
        self,
        name1,
        name2,
        price1_values,
        price2_values,
        index_values,
        formation_start_index,
        formation_end_index,
        trading_start_index,
        trading_end_index,
    ):
        ...

    def __init__(
        self,
        price1=None,
        price2=None,
        formation_start=None,
        formation_end=None,
        trading_start=None,
        trading_end=None,
        name1=None,
        name2=None,
        price1_values=None,
        price2_values=None,
        index_values=None,
        formation_start_index=None,
        formation_end_index=None,
        trading_start_index=None,
        trading_end_index=None,
    ):
        if price1 is not None and price2 is not None:
            self._pandas_init(
                price1=price1,
                price2=price2,
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            )

        elif price1_values is not None and price2_values is not None:
            self._numpy_init(
                name1=name1,
                name2=name2,
                price1_values=price1_values,
                price2_values=price2_values,
                index_values=index_values,
                formation_start_index=formation_start_index,
                formation_end_index=formation_end_index,
                trading_start_index=trading_start_index,
                trading_end_index=trading_end_index,
            )

        else:
            raise ValueError(
                "Initialize pair either through pandas or numpy route."
            )

        self.formation_period_index = slice(
            self.formation_start_index,
            self.formation_end_index,
        )
        self.trading_period_index = slice(
            self.trading_start_index,
            self.trading_end_index,
        )

        self.hedge_ratio = None
        self.spread_values = None
        self.spread_mean = None
        self.spread_std = None

    def _pandas_init(
        self,
        price1,
        price2,
        formation_start,
        formation_end,
        trading_start,
        trading_end,
    ):
        self.name1 = price1.name
        self.name2 = price2.name
        self.price1_values = price1.values
        self.price2_values = price2.values

        pandas_index = price1.index
        self.index_values = (
            pandas_index
                .values
                .astype("datetime64[ms]")
                .astype(np.int64)
        )

        self.formation_start_index = pandas_index.searchsorted(
            pd.Timestamp(formation_start)
        )
        self.formation_end_index = pandas_index.searchsorted(
            pd.Timestamp(formation_end)
        ) + 1
        self.trading_start_index = pandas_index.searchsorted(
            pd.Timestamp(trading_start)
        )
        self.trading_end_index = pandas_index.searchsorted(
            pd.Timestamp(trading_end)
        ) + 1

    def _numpy_init(
        self,
        name1,
        name2,
        price1_values,
        price2_values,
        index_values,
        formation_start_index,
        formation_end_index,
        trading_start_index,
        trading_end_index,
    ):
        self.name1 = name1
        self.name2 = name2
        self.price1_values = price1_values
        self.price2_values = price2_values

        self.index_values = index_values
        self.formation_start_index = formation_start_index
        self.formation_end_index = formation_end_index
        self.trading_start_index = trading_start_index
        self.trading_end_index = trading_end_index

    @property
    def formation_start(self):
        loc = self.formation_start_index
        return self.index_values[loc].astype("datetime64[ms]")

    @property
    def formation_end(self):
        loc = self.formation_end_index - 1
        return self.index_values[loc].astype("datetime64[ms]")


    @property
    def trading_start(self):
        loc = self.trading_start_index
        return self.index_values[loc].astype("datetime64[ms]")

    @property
    def trading_end(self):
        loc = self.trading_end_index - 1
        return self.index_values[loc].astype("datetime64[ms]")

    @property
    def formation_period(self):
        return slice(
            self.formation_start,
            self.formation_end,
        )

    @property
    def trading_period(self):
        return slice(
            self.trading_start,
            self.trading_end,
        )

    @property
    def price1(self):
        return pd.Series(
            data=self.price1_values,
            index=self.index_values.astype("datetime64[ms]"),
            name=self.name1,
        )

    @property
    def price2(self):
        return pd.Series(
            data=self.price2_values,
            index=self.index_values.astype("datetime64[ms]"),
            name=self.name2,
        )

    @property
    def spread(self):
        return pd.Series(
            data=self.spread_values,
            index=self.index_values.astype("datetime64[ms]"),
        )

    @property
    def name(self):
        return self.name1 + '-' + self.name2

    @property
    def id(self):
        return self.name + '_' + str(self.formation_start)

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

        self.spread_values = numba_create_spread(
            self.price1_values,
            self.price2_values,
            self.hedge_ratio
        )

        formation_spread = self.spread_values[self.formation_period_index]
        self.spread_mean = numba_mean(formation_spread)
        self.spread_std = numba_std(formation_spread, ddof=1)

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
        if period == 'formation':
            period_slice = self.formation_period
        elif period == 'trading':
            period_slice = self.trading_period
        elif period == 'all':
            period_slice = slice(
                self.formation_start,
                self.trading_end,
            )
        else:
            raise ValueError(
                "Invalid period. Must be 'formation', 'trading', or 'all'."
            )

        prices = pd.concat(
            [
                self.price1[period_slice],
                self.price2[period_slice],
            ],
            axis=1,
        )
        adv = np.ones_like(prices) * 1e12
        tc_engine = TransactionCostModel.make_engine(
            commision=0.0,
            borrowing_cost=0.0,
            spread=np.zeros_like(adv),
            slippage=np.zeros_like(adv),
            market_impact=np.zeros_like(adv),
            adv=adv,
        )
        trade_sources = self.get_trades(
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

        return backtester
