from itertools import groupby
from typing import Literal
import pandas as pd


class Portfolio:
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

        # self.metadata = metadata
        self.source_id_to_index=source_id_to_index
        self.ticker_to_index=ticker_to_index
        self.date_to_index=date_to_index

    def exposure(
        type_: Literal['net', 'gross'],
        level: Literal['pair_instance', 'pair', 'asset', 'sector', 'industry', 'country', 'long_short'],
    ):
        pass

    def contributions(
        self,
        level: Literal['pair_instance', 'pair', 'asset', 'sector', 'industry', 'country', 'long_short'],
    ):
        pass

    def performance(self):
        # ret, vol, sharpe, drawdown, etc.
        # rolling metrics as well
        # exposure, sector, industry, country, factors?
        pass

    @staticmethod
    def _group_trade_to_asset(trades):
        return [
            list(group)
            for _, group in groupby(
                sorted(trades, key=lambda t: t.ticker),
                key=lambda t: t.ticker,
            )
        ]

    @staticmethod
    def _group_trade_to_pair_instance(trades):
        return [
            list(group)
            for _, group in groupby(
                sorted(trades, key=lambda t: t.source_id),
                key=lambda t: t.source_id,
            )
        ]

    @staticmethod
    def _group_trade_to_long_short(trades):
        groups = {'long': [], 'short': []}
        sorted_trades = sorted(
            trades,
            key=lambda trade: (trade.source_id, trade.ticker, trade.date),
        )

        for i in range(0, len(sorted_trades), 2):
            opening_trade = sorted_trades[i]
            closing_trade = sorted_trades[i + 1]

            group = (
                groups['long']
                if opening_trade.amount > 0
                else groups['short']
            )
            group.append(opening_trade)
            group.append(closing_trade)

        return list(groups.values())

    def _group_trade_to_pair(self, trades):
        groups = {}
        instance_to_pair = {
            v: k.split('_')[0] for k, v in self.source_id_to_index.items()
        }

        for trade in trades:
            groups.setdefault(
                instance_to_pair[trade.source_id], [],
            ).append(trade)

        return list(groups.values())

    def _group_trade_to_metadata_field(
        self,
        trades,
        field,
    ):
        groups = {}
        index_to_ticker = {v: k for k, v in self.ticker_to_index.items()}

        for trade in trades:
            ticker_name = index_to_ticker[trade.ticker]
            field_name = self.metadata.loc[ticker_name, field]
            groups.setdefault(field_name, []).append(trade)

        return list(groups.values())
