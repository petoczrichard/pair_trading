from itertools import chain

from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.logger.logger_decorator import logger_decorator


class TradingRulesStep(AbstractStep):

    alias = 'trading_rules'

    @logger_decorator(
        input_formatter={
            'all_period_pairs': ('__len__',),
        },
        output_formatter={
            'trade_sources': ('__len__',),
        },
    )
    def run(self, all_period_pairs):
        trade_sources = []

        for period_pairs in all_period_pairs:

            max_ratio_of_portfolio_value = 1 / len(period_pairs)
            period_trade_sources = [
                trade | {'max_ratio_of_portfolio_value': max_ratio_of_portfolio_value}  # noqa: E501
                for pair in period_pairs
                for trade in pair.get_trades(
                    period="trading",
                    **(self.config or {}),
                )
            ]
            trade_sources.append(period_trade_sources)
        
        return list(chain.from_iterable(trade_sources))
