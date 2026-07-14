from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog
from pair_trading.logger import logger_decorator


class GrouperStep(AbstractStep):

    alias = 'grouper'

    @logger_decorator(
        formatter={
            'prices': ('shape',),
            'metadata': ('shape',),
        }
    )
    def run(self, prices, metadata, formation_start, formation_end):
        grouper = PairTradingCatalog.invoke(
            category='grouper',
            **self.config['setup'],
        )

        returns = (
            prices
                .loc[formation_start:formation_end]
                .pct_change()
                .dropna()
            )
        groups = grouper.create_groups(
            metadata=metadata,
            returns=returns,
            **(self.config['create_groups'] or {}),
        )

        return list(grouper.create_pairs(groups=groups))
