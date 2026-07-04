from abc import ABC, abstractmethod
import itertools

from pair_trading.catalog import PairTradingMeta


class AbstractGrouper(
    ABC,
    metaclass=PairTradingMeta,
):

    alias = 'grouper'

    @abstractmethod
    def create_groups(self, **kwargs):
        pass

    def create_pairs(self, groups):
        return itertools.chain.from_iterable(
            itertools.combinations(group, 2) for group in groups
        )
