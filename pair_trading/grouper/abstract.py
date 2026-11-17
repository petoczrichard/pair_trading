from abc import ABC, abstractmethod
import itertools

from algo_toolbox.catalogue import AlgoToolboxMeta


class AbstractGrouper(
    ABC,
    metaclass=AlgoToolboxMeta,
):

    alias = 'grouper'

    @abstractmethod
    def create_groups(self, **kwargs):
        pass

    def create_pairs(self, groups):
        return itertools.chain.from_iterable(
            itertools.combinations(group, 2) for group in groups
        )
