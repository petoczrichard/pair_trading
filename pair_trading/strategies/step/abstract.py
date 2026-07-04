from abc import ABC, abstractmethod

from pair_trading.catalog import PairTradingMeta


class AbstractStep(ABC, metaclass=PairTradingMeta):

    alias = 'step'

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, **kwargs):
        pass
