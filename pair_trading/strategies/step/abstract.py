from abc import ABC, abstractmethod

from algo_toolbox.catalogue import PairTradingMeta


class AbstractStep(ABC, metaclass=PairTradingMeta):

    alias = 'step'

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, **kwargs):
        pass
