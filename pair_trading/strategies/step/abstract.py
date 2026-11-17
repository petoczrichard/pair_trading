from abc import ABC, abstractmethod

from algo_toolbox.catalogue import AlgoToolboxMeta


class AbstractStep(ABC, metaclass=AlgoToolboxMeta):

    alias = 'step'

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, **kwargs):
        pass
