from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.logger import logger_decorator


class NullStep(AbstractStep):

    alias = 'null'

    @logger_decorator()
    def run(self):
        pass
