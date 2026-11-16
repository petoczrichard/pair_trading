from pair_trading.strategies.step.abstract import AbstractStep
from algo_toolbox.logger.logger_decorator import logger_decorator


class NullStep(AbstractStep):

    alias = 'null'

    @logger_decorator()
    def run(self):
        pass
