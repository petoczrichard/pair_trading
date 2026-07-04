from pair_trading.strategies.step.abstract import AbstractStep


class NullStep(AbstractStep):

    alias = 'null'

    def run(self):
        pass
