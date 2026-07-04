from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog
from pair_trading.utils import date_offset


class PeriodStep(AbstractStep):

    alias = 'period'

    def run(self, ohlcv):
        first_formation_start = self.config['period_start']
        first_formation_end = date_offset(
            first_formation_start,
            self.config['formation_length'],
        )
        first_trading_start = date_offset(
            first_formation_end,
            self.config['formation_trading_gap'],
        )
        first_trading_end = date_offset(
            first_trading_start,
            self.config['trading_length'],
        )

        return PairTradingCatalog.create_instance(
            name='rolling_period',
            formation_start=first_formation_start,
            formation_end=first_formation_end,
            trading_start=first_trading_start,
            trading_end=first_trading_end,
            time_step=self.config['timestep'],
            number_of_periods=self.config.get('number_of_periods'),
            first_valid_date=first_formation_start,
            last_valid_date=self.config.get('period_end') or ohlcv.index.max(),
        )
