from itertools import chain
import yaml

from pair_trading.catalog import PairTradingCatalog
from pair_trading.logger.loggers import setup_logging


class Workflow(metaclass=PairTradingCatalog):

    alias = 'workflow'

    def __init__(
        self,
        config: dict | str,
        logger: str = None,
    ):
        if isinstance(config, str) and config.endswith('.yaml'):
            with open(config, "r") as config_file:
                config = yaml.safe_load(config_file)

        if logger is not None:
            setup_logging(logger)

        self.data = None
        self.metadata = None
        self.selected_pairs = None
        self.backtester = None
        self.portfolio = None

        self.data_loader = PairTradingCatalog.invoke(
            category='step',
            variant='data_loader',
            config=config['data_loader'],
        )

        self.period = PairTradingCatalog.invoke(
            category='step',
            variant='period',
            config=config['period'],
        )

        self.data_cleaner = PairTradingCatalog.invoke(
            category='step',
            variant='data_cleaner',
            config=config['data_cleaner'],
        )

        self.grouper = PairTradingCatalog.invoke(
            category='step',
            variant='grouper',
            config=config['grouper'],
        )

        self.pair_selection = PairTradingCatalog.invoke(
            category='step',
            variant='pair_selection',
            config=config['pair_selection'],
        )

        self.trading_rules = PairTradingCatalog.invoke(
            category='step',
            variant='trading_rules',
            config=config.get('trading_rules') or {},
        )

        self.backtest = PairTradingCatalog.invoke(
            category='step',
            variant='backtest',
            config=config['backtest'],
        )

    def run(self):
        self.metadata, self.ohlcv = self.data_loader.run()
        periods = self.period.run(self.ohlcv)

        all_period_pairs = []
        trade_sources = []

        for period in periods:
            period_dates = period.date_properties

            period_metadata, period_prices = self.data_cleaner.run(
                metadata=self.metadata,
                ohlcv=self.ohlcv,
                formation_start=period_dates['formation_start'],
                formation_end=period_dates['formation_end'],
                trading_end=period_dates['trading_end'],
            )

            pair_names = self.grouper.run(
                prices=period_prices,
                metadata=period_metadata,
                formation_start=period_dates['formation_start'],
                formation_end=period_dates['formation_end'],
            )

            selected_pairs = self.pair_selection.run(
                pair_names=pair_names,
                prices=period_prices,
                formation_start=period_dates['formation_start'],
                formation_end=period_dates['formation_end'],
                trading_start=period_dates['trading_start'],
                trading_end=period_dates['trading_end'],
            )
            all_period_pairs.append(selected_pairs)

        self.selected_pairs = list(chain.from_iterable(all_period_pairs))
        trade_sources = self.trading_rules.run(all_period_pairs)

        self.portfolio = self.backtest.run(
            ohlcv=self.ohlcv,
            trade_sources=trade_sources,
        )
        self.backtester = self.backtest.backtester
