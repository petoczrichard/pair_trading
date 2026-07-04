from pair_trading.catalog import PairTradingCatalog


class Workflow(metaclass=PairTradingCatalog):

    alias = 'workflow'

    def __init__(self, config):
        self.data = None
        self.metadata = None
        self.backtester = None


        self.data_loader = PairTradingCatalog.create_instance(
            category='step',
            variant='data_loader',
            config=config['data_loader'],
        )

        self.period = PairTradingCatalog.create_instance(
            category='step',
            variant='period',
            config=config['period'],
        )

        self.data_cleaner = PairTradingCatalog.create_instance(
            category='step',
            variant='data_cleaner',
            config=config['data_cleaner'],
        )

        self.grouper = PairTradingCatalog.create_instance(
            category='step',
            variant='grouper',
            config=config['grouper'],
        )

        self.pair_selection = PairTradingCatalog.create_instance(
            category='step',
            variant='pair_selection',
            config=config['pair_selection'],
        )

        self.backtest = PairTradingCatalog.create_instance(
            category='step',
            variant='backtest',
            config=config['backtest'],
        )

    def run(self):
        self.metadata, self.data = self.data_loader.run()
