from itertools import chain

from pair_trading.catalog import PairTradingCatalog


class Workflow(metaclass=PairTradingCatalog):

    alias = 'workflow'

    def __init__(self, config):
        self.data = None
        self.metadata = None
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

        self.backtest = PairTradingCatalog.invoke(
            category='step',
            variant='backtest',
            config=config['backtest'],
        )

    def run(self):
        self.metadata, self.ohlcv = self.data_loader.run()
        periods = self.period.run(self.ohlcv)

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

            max_ratio_of_portfolio_value = 1 / len(selected_pairs)
            period_trade_sources = [
                trade | {'max_ratio_of_portfolio_value': max_ratio_of_portfolio_value}  # noqa: E501
                for pair in selected_pairs
                for trade in pair.get_trades(period="trading")
            ]
            trade_sources.append(period_trade_sources)

        trade_sources = list(chain.from_iterable(trade_sources))

        self.portfolio = self.backtest.run(
            ohlcv=self.ohlcv,
            trade_sources=trade_sources,
        )
        self.backtester = self.backtest.backtester
