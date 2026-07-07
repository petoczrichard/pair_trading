from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog


class DataLoaderStep(AbstractStep):

    alias = 'data_loader'

    def run(self):
        data_loader = PairTradingCatalog.invoke(
            category='data_loader',
            **self.config['setup'],
        )
        tickers = data_loader.get_tickers(
            **(self.config.get('get_tickers') or {})
        )
        metadata = data_loader.get_metadata(
            tickers=tickers,
            **(self.config.get('get_metadata') or {})
        )
        ohlcv = data_loader.get_ohlcv(
            tickers=tickers,
            **(self.config.get('get_ohlcv') or {})
        )

        return metadata, ohlcv
