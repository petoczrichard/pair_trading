from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog
from pair_trading.logger.logger_decorator import logger_decorator


class DataLoaderStep(AbstractStep):

    alias = 'data_loader'

    @logger_decorator(
        output_names=('metadata', 'ohlcv'),
        output_formatter={
            'metadata': ('shape',),
            'ohlcv': ('shape',),
        },
    )
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
