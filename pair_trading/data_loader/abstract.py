from abc import ABC, abstractmethod

from pair_trading.catalog import PairTradingMeta


class AbstractDataLoader(ABC, metaclass=PairTradingMeta):

    alias = "data_loader"

    @abstractmethod
    def get_tickers(self, **kwargs):
        pass

    @abstractmethod
    def get_metadata(self, **kwargs):
        pass

    @abstractmethod
    def get_ohlcv(self, **kwargs):
        pass
