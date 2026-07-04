from importlib import resources
import json
import pandas as pd

from pair_trading.data_loader.abstract import AbstractDataLoader


RESOURCE_FOLDER = resources.files("pair_trading.data_store")


class StaticDataSource(AbstractDataLoader):

    alias = "static"

    def get_tickers(self, **kwargs) -> None:
        return None

    def get_metadata(self, market: str, **kwargs) -> pd.DataFrame:
        file_name = RESOURCE_FOLDER / market / "ticker_info.csv"

        metadata = pd.read_csv(file_name)
        metadata.index = metadata['symbol'].rename(None)
        return metadata

    def get_prices(self, market: str, **kwargs) -> pd.DataFrame:
        file_name = RESOURCE_FOLDER / market / "ohlcv.parquet"
        return pd.read_parquet(file_name)
