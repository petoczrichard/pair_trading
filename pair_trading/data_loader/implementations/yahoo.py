import time
import random
import requests
import pandas as pd
import yfinance as yf

from pair_trading.data_loader.abstract import AbstractDataLoader


HEADERS = {"User-Agent": "Mozilla/5.0"}

SCREEN_MAP = {
    "us_large_cap": "large_cap_stocks",
    "crypto": "all_cryptocurrencies_us",
}


class YahooDataSource(AbstractDataLoader):

    alias = "yahoo"

    def get_tickers(
        self,
        screen="us_large_cap",
        min_market_cap=None,
        only_tickers=True,
    ):
        min_market_cap = min_market_cap or float("-inf")

        base_url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"  # noqa: E501
        params = {
            "scrIds": SCREEN_MAP[screen],
            "count": 100,
            "start": 0
        }

        session = requests.Session()
        session.headers.update(HEADERS)

        all_rows = []
        while True:
            r = session.get(base_url, params=params)
            r.raise_for_status()

            data = r.json()
            quotes = data["finance"]["result"][0]["quotes"]

            if not quotes:
                break

            all_rows.extend(quotes)
            params["start"] += params["count"]

        df = pd.DataFrame(all_rows)
        df = df[df["marketCap"] >= min_market_cap]

        return df['symbol'].tolist() if only_tickers else df

    def get_metadata(self, tickers: list[str]) -> pd.DataFrame:
        ticker_info = {}

        for symbol in tickers:
            ticker = yf.Ticker(symbol)
            ticker_info[symbol] = ticker.info

            time.sleep(random.uniform(2, 3))

        return pd.DataFrame(ticker_info).T

    def get_prices(self, tickers: list[str], step: int = 50) -> pd.DataFrame:
        start = 0
        data_list = []

        while True:
            batch = tickers[start:start + step]
            if not batch:
                break

            data = yf.download(
                batch,
                period="max",
                group_by='ticker',
                threads=True,
            )
            start += step
            data_list.append(data)

            time.sleep(random.uniform(20, 30))

        return pd.concat(data_list, axis=1)
