import numpy as np

from pair_trading.data_cleaner.abstract import AbstractDataCleaner


class CryptoDataCleaner(AbstractDataCleaner):

    alias = 'crypto'

    def clean_tickers(
        self,
        metadata,
        **kwargs,
    ):
        cashlike_cryptos = (
            metadata['symbol']
            .str.split('-')
            .str[0]
            .str.contains('USD')
        )

        return metadata.loc[~cashlike_cryptos]

    def clean_prices(
        self,
        prices,
        volumes,
        dropna_threshold=0.8,
        max_corr_threshold=0.99,
        **kwargs,
    ):
        prices = prices.dropna(
            axis=1,
            thresh=prices.shape[0] * dropna_threshold,
        )
        prices = prices.ffill()
        prices = prices.dropna(axis=0, how='all')
        prices = prices.dropna(axis=1, how='any')

        self._validate_no_missing_prices(prices)

        volume_order = (
            volumes[prices.columns]
            .mean()
            .sort_values(ascending=False)
        )
        corr = prices[volume_order.index].corr()
        corr_lower = corr.mask(
            np.triu(np.ones(corr.shape), k=0).astype(bool)
        )

        correlated_cryptos = (corr_lower > max_corr_threshold).any(axis=1)
        not_correlated_cryptos = correlated_cryptos[~correlated_cryptos]

        return prices[not_correlated_cryptos.index]
