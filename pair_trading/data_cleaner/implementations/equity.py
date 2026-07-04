from pair_trading.data_cleaner.abstract import AbstractDataCleaner


class EquityDataCleaner(AbstractDataCleaner):

    alias = 'equity'

    def clean_tickers(
        self,
        metadata,
        currency_volume,
        fields_to_make_unique=('shortName', 'longName', 'website'),
        **kwargs,
    ):
        metadata = metadata.copy()
        metadata['avg_curr_volume'] = currency_volume.mean().fillna(0)

        for field in fields_to_make_unique:
            metadata = metadata.loc[
                metadata.groupby(field)['avg_curr_volume'].idxmax()
            ]

        return metadata

    def clean_prices(self, prices, dropna_threshold=0.8, **kwargs):
        prices = prices.dropna(
            axis=1,
            thresh=prices.shape[0] * dropna_threshold,
        )
        prices = prices.ffill()
        prices = prices.dropna(axis=0, how='all')
        prices = prices.dropna(axis=1, how='any')

        self._validate_no_missing_prices(prices)

        return prices
