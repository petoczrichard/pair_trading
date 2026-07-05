from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog


class DataCleanerStep(AbstractStep):

    alias = 'data_cleaner'

    def run(
        self,
        metadata,
        ohlcv,
        formation_start,
        formation_end,
        trading_end,
    ):
        is_crypto = self.config['setup']['variant'] == 'crypto'

        prices = ohlcv.loc[formation_start:trading_end].xs(
            'Close', level=1, axis=1,
        )
        volumes = ohlcv.loc[formation_start:trading_end].xs(
            'Volume', level=1, axis=1,
        )
        currency_volume = volumes * (1 if is_crypto else prices)

        data_cleaner = PairTradingCatalog.invoke(
            **self.config['setup'],
        )

        filtered_metadata = data_cleaner.clean_tickers(
            metadata=metadata,
            currency_volume=currency_volume,
            **(self.config['clean_tickers'] or {}),
        )

        filtered_prices = data_cleaner.clean_prices(
            prices=prices[filtered_metadata.index.tolist()],
            volumes=currency_volume[filtered_metadata.index.tolist()],
            **(self.config['clean_prices'] or {}),
        )

        if 'remove_negative_prices' in self.config:
            filtered_prices = data_cleaner.remove_negative_prices(
                prices=filtered_prices,
                formation_start=formation_start,
                formation_end=formation_end,
                **(self.config['remove_negative_prices'] or {}),
            )

        if 'remove_too_many_zero_returns' in self.config:
            filtered_prices = data_cleaner.remove_too_many_zero_returns(
                prices=filtered_prices,
                formation_start=formation_start,
                formation_end=formation_end,
                **(self.config['remove_too_many_zero_returns'] or {}),
            )

        if 'remove_stationary_prices' in self.config:
            filtered_prices = data_cleaner.remove_stationary_prices(
                prices=filtered_prices,
                formation_start=formation_start,
                formation_end=formation_end,
                **(self.config['remove_stationary_prices'] or {}),
            )

        if 'minimum_liquidity' in self.config:
            filtered_prices = data_cleaner.minimum_liquidity(
                prices=filtered_prices,
                currency_volume=currency_volume[filtered_prices.columns],
                formation_start=formation_start,
                formation_end=formation_end,
                **(self.config['minimum_liquidity'] or {}),
            )

        filtered_metadata = filtered_metadata.loc[filtered_prices.columns]

        return filtered_metadata, filtered_prices
