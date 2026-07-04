import numpy as np

from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog

from trading_core import Basket, TradeSource, BacktesterEngine


class BacktestStep(AbstractStep):

    alias = 'backtest'

    def run(
        self,
        ohlcv,
        trade_sources,
    ):
        backtest_tickers = set()

        for ts in trade_sources:
            if (basket := ts.get('basket')):
                asset_ids = basket['asset_ids']
                backtest_tickers.add(asset_ids[0])
                backtest_tickers.add(asset_ids[1])

        backtest_start_date = min(ts['date'] for ts in trade_sources)
        backtest_end_date = max(ts['date'] for ts in trade_sources)

        ohlcv = ohlcv.loc[
            backtest_start_date:backtest_end_date,
            list(backtest_tickers),
        ]

        prices = ohlcv.xs('Close', level=1, axis=1)

        source_id_to_index = {
            source_id: index
            for index, source_id
            in enumerate(set(ts['source_id'] for ts in trade_sources))
        }
        ticker_to_index = {
            ticker: index
            for index, ticker
            in enumerate(prices.columns)
        }
        date_to_index = {
            date: index
            for index, date
            in enumerate(prices.index)
        }

        trade_sources_cpp = [
            TradeSource(
                source_id=source_id_to_index[trade_source['source_id']],
                type=trade_source['type'],
                date=date_to_index[trade_source['date']],
                basket=(
                    Basket(
                        **(
                            trade_source['basket'] | {
                                'asset_ids': [
                                    ticker_to_index[
                                        (asset_ids :=trade_source[
                                            'basket'
                                        ]['asset_ids'])[0]
                                    ],
                                    ticker_to_index[asset_ids[1]],
                                ]
                            }
                        )
                    )
                    if trade_source.get('basket') is not None
                    else None
                ),
                max_adv_participation_rate=self.config[
                    'max_adv_participation_rate'
                ],
                max_ratio_of_portfolio_value=(
                    trade_source['max_ratio_of_portfolio_value']
                    / self.config['overlapping_periods']
                ),
            )
            for trade_source
            in trade_sources
        ]

        if 'transaction_cost' not in self.config:
            adv = np.ones_like(prices) * 1e100
            tc_model = PairTradingCatalog.create_instance(
                name='transaction_cost_model',
                open_prices=None,
                high_prices=None,
                low_prices=None,
                close_prices=None,
                currency_volume=None,
            )
            tc_engine = tc_model.make_engine(
                commision=0.0,
                borrowing_cost=0.0,
                spread=np.zeros_like(prices),
                slippage=np.zeros_like(prices),
                market_impact=np.zeros_like(prices),
                adv=adv,
            )
        else:
            tc_config = self.config['transaction_cost']

            open_prices = ohlcv.xs('Open', level=1, axis=1)
            high_prices = ohlcv.xs('High', level=1, axis=1)
            low_prices = ohlcv.xs('Low', level=1, axis=1)
            close_prices = ohlcv.xs('Close', level=1, axis=1)
            currency_volume = (
                close_prices * ohlcv.xs('Volume', level=1, axis=1)
            )

            tc_model = PairTradingCatalog.create_instance(
                name='transaction_cost_model',
                open_prices=open_prices,
                high_prices=high_prices,
                low_prices=low_prices,
                close_prices=close_prices,
                currency_volume=currency_volume,
            )
            adv = tc_model.adv(
                **(tc_config.get('adv') or {}),
            ).ffill().bfill().values

            tc_engine = tc_model.make_engine(
                commision=tc_config.get('commision', 0),
                borrowing_cost=tc_config.get('borrowing_cost', 0),
                spread=tc_model.calibrate_spread_cost(
                    **(tc_config.get('spread') or {}),
                ).fillna(0).values,
                slippage=tc_model.calibrate_slippage_cost(
                    **(tc_config.get('slippage') or {}),
                ).fillna(0).values,
                market_impact=tc_model.calibrate_impact_cost(
                    **(tc_config.get('market_impact') or {}),
                ).fillna(0).values,
                adv=adv,
            )

        backtester = BacktesterEngine(
            initial_portfolio_value=self.config['initial_portfolio_value'],
            leverage=self.config['leverage'],
            prices=prices.values,
            adv=adv,
            trade_sources=trade_sources_cpp,
            tc_engine=tc_engine,
        )
        backtester.run()

        return backtester
