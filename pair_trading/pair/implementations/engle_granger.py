from pair_trading.numba_helpers.statistical import rma_regression
from pair_trading.numba_helpers.adf import adf
from pair_trading.pair.abstract import AbstractPair


class EngleGrangerPair(AbstractPair):

    alias = 'engle_granger'

    def calculate(
        self,
        check_stationary_assets: bool = True,
    ) -> None:
        price1 = self.price1_values[self.formation_period_index]
        price2 = self.price2_values[self.formation_period_index]

        if check_stationary_assets:
            self.price1_adf_pvalue = adf(price1)[1]
            self.price2_adf_pvalue = adf(price2)[1]

        slope, _ = rma_regression(
            x=price1,
            y=price2,
        )
        self.hedge_ratio = 1 / slope
        self.create_spread()
        self.spread_adf_pvalue = adf(
            self.spread_values[self.formation_period_index]
        )[1]

        return self

    def is_significant(self, significance_level: int = 0.05, **kwargs):
        return self.spread_adf_pvalue < significance_level
