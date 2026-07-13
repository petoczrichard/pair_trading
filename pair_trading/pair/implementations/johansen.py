from typing import Literal
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pair_trading.pair.abstract import AbstractPair
from pair_trading.numba_helpers.var import var
from pair_trading.numba_helpers.ljung_box import ljung_box


SIG_TO_COL = {
    0.1: 0,
    0.05: 1,
    0.01: 2,
}


class JohansenPair(AbstractPair):

    alias = 'johansen'

    def calculate(self):
        prices = np.stack(
            [
                self.price1_values[self.formation_period_index],
                self.price2_values[self.formation_period_index],
            ],
            axis=1,
        )

        resids, selected_lag, _ = var(prices, trend='c', ic_method='aic')

        self.ljung_box_pvalues1 = ljung_box(
            resids[:, 0], lags=selected_lag,
        )[1]
        self.ljung_box_pvalues2 = ljung_box(
            resids[:, 1], lags=selected_lag,
        )[1]

        self.johansen_result = coint_johansen(
            prices,
            det_order=0,
            k_ar_diff=max(0, selected_lag - 1),
        )

        beta = self.johansen_result.evec[:, 0]
        self.hedge_ratio = -beta[1] / beta[0]
        self.create_spread()

        return self

    def are_residuals_autocorrelated(
        self,
        significance_level: Literal[0.01, 0.05, 0.1] = 0.05,
    ):
        return (
            (self.ljung_box_pvalues1 < significance_level).any()
            or (self.ljung_box_pvalues2 < significance_level).any()
        )

    def is_rank_one(
        self,
        significance_level: Literal[0.01, 0.05, 0.1] = 0.05,
        method: Literal['trace', 'max_eigen'] = 'trace',
    ):
        johansen_sig_index = SIG_TO_COL[significance_level]

        if method == 'trace':
            stat = self.johansen_result.lr1
            crit_value = self.johansen_result.cvt
        elif method == 'max_eigen':
            stat = self.johansen_result.lr2
            crit_value = self.johansen_result.cvm
        else:
            raise ValueError("Invalid method.")

        stat0 = stat[0]
        stat1 = stat[1]
        crit0 = crit_value[0, johansen_sig_index]
        crit1 = crit_value[1, johansen_sig_index]

        return (stat0 > crit0) and (stat1 < crit1)

    def is_significant(
        self,
        ljung_box_significance_level: Literal[0.01, 0.05, 0.1] = 0.05,
        johansen_significance_level: Literal[0.01, 0.05, 0.1] = 0.05,
        method: Literal['trace', 'max_eigen', 'both', 'either'] = 'trace',
        **kwargs,
    ):
        are_residuals_autocorrelated = self.are_residuals_autocorrelated(
            ljung_box_significance_level,
        )

        if method == 'trace':
            is_rank_one = self.is_rank_one(johansen_significance_level, method)
            return (not are_residuals_autocorrelated) and is_rank_one

        elif method == 'max_eigen':
            is_rank_one = self.is_rank_one(johansen_significance_level, method)
            return (not are_residuals_autocorrelated) and is_rank_one
        
        else:
            is_rank_one_trace = self.is_rank_one(
                johansen_significance_level,
                'trace',
            )
            is_rank_one_max_eigen = self.is_rank_one(
                johansen_significance_level,
                'max_eigen',
            )
            if method == 'both':
                is_rank_one = is_rank_one_trace and is_rank_one_max_eigen
                return (not are_residuals_autocorrelated) and is_rank_one
            else:
                is_rank_one = is_rank_one_trace or is_rank_one_max_eigen
                return (not are_residuals_autocorrelated) and is_rank_one
