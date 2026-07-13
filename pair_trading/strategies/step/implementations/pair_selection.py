from os import cpu_count
from concurrent.futures import ProcessPoolExecutor

from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog


class PairSelectionStep(AbstractStep):

    alias = 'pair_selection'

    def __init__(self, config):
        super().__init__(config=config)

        multiprocess_n_jobs = config.get('multiprocess_n_jobs', 1)
        multiprocess_n_jobs = multiprocess_n_jobs or max(1, cpu_count() - 1)

        self.pool = (
            ProcessPoolExecutor(max_workers=multiprocess_n_jobs)
            if multiprocess_n_jobs != 1
            else None
        )

    def run(
        self,
        pair_names,
        prices,
        formation_start,
        formation_end,
        trading_start,
        trading_end,
    ):
        pair_selection = PairTradingCatalog.invoke(
            name='pair_selection',
            pair_names=pair_names,
            prices=prices,
            formation_start=formation_start,
            formation_end=formation_end,
            trading_start=trading_start,
            trading_end=trading_end,
            **self.config['setup'],
        )

        pair_selection.calculate_pairs(
            pool=self.pool,
            **(self.config.get('calculate_pairs') or {}),
        )

        filtered_pairs = pair_selection.filter_pairs(
            **(self.config.get('filter_pairs') or {}),
        )

        return filtered_pairs
