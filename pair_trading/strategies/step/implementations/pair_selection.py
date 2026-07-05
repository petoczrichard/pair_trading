from pair_trading.strategies.step.abstract import AbstractStep
from pair_trading.catalog import PairTradingCatalog


class PairSelectionStep(AbstractStep):

    alias = 'pair_selection'

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
            pair_names=pair_names,
            prices=prices,
            formation_start=formation_start,
            formation_end=formation_end,
            trading_start=trading_start,
            trading_end=trading_end,
            **self.config['setup'],
        )

        pair_selection.calculate_pairs(
            **(self.config.get('calculate_pairs') or {}),
        )

        filtered_pairs = pair_selection.filter_pairs(
            **(self.config.get('filter_pairs') or {}),
        )

        return filtered_pairs
