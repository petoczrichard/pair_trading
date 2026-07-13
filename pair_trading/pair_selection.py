import itertools
import numpy as np

from pair_trading.catalog import PairTradingCatalog
from pair_trading.utils import filter_allowed_kwargs


def _calculate_pair_worker(pair, kwargs):
    pair.calculate(**kwargs)
    return pair


class PairSelection(metaclass=PairTradingCatalog):

    alias = 'pair_selection'

    def __init__(
        self,
        pair_names,
        pair_type,
        prices,
        formation_start,
        formation_end,
        trading_start,
        trading_end,
    ):
        self.pairs = [
            PairTradingCatalog.invoke(
                category='pair',
                variant=pair_type,
                price1=prices[name1],
                price2=prices[name2],
                formation_start=formation_start,
                formation_end=formation_end,
                trading_start=trading_start,
                trading_end=trading_end,
            )
            for name1, name2
            in pair_names
        ]

    def calculate_pairs(
        self,
        pool=None,
        **kwargs,
    ):
        if pool is None:
            for pair in self.pairs:
                pair.calculate(**kwargs)
            return

        self.pairs = list(
            pool.map(
                _calculate_pair_worker,
                self.pairs,
                itertools.repeat(kwargs),
                chunksize=100,
            )
        )

    def filter_pairs(
        self,
        filters: list[dict],
    ):
        filtered_pairs = self.pairs

        for filter_ in filters:
            match filter_:

                case {"type": "individual"}:
                    filtered_pairs = [
                        pair
                        for pair
                        in filtered_pairs
                        if self._resolve_filter_to_bool(pair=pair, **filter_)
                    ]

                case {"type": "global"}:
                    filtered_pairs = PairTradingCatalog.invoke(
                        pairs=filtered_pairs,
                        **filter_,
                    )

                case _:
                    raise ValueError(
                        f"Unknown filter type: {filter_['type']}."
                    )

        return filtered_pairs

    @staticmethod
    def _resolve_filter_to_bool(
        pair,
        *,
        property_name,
        min_limit=float("-inf"),
        max_limit=float("inf"),
        **kwargs,
    ):
        property_ = getattr(pair, property_name)

        if callable(property_):
            property_ = property_(**filter_allowed_kwargs(property_, kwargs))

        if not isinstance(property_, (bool, np.bool_)):
            property_ = min_limit <= property_ <= max_limit

        return property_
