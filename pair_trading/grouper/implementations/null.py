from pair_trading.grouper.abstract import AbstractGrouper


class NullGrouper(AbstractGrouper):

    alias = 'null'

    def create_groups(self, metadata, **kwargs):
        return [metadata['symbol'].tolist()]
