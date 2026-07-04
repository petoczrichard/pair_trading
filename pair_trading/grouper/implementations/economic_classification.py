from pair_trading.grouper.abstract import AbstractGrouper


class EconomicClassificationGrouper(AbstractGrouper):

    alias = 'economic_classification'

    def create_groups(self, metadata, group_by_field='sector', **kwargs):
        return [
            group_data.index.tolist()
            for _, group_data
            in metadata.groupby(group_by_field)
        ]
