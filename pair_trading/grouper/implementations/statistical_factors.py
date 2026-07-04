import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS

from pair_trading.grouper.abstract import AbstractGrouper
from pair_trading.utils import filter_allowed_kwargs


class StatisticalFactorsGrouper(AbstractGrouper):

    alias = 'statistical_factor'

    def create_groups(self, returns, **kwargs) -> list[list[str]]:
        pca = PCA(**filter_allowed_kwargs(PCA, kwargs))
        pca.fit(returns)

        clustering = OPTICS(**filter_allowed_kwargs(OPTICS, kwargs))
        clustering.fit(pca.components_.T)

        clusters = pd.DataFrame(
            clustering.labels_,
            index=returns.columns,
            columns=['Cluster'],
        )
        clusters = clusters[clusters['Cluster'] != -1]

        return [
            cluster_data.index.tolist()
            for _, cluster_data
            in clusters.groupby('Cluster')
        ]
