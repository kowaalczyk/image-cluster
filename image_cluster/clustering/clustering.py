import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfTransformer


class DbScanClusterizer(object):
    def __init__(
            self,
            feature_transformer=TfidfTransformer(),
            dbscan_kwargs=dict()
    ):
        self.transformer = feature_transformer
        self.dbscan = DBSCAN(**dbscan_kwargs)

    def __call__(self, X):
        transformed_vectors = self.transformer.fit_transform(
            X.astype(np.float32)
        )
        return self.dbscan.fit_predict(transformed_vectors)
