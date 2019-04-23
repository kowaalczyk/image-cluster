from .pipeline import Pipeline
from .preprocessing import *
from .clustering import DbScanClusterizer

from sklearn.preprocessing import StandardScaler


def vectorizer_factory():
    return FilterIOUVectorizer(filter_shape=(3, 3), return_scaled=True)


def clusterizer_factory(n_images: int):
    return DbScanClusterizer(
        feature_transformer=StandardScaler(),
        dbscan_kwargs={
            'eps': 0.1,
            'min_samples': n_images / 50
    })
