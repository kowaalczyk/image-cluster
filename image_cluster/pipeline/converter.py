from typing import Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from image_cluster.pipeline.utils import VerboseMixin
from image_cluster.types import ImageData, ClusterData


class Converter(
        BaseEstimator,
        TransformerMixin,
        VerboseMixin
):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(
            self,
            image_data: Iterable[ImageData],
            raw_cluster_data: np.array,
    ):
        self.cluster_data_ = {
            idx: ClusterData(idx)
            for idx in np.unique(raw_cluster_data)
        }
        self.raw_cluster_data_ = raw_cluster_data
        return self

    def transform(
            self,
            image_data: Iterable[ImageData],
    ) -> List[ClusterData]:
        for image, cluster_id in self._progress(zip(
                image_data,
                self.raw_cluster_data_
        )):
            image.cluster = cluster_id
            self.cluster_data_[cluster_id].images.append(image)
        return [self.cluster_data_[key] for key in self.cluster_data_]
