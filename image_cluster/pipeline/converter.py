from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from image_cluster.pipeline.utils import VerboseMixin
from image_cluster.pipeline.reader import BaseImageReader
from image_cluster.types import ClusterData


class Converter(
        BaseEstimator,
        TransformerMixin,
        VerboseMixin
):
    """
    Converts raw cluster data to ClusterData objects, updating
    initially read images with cluster ids in the process.
    """
    def __init__(self, image_reader: BaseImageReader, verbose: bool = False):
        self.image_reader = image_reader
        self.verbose = verbose

    def fit(self, raw_cluster_data: np.array):
        """
        Computes number of clusters
        """
        self.cluster_data_ = {
            idx: ClusterData(idx)
            for idx in np.unique(raw_cluster_data)
        }
        self.raw_cluster_data_ = raw_cluster_data
        return self

    def transform(
            self,
            raw_cluster_data: np.array = None
    ) -> List[ClusterData]:
        if raw_cluster_data is not None:
            # re-fit the model in case there are new clusters
            self.fit(raw_cluster_data)
        for image, cluster_id in self._progress(zip(
                self.image_reader.images_,
                self.raw_cluster_data_
        )):
            image.cluster = cluster_id
            self.cluster_data_[cluster_id].images.append(image)
        return [self.cluster_data_[key] for key in self.cluster_data_]
