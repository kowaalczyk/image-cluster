from typing import Iterable, Callable
from time import time

import numpy as np
from tqdm import tqdm

from image_cluster.data.types import ClusterData
from image_cluster.data import Reader


class Pipeline(object):
    def __init__(self, reader: Reader, preprocessor):
        self.data = reader.metadata
        self.preprocessed = np.array([
            preprocessor(img)
            for img in tqdm(reader, desc="Preprocessing")
        ])

    def group_clusters(self) -> Iterable[ClusterData]:
        clusters = {
            c: ClusterData(c, [])
            for c in np.unique(self.clustered)
        }
        for idx, img_data in tqdm(
                enumerate(self.data),
                desc="Grouping clusters"
        ):
            cluster = self.clustered[idx]
            img_data.cluster = cluster
            clusters[cluster].images.append(img_data)
        return [clusters[k] for k in clusters]

    def process(
            self,
            clusterizer: Callable[[np.array], np.array]
    ) -> Iterable[ClusterData]:
        print("Clustering ...")
        start = time()
        self.clustered = clusterizer(self.preprocessed)
        end = time()
        print(f"Clustering finshed in {end-start:.2f} seconds.")
        self.clusters = self.group_clusters()
        print(f"Clustering resulted in {len(self.clusters)} clusters.")
        return self.clusters
