from typing import Callable

from sklearn.base import ClusterMixin


class ModelFactory(object):
    """
    Factory for creating models, with number of clusters calculated
    based on to the number of samples in training data using
    provided startegy ('samples_to_clusters')
    """
    def __init__(
            self,
            samples_to_clusters: Callable[[int], int],
            model_class: ClusterMixin,
            *model_args,
            **model_kwargs
    ):
        self.samples_to_clusters = samples_to_clusters
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    @classmethod
    def fixed_clusters(
            cls,
            model_class: ClusterMixin,
            n_clusters: int,
            *model_args,
            **model_kwargs
    ):
        return cls(
            lambda _: n_clusters,
            model_class,
            *model_args,
            **model_kwargs
        )

    def __call__(self, n_samples: int):
        return self.model_class(
            n_clusters=self.samples_to_clusters(n_samples),
            *self.model_args,
            **self.model_kwargs
        )
