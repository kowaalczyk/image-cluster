from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import numpy as np

from image_cluster.pipeline.utils import VerboseMixin
from image_cluster.types import Score


class Pipeline(BaseEstimator, ClusterMixin, VerboseMixin):
    """
    Wrapper for preprocessing pipeline, clustering model and metrics function.
    """
    def __init__(
            self,
            preprocessing: TransformerMixin,
            model: ClusterMixin,
            verbose: bool = False
    ):
        self.preprocessing = preprocessing
        self.model = model
        self.verbose = verbose

    def fit(self, X):
        self._log("Preprocessing...")
        self.preprocessed_ = self.preprocessing.fit_transform(X)
        self._log("Fitting model...")
        self.model.fit(self.preprocessed_)
        return self

    def predict(self, X=None):
        self._log("Assigning clusters...")
        if X is not None:
            self.preprocessed_ = self.preprocessing.transform(X)
        self.labels_ = self.model.fit_predict(self.preprocessed_)
        return self.labels_

    def score(self):
        self._log("Scoring...")
        normal_labels = self.labels_[self.labels_ >= 0]
        label_value_counts = np.bincount(normal_labels)
        outliers = self.labels_[self.labels_ == -1]
        self.score_ = Score(
            silhouette_score=silhouette_score(
                self.preprocessed_,
                self.labels_
            ),
            calinski_harabas_score=calinski_harabaz_score(
                self.preprocessed_,
                self.labels_
            ),
            n_samples=len(self.preprocessed_),
            n_clusters=self.model.n_clusters,
            n_outliers=len(outliers),
            label_size_min=np.min(label_value_counts),
            label_size_max=np.max(label_value_counts),
            label_size_mean=np.mean(label_value_counts),
            label_size_var=np.var(label_value_counts)
        )
        self._log(self.score_)
        return self.score_
