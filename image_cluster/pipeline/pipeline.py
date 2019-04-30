from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import silhouette_score, calinski_harabaz_score
import numpy as np

from image_cluster.pipeline.utils import VerboseMixin
from image_cluster.pipeline.model_factory import ModelFactory
from image_cluster.types import Score


class Pipeline(BaseEstimator, ClusterMixin, VerboseMixin):
    """
    Wrapper for preprocessing pipeline, clustering model,
    metric calculation and postprocessing.
    """
    def __init__(
            self,
            preprocessing: TransformerMixin,
            model_factory: ModelFactory,
            postprocessing: TransformerMixin,
            verbose: bool = False
    ):
        self.preprocessing = preprocessing
        self.model_factory = model_factory
        self.postprocessing = postprocessing
        self.verbose = verbose

    def fit(self, X):
        """
        Builds a model and fits it on preprocessed data.
        After fitting, postprocessing is performed,
        however predict and fit_predict methods return model labels
        (without postprocessing) to keep consistency with
        scikt-learn estimators API.
        """
        self._log("Preprocessing...")
        self.preprocessed_ = self.preprocessing.fit_transform(X)
        self._log("Creating model...")
        self.n_samples_ = len(self.preprocessed_)
        self.model_ = self.model_factory(self.n_samples_)
        self._log("Fitting model...")
        self.model_.fit(self.preprocessed_)
        self.labels_ = self.model_.fit_predict(self.preprocessed_)
        self._log("Postprocessing...")
        self.postprocessed_ = self.postprocessing.fit_transform(
            self.labels_
        )
        return self

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
            calinski_harabaz_score=calinski_harabaz_score(
                self.preprocessed_,
                self.labels_
            ),
            n_samples=len(self.preprocessed_),
            n_clusters=self.model_.n_clusters,
            n_outliers=len(outliers),
            label_size_min=np.min(label_value_counts),
            label_size_max=np.max(label_value_counts),
            label_size_mean=np.mean(label_value_counts),
            label_size_var=np.var(label_value_counts)
        )
        self._log(self.score_)
        return self.score_
