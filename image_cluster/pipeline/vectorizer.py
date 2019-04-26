from typing import Iterable, Tuple
from itertools import product

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from image_cluster.pipeline.utils import NoFitMixin, VerboseMixin
from image_cluster.types import ImageData


class BaseImageVectorizer(
        BaseEstimator,
        TransformerMixin,
        NoFitMixin,
        VerboseMixin
):
    """
    Base class for transformations:
    [ImageData] -> np.array(n_samples, n_features).
    """
    def __init__(
            self,
            filter_shape: Tuple[int, int] = (3, 3),
            verbose: bool = False
    ):
        self.filter_shape = filter_shape
        self.filters = self._generate_filters(*filter_shape)
        self.n_filters = len(self.filters)
        self.verbose = verbose

    def _generate_filters(self, x: int, y: int) -> np.array:
        """
        Generates all possible (2^{x*y}) boolean matrices.
        """
        return np.reshape(
            np.arange(2**(x*y))[:, np.newaxis] >> np.arange(x*y)[::-1] & 1,
            (2**(x*y), x, y),
        ).astype(np.uint8)

    def transform(self, image_data: Iterable[ImageData]) -> np.array:
        self.vectors_ = np.array([
            self.vectorize(img)
            for img in self._progress(image_data)
        ])
        return self.vectors_

    def vectorize(self, image_data: ImageData) -> ImageData:
        raise NotImplementedError()


class FilterImageVectorizer(BaseImageVectorizer):
    """
    Features: number of identity matches
    between image chunks and vectorizer filters.
    """
    def vectorize(self, image_data):
        img = image_data.image
        vector = np.zeros(self.n_filters, dtype=np.uint8)
        xrange = np.arange(img.shape[0] - self.filter_shape[0] + 1)
        yrange = np.arange(img.shape[1] - self.filter_shape[1] + 1)
        for x, y in product(xrange, yrange):
            img_chunk = img[x:x+self.filter_shape[0], y:y+self.filter_shape[1]]
            filter_matches = np.all(self.filters == img_chunk, axis=(1, 2))
            vector += filter_matches.astype(np.uint8)
        return vector


class IOUImageVectorizer(BaseImageVectorizer):
    """
    Features: sum of intersection-over-union metric values
    between image chunks and vectorizer filters.
    """
    def vectorize(self, image_data):
        img = image_data.image
        vector = np.zeros(self.n_filters, dtype=np.float)
        xrange = np.arange(img.shape[0] - self.filter_shape[0] + 1)
        yrange = np.arange(img.shape[1] - self.filter_shape[1] + 1)
        for x, y in product(xrange, yrange):
            img_chunk = img[x:x+self.filter_shape[0], y:y+self.filter_shape[1]]
            # using non-boolean sum and union for both binary masks
            # and (0,1) floating point range greyscale images
            intersections = np.sum(self.filters * img_chunk, axis=(1, 2))
            unions = np.sum(self.filters + img_chunk, axis=(1, 2))
            # special case handling: 0/0 division => IOU = 1.
            intersections[unions == 0] = 1
            unions[unions == 0] = 1
            filter_ious = intersections / unions
            vector += filter_ious
        return vector


class BaseFeatureGenerator(
        BaseEstimator,
        TransformerMixin,
        NoFitMixin,
        VerboseMixin
):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def transform(self, image_data: Iterable[ImageData]):
        self.features_ = np.array([
            self.compute_feature(img)
            for img in self._progress(image_data)
        ]).reshape(len(image_data), -1)
        return self.features_

    def compute_feature(self, image_data: ImageData) -> ImageData:
        raise NotImplementedError()


class ShapeVectorizer(BaseFeatureGenerator):
    def compute_feature(self, image_data):
        return image_data.image.shape


class TextDensityVectorizer(BaseFeatureGenerator):
    def compute_feature(self, image_data):
        img = image_data.image
        return np.sum(img, axis=(0, 1)) / (
            img.shape[0] * img.shape[1]
        )
