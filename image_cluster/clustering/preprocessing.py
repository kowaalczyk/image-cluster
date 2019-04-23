from typing import Tuple
from itertools import product

import numpy as np
from sklearn.preprocessing import minmax_scale


class FilterVectorizer(object):
    def __init__(
            self,
            filter_shape: Tuple[int, int] = (2, 2),
            return_scaled: bool = True
    ):
        self.filters = self._generate_filters(*filter_shape)
        self.n_filters = len(self.filters)
        self.filter_shape = filter_shape
        self.return_scaled = return_scaled

    def _generate_filters(self, x, y):
        return np.reshape(
            np.arange(2**(x*y))[:, np.newaxis] >> np.arange(x*y)[::-1] & 1,
            (2**(x*y), x, y),
        ).astype(np.uint8) * 255

    def _format_feature_vector(self, feature_vector: np.array) -> np:
        if self.return_scaled:
            return np.hstack([
                feature_vector.astype(np.float),
                minmax_scale(feature_vector.astype(np.float))
            ])
        else:
            return feature_vector


class FilterMatchVectorizer(FilterVectorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        vector = np.zeros(self.n_filters, dtype=np.uint8)
        xrange = np.arange(img.shape[0] - self.filter_shape[0] + 1)
        yrange = np.arange(img.shape[1] - self.filter_shape[1] + 1)
        for x, y in product(xrange, yrange):
            img_chunk = img[x:x+self.filter_shape[0], y:y+self.filter_shape[1]]
            filter_matches = np.all(self.filters == img_chunk, axis=(1, 2))
            vector += filter_matches.astype(np.uint8)
        return self._format_feature_vector(vector)


class FilterIOUVectorizer(FilterVectorizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        vector = np.zeros(self.n_filters, dtype=np.uint8)
        xrange = np.arange(img.shape[0] - self.filter_shape[0] + 1)
        yrange = np.arange(img.shape[1] - self.filter_shape[1] + 1)
        for x, y in product(xrange, yrange):
            img_chunk = img[x:x+self.filter_shape[0], y:y+self.filter_shape[1]]
            intersections = np.sum(self.filters & img_chunk, axis=(1,2))
            unions = np.sum(self.filters & img_chunk, axis=(1,2))
            intersections[unions == 0] = 1
            unions[unions == 0] = 1
            filter_ious = intersections / unions
            vector += filter_ious.astype(np.uint8)
        return self._format_feature_vector(vector)
