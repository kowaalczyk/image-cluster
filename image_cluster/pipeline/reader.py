from typing import Union, Iterable
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
import cv2 as cv
import numpy as np

from image_cluster.pipeline.utils import NoFitMixin, VerboseMixin
from image_cluster.types import ImageData


class MetadataReader(BaseEstimator, TransformerMixin, NoFitMixin):
    """
    Reads metadata, yielding ImageData.
    Metadata path can be specified either
    during construction or during transform,
    for easier experimenting.
    """
    def __init__(self, meta_path: Union[str, Path] = None):
        self._store_meta(meta_path)

    def transform(
            self,
            meta_path: Union[str, Path] = None
    ) -> Iterable[ImageData]:
        self._store_meta(meta_path)
        return self.metadata_

    def _store_meta(self, meta_path: Union[str, Path]):
        if meta_path is not None:
            with open(meta_path, 'r') as f:
                self.metadata_ = list(
                    map(self.parse_meta_line, f.readlines())
                )

    def parse_meta_line(self, line: str) -> ImageData:
        img_path = Path(line.strip())
        return ImageData(
            img_path.name,
            img_path.resolve()
        )


class BaseImageReader(
        BaseEstimator,
        TransformerMixin,
        NoFitMixin,
        VerboseMixin
):
    """
    Base class for all image readers.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def transform(
            self,
            image_data: Iterable[ImageData]
    ) -> Iterable[ImageData]:
        self.images_ = [
            self.strategy(img)
            for img in self._progress(image_data)
        ]
        return self.images_

    def strategy(self, image_data: ImageData) -> ImageData:
        raise NotImplementedError()


class MaskImageReader(BaseImageReader):
    """
    Reads image to binary mask, 0 = white, 1 = black.
    """
    def strategy(self, image_data):
        img = cv.imread(str(image_data.path), cv.IMREAD_GRAYSCALE)
        _, img = cv.threshold(
            img,
            0,
            255,
            cv.THRESH_BINARY+cv.THRESH_OTSU
        )
        image_data.image = (1 - (img // 255)).astype(np.uint8)
        return image_data


class GreyscaleImageReader(BaseImageReader):
    """
    Reads image as greyscale, 0. = white, 1. = black.
    """
    def strategy(self, image_data):
        img = cv.imread(str(image_data.path), cv.IMREAD_GRAYSCALE)
        image_data.image = 1 - img.astype(np.float) / 255
        return image_data
