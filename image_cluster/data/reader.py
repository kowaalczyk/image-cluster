from typing import Iterable, Callable, Tuple
from pathlib import Path

import cv2 as cv
from sklearn.preprocessing import minmax_scale

from image_cluster.data.types import Image, ImageData


def parse_metadata_line(line: str) -> ImageData:
    img_path = Path(line.strip())
    return ImageData(
        img_path.stem,
        img_path
    )


def read_grayscale_mask(img_path: Path) -> Image:
    img_path = img_path.resolve(strict=True)
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    _, img = cv.threshold(
        img,
        0,
        255,
        cv.THRESH_BINARY+cv.THRESH_OTSU
    )
    return img


def read_binary_image(img_path: Path) -> Image:
    img_path = img_path.resolve(strict=True)
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    shape = img.shape
    img = img.ravel().minmax_scale(
        img,
        feature_range=(0,1)
    ).reshape(shape)
    return img


class Reader(object):
    def __init__(
            self,
            input_file_path: Path,
            meta_strategy: Callable[[str], ImageData] = parse_metadata_line,
            strategy: Callable[[Path], Image] = read_grayscale_mask
    ):
        self.strategy = strategy
        with open(input_file_path, 'r') as f:
            self.metadata = list(
                map(meta_strategy, f.readlines())
            )
            self.img_idx = 0

    def __iter__(self) -> Iterable[Image]:
        self.img_idx = 0
        return self

    def __next__(self) -> Image:
        try:
            img_data = self.metadata[self.img_idx]
        except IndexError:
            raise StopIteration
        img = self.strategy(img_data.path)
        self.img_idx += 1
        return img

    def __len__(self):
        return len(self.metadata)
