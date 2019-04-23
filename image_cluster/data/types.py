from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np


Image = np.array
ImageShape = Tuple[int, int]


@dataclass
class ImageData(object):
    name: str
    path: Path
    cluster: int = None


@dataclass
class ClusterData(object):
    idx: int
    images: List[ImageData]
