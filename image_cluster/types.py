from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


Image = np.array
ImageShape = Tuple[int, int]


@dataclass
class ImageData(object):
    name: str
    path: Path
    image: Image = None
    cluster: int = None

    def show(self):
        printable_image = ((1 - self.image) * 255).astype(np.unit8)
        plt.title(f"Image {self.name} in cluster {self.cluster}")
        plt.imshow(printable_image)
        plt.show()


@dataclass
class ClusterData(object):
    idx: int
    images: List[ImageData] = field(default_factory=list)

    def show(self, n_samples: int = -1):
        if n_samples == -1:
            last_sample = len(self.images)
        else:
            last_sample = min(len(self.images, n_samples))
        for img in self.images[:last_sample]:
            img.show()


@dataclass
class Score(object):
    silhouette_score: float
    calinski_harabaz_score: float
    n_samples: int
    n_clusters: int
    n_outliers: int
    label_size_min: int
    label_size_max: int
    label_size_mean: float
    label_size_var: float

    def __str__(self):
        title = f"{self.__class__.__name__}:"
        dictionary = asdict(self)
        metrics = [
            f"\t{key}: {_format(dictionary[key])}"
            for key in dictionary
        ]
        r = [title] + metrics
        return "\n".join(r)


def _format(num):
    if 'float' in str(type(num)):
        return f"{num:.2f}"
    else:
        return str(num)
