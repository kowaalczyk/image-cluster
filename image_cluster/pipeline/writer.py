import os
from pathlib import Path
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

from image_cluster.pipeline.utils import NoFitMixin, VerboseMixin
from image_cluster.types import ClusterData


class Writer(BaseEstimator, TransformerMixin, VerboseMixin, NoFitMixin):
    """
    Composite class in composite pattern of writers.
    """
    def __init__(self, output_dir: Path, html: bool, verbose: bool = False):
        output_dir = output_dir.resolve()
        if not output_dir.exists():
            os.makedirs(output_dir)
        self.writers = [TxtWriter(
            output_dir / 'clusters.txt',
            verbose=verbose
        )]
        if html:
            self.writers.append(HtmlWriter(
                output_dir / 'clusters.html',
                verbose=verbose
            ))

    def transform(self, clusters: Iterable[ClusterData]):
        for writer in self.writers:
            writer.transform(clusters)


class TxtWriter(VerboseMixin):
    """
    Child writer component, saves cluster data as txt file.
    """
    def __init__(self, output_file: Path, verbose: bool = False):
        self.output_file = output_file
        self.verbose = verbose

    def _format(self, cluster: ClusterData) -> str:
        return " ".join(
            [img.name for img in cluster.images]
        )

    def transform(self, clusters: Iterable[ClusterData]):
        with open(self.output_file, 'w') as f:
            f.writelines([
                self._format(cluster)
                for cluster in self._progress(clusters)
            ])


class HtmlWriter(VerboseMixin):
    """
    Child writer component, saves cluster data as html file.
    """
    def __init__(self, output_file: Path, verbose: bool = False):
        self.output_file = output_file
        self.verbose = verbose

    def _format(self, cluster: ClusterData) -> str:
        imgs = [
            f'<img src={str(img.path)} alt={str(img.name)}>'
            for img in cluster.images
        ]
        return " ".join(imgs) + "\n"

    def transform(self, clusters: Iterable[ClusterData]):
        with open(self.output_file, 'w') as f:
            f.write("<hr>\n".join([
                self._format(cluster)
                for cluster in self._progress(clusters)
            ]))
            f.write("\n")
