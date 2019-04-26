import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from image_cluster.types import ClusterData


class Writer(object):
    def __init__(self, output_dir: Path, html: bool):
        output_dir = output_dir.resolve()
        if not output_dir.exists():
            os.makedirs(output_dir)
        self.writers = [TxtWriter(output_dir / 'clusters.txt')]
        if html:
            self.writers.append(HtmlWriter(output_dir / 'clusters.html'))

    def write(self, clusters: Iterable[ClusterData]):
        for writer in self.writers:
            writer.write(clusters)


class TxtWriter(object):
    def __init__(self, output_file: Path):
        self.output_file = output_file

    def _format(self, cluster: ClusterData) -> str:
        return " ".join(
            [img.name for img in cluster.images]
        )

    def write(self, clusters: Iterable[ClusterData]):
        with open(self.output_file, 'w') as f:
            f.writelines([
                self._format(cluster)
                for cluster in tqdm(
                    clusters,
                    desc=f"Writing {self.output_file} ..."
                )
            ])


class HtmlWriter(object):
    def __init__(self, output_file: Path):
        self.output_file = output_file

    def _format(self, cluster: ClusterData) -> str:
        imgs = [
            f'<img src={str(img.path)} alt={str(img.name)}>'
            for img in cluster.images
        ]
        return " ".join(imgs) + "\n"

    def write(self, clusters: Iterable[ClusterData]):
        with open(self.output_file, 'w') as f:
            f.write("<hr>\n".join([
                self._format(cluster)
                for cluster in tqdm(
                    clusters,
                    desc=f"Writing {self.output_file} ..."
                )
            ]))
            f.write("\n")
