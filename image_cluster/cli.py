from pathlib import Path
import os
import random

import click

from image_cluster.data import Reader, Writer
from image_cluster.clustering import (
    Pipeline,
    vectorizer_factory,
    clusterizer_factory
)


@click.group(name='image_cluster')
def main():
    pass


@main.command(
    name="cluster",
    help="Cluster images listed in input file"
)
@click.option("-i", "--input-file", type=Path, required=True)
@click.option("-o", "--output-dir", type=Path)
@click.option("--html/--no-html", default=True)
def cluster_images(input_file, output_dir, html):
    reader = Reader(input_file)
    vectorizer = vectorizer_factory()
    clustering_pipeline = Pipeline(reader, vectorizer)
    clusterizer = clusterizer_factory(len(reader))
    clusters = clustering_pipeline.process(clusterizer)
    if output_dir is not None:
        writer = Writer(output_dir, html)
        writer.write(clusters)
    return clusters


@main.command(
    name="meta",
    help="Generate metadata based on supplied directory of images"
)
@click.argument("images_dir", type=Path, required=True)
@click.option("-o", "--output-file", type=Path, required=True)
@click.option("-n", "--n-images", type=int, default=-1)
def generate_meta(images_dir, output_file, n_images):
    images_dir = images_dir.resolve(strict=True)
    img_names = os.listdir(images_dir)
    if n_images > 0:
        img_names = random.sample(img_names, n_images)
    meta = [str(images_dir / img) + '\n' for img in img_names]
    with open(output_file, 'w') as f:
        f.writelines(meta)
    return meta


if __name__ == '__main__':
    main()
