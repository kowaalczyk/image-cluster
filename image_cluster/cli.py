from pathlib import Path
import os
import random

import click
from sklearn.cluster import AgglomerativeClustering

from image_cluster.pipeline import (
    default_transformers,
    optimal_clusters,
    Writer,
    ModelFactory,
    Pipeline,
)


@click.group(name='image_cluster')
def main():
    pass


@main.command(name="cluster")
@click.option("-i", "--input-file", type=Path, required=True)
@click.option(
    "-o", "--output-dir", type=Path,
    help="Unless provided, output will be lost.")
@click.option(
    "-n", "--n-clusters", type=int, default=None,
    help="Unless provided, will be autaomatically calculated.")
@click.option(
    "--html/--no-html", default=True,
    help="Whether to save output in html format or only in text format.")
@click.option("--verbose/--silent", default=True)
@click.option(
    "--score/--no-score", default=True,
    help="Calculate metrics after fitting the model.")
def cluster_images(input_file, output_dir, n_clusters, html, verbose, score):
    """
    Main command for clustering images.
    Input file should contain paths to images,
    as described in task specification.
    """
    if output_dir is None:
        writer = None
    else:
        writer = Writer(output_dir, html, verbose)
    preprocessing, postprocessing = default_transformers(writer, verbose)
    if n_clusters is None:
        model_factory = ModelFactory(
            optimal_clusters,
            AgglomerativeClustering,
            affinity='euclidean',
            linkage='ward'
        )
    else:
        model_factory = ModelFactory.fixed_clusters(
            n_clusters,
            AgglomerativeClustering,
            affinity='euclidean',
            linkage='ward'
        )
    pipeline = Pipeline(
        preprocessing,
        model_factory,
        postprocessing,
        verbose
    )
    pipeline.fit_predict(input_file)
    if score:
        pipeline.score()
    return pipeline  # for use in python scripts


@main.command(name="meta")
@click.argument("images_dir", type=Path, required=True)
@click.option("-o", "--output-file", type=Path, required=True)
@click.option("-n", "--n-images", type=int, default=-1)
def generate_meta(images_dir, output_file, n_images):
    """
    Helper method for generating metadata from a folder of images.
    Generated metadata is compatible with the task specification.
    """
    images_dir = images_dir.resolve(strict=True)
    img_names = os.listdir(images_dir)
    if n_images > 0:
        img_names = random.sample(img_names, n_images)
    meta = [str(images_dir / img) + '\n' for img in img_names]
    with open(output_file, 'w') as f:
        f.writelines(meta)
    return meta  # for use in python scripts


if __name__ == '__main__':
    main()
