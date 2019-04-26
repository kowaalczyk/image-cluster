from pathlib import Path
import os
import random

import click

from image_cluster.pipeline import (
    Pipeline,
    SklearnPipeline,
    MetadataReader,
    GreyscaleImageReader,
    IOUImageVectorizer,
    ShapeVectorizer,
    TextDensityVectorizer,
    FeatureUnion,
    MinMaxScaler,
    AgglomerativeClustering,
    Converter
)
from image_cluster.writer import Writer  # TODO: Move to pipeline


@click.group(name='image_cluster')
def main():
    pass


@main.command(
    name="cluster",
    help="Cluster images listed in input file"
)
@click.option("-i", "--input-file", type=Path, required=True)
@click.option("-o", "--output-dir", type=Path)
@click.option("-n", "--n-clusters", type=int)
@click.option("--html/--no-html", default=True)
@click.option("--verbose/--silent", default=True)
@click.option("--score/--no-score", default=True)
def cluster_images(input_file, output_dir, n_clusters, html, verbose, score):
    transformer = SklearnPipeline([
        ('meta', MetadataReader()),
        ('image', GreyscaleImageReader(verbose=verbose)),
        ('vectorizer', FeatureUnion([
            ('iou', IOUImageVectorizer(filter_shape=(3, 3), verbose=verbose)),
            ('shape', ShapeVectorizer(verbose=verbose)),
            ('density', TextDensityVectorizer(verbose=verbose))
        ])),
        ('scaler', MinMaxScaler()),
    ])
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',
        linkage='ward'
    )
    pipeline = Pipeline(transformer, model, verbose=verbose).fit(
        input_file
    )  # TODO: Get rid of Pipeline class, add automatic cluster number selection
    clusters = Converter(verbose=verbose).fit_transform(
        transformer.named_steps['image'].images_,
        pipeline.predict()
    )
    if score:
        pipeline.score()
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
