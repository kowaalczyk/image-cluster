from .utils import optimal_clusters
from .pipeline import Pipeline
from .reader import (
    MetadataReader,
    GreyscaleImageReader,
    MaskImageReader
)
from .vectorizer import (
    FilterImageVectorizer,
    IOUImageVectorizer,
    ShapeVectorizer,
    TextDensityVectorizer
)
from .model_factory import ModelFactory
from .converter import Converter
from .writer import Writer

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import (
    Pipeline as SklearnPipeline,
    FeatureUnion
)


def default_transformers(writer: Writer, verbose: bool):
    preprocessing = SklearnPipeline([
        ('meta', MetadataReader()),
        ('image', GreyscaleImageReader(verbose=verbose)),
        ('vectorizer', FeatureUnion([
            ('iou', IOUImageVectorizer(filter_shape=(3, 3), verbose=verbose)),
            ('shape', ShapeVectorizer(verbose=verbose)),
            ('density', TextDensityVectorizer(verbose=verbose))
        ])),
        ('scaler', MinMaxScaler()),
    ])
    postprocessing = SklearnPipeline([
        ('converter', Converter(
            image_reader=preprocessing.named_steps['image'],
            verbose=verbose
        )),
        ('writer', writer),
    ])
    return preprocessing, postprocessing
