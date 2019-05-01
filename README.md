# Image cluster
An application for clustering black-and-white png images.

## Application structure

Application is written in Python 3.7 and can be installed via provided `Makefile`.
Running `make` creates a virtual environemnt where the package is installed.

After installation, there are 2 commands available:

1. Metadata generation helper, that generates list of files from a given folder.
This was not required, but was necessary to design tests and experiments.
(a format specified in the task description):
```bash
python -m image_cluster meta
```

2. Main application, that takes generated metadata (list of filenames), and
fits a clustering pipeline (described below) on them:
```bash
python -m image_cluster cluster
```

Both commands support `--help` option for argument and option reference.

## Clustering method
There are 6 main parts of the clustering pipeline that I experimented with,
to make sure the results are good looking and consistent:
- reading images: grayscale or binary mask (black or white)
- vectorization: IOU or direct match
- additional image features: shape, average color
- feature scaling: min-max, standard (0 mean, 1 standard deviation), no scaling
- feature dimentionality reduction: PCA or no reduction
- model: K-Means, Agglomerative Hierarchical Clustering, DBSCAN

The best and most consistent pipeline consisted of:
1. reading grayscale images
2. encoding images by computing IOU metric
with all 512 possible 3x3 binary "filter" masks
3. Adding shape and average color features
4. Using min-max scaling (between 0 and 1)
and no dimentionality reduction methods
5. Clustering using the Agglomerative Hierarchical Clustering

As for the individual components' hyperparameters,
I have experimented with:
- shape of encoding filters (in the vectorization step)
- model hyperparameters (for all models)
- number of clusters

To aid the manual parameter search process, I have also implemented the
calculation of some most important metrics as the part of main application.

I also did not assume that distribution of characters is uniform,
and by experimenting on subsets of the given image dataset
(randomly sampled 100, 200, 400, 800, 1600 images),
I arrived at the conclusion that the optimal number of clusters is
smaller for smaller number of samples. This behaviour is implemented
in the `pipeline.utils.optimal_clusters` function, but can be manually
overriden by specifying `--n-clusters` command line option.

## Expected running time
Depends on the size of input data, for the given sample of 7618 images
it is expected to be around 3 minutes.

The longest part of the process is vectorization (which takes nearly 2 minutes).

By default, application has `--verbose` flag enabled, which displays progressbars
to track the expected time of computation for most steps.
