import math

from tqdm import tqdm


class VerboseMixin(object):
    def _progress(self, iterator):
        if self.verbose:
            return tqdm(iterator, desc=self.__class__.__name__)
        else:
            return iterator

    def _log(self, message):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")


class NoFitMixin(object):
    def fit(self, X, y=None, *args, **kwargs):
        return self


def optimal_clusters(n_samples: int) -> int:
    """
    Returns optimal number of clusters for the given number of samples
    (uneven distribution of character images results in fewer image classes
    being present in the smaller sample, preventing fixed number of clusters
    to yield desired results)
    The formula was developed via empirical trial-and-error tesing.
    """
    return 60 + int(12 * math.log10(n_samples // 100))
