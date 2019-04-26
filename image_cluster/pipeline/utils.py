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
