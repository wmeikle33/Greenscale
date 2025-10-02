import numpy as np
from .interfaces import Dataset

class Normalize01:
    def __init__(self, flatten: bool = True):
        self.flatten = flatten
        self.maxv = 1.0
    def fit(self, ds: Dataset):
        x = ds.X.astype('float32')
        self.maxv = float(x.max()) if x.size else 1.0
        return self
    def transform(self, ds: Dataset) -> Dataset:
        x = ds.X.astype('float32')
        mv = self.maxv if self.maxv and self.maxv > 1 else 1.0
        x = x / mv
        if self.flatten and x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return Dataset(X=x, y=ds.y)
