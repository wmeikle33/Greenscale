import numpy as np
from .interfaces import Dataset

class NPZLoader:
    def __init__(self, path: str):
        self.path = path
    def load(self) -> Dataset:
        d = np.load(self.path, allow_pickle=True)
        return Dataset(X=d['X'], y=d['y'])
