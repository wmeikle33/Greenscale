from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray

class Loader(Protocol):
    def load(self) -> Dataset: ...

class Preprocessor(Protocol):
    def fit(self, ds: Dataset): ...
    def transform(self, ds: Dataset) -> Dataset: ...

class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray): ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
