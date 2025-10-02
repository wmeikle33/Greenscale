import numpy as np

def _factorize(y):
    classes, inv = np.unique(y, return_inverse=True)
    return inv, classes

class CentroidClassifier:
    def __init__(self):
        self.classes_ = None
        self.centroids_ = None
    def fit(self, X, y):
        y_enc, classes = _factorize(y)
        self.classes_ = classes
        cents = []
        for c in range(len(classes)):
            cents.append(X[y_enc == c].mean(axis=0))
        self.centroids_ = np.stack(cents, axis=0)
        return self
    def predict(self, X):
        C = self.centroids_
        dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        idx = dists.argmin(axis=1)
        return self.classes_[idx]
