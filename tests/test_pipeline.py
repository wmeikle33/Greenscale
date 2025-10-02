import numpy as np
from imgrec.adapters_npz import NPZLoader
from imgrec.preprocess import Normalize01
from imgrec.models import CentroidClassifier
from imgrec.pipeline import run_train_eval

def _blob(n, d, mean, seed):
    r = np.random.default_rng(seed)
    return r.normal(loc=mean, scale=0.5, size=(n, d))

def test_centroid_pipeline(tmp_path):
    X0 = _blob(40, 16, -1.0, 1)
    X1 = _blob(40, 16, +1.0, 2)
    Xtr = np.vstack([X0, X1]).astype('float32').reshape(-1, 4, 4)
    ytr = np.array(['a']*len(X0) + ['b']*len(X1))

    X0t = _blob(20, 16, -1.0, 3)
    X1t = _blob(20, 16, +1.0, 4)
    Xte = np.vstack([X0t, X1t]).astype('float32').reshape(-1, 4, 4)
    yte = np.array(['a']*len(X0t) + ['b']*len(X1t))

    tr = tmp_path / 'tr.npz'; te = tmp_path / 'te.npz'
    np.savez_compressed(tr, X=Xtr, y=ytr)
    np.savez_compressed(te, X=Xte, y=yte)

    metrics = run_train_eval(
        NPZLoader(str(tr)), NPZLoader(str(te)),
        Normalize01(flatten=True), CentroidClassifier()
    )
    assert metrics['accuracy'] >= 0.9
