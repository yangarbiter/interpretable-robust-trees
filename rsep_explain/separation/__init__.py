import joblib
from joblib import Parallel, delayed
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .pruning import adversarial_pruning

def get_nearest_oppo_dist(X, y, tstX=None, tsty=None, norm=2, n_jobs=10):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
        if tstX is not None:
            tstX = tstX.reshape(len(tstX), -1)

    def helper(yi):
        return NearestNeighbors(n_neighbors=1,
                metric='minkowski', p=norm, n_jobs=12).fit(X[y != yi])
    nns = Parallel(n_jobs=n_jobs)(delayed(helper)(yi) for yi in np.unique(y))
    ret = np.zeros(len(X))
    tst_ret = []
    if tstX is not None:
        tst_ret = np.zeros(len(tstX))

    for yi in np.unique(y):
        dist, _ = nns[yi].kneighbors(X[y==yi], n_neighbors=1)
        ret[np.where(y==yi)[0]] = dist[:, 0]
        
        if tstX is not None:
            dist, _ = nns[yi].kneighbors(tstX[tsty==yi], n_neighbors=1)
            tst_ret[np.where(tsty==yi)[0]] = dist[:, 0]

    return ret, tst_ret

def get_rseparation(X, y, tstX=None, tsty=None, norm=2, n_jobs=12, nn_backend="sklearn"):
    ret, tst_ret = get_nearest_oppo_dist(X, y, tstX=tstX, tsty=tsty, norm=norm, n_jobs=n_jobs)
    return ret, tst_ret
