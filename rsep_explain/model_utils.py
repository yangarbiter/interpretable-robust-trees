from sklearn.base import BaseEstimator
import numpy as np

class DummyClfWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model
        self.dummy_cls = None

    def fit(self, X, y):
        self.dummy_cls = None
        if len(np.unique(y)) == 1:
            self.dummy_cls = np.unique(y)[0]
            return
        else:
            return self.model.fit(X, y)

    def predict(self, X):
        if self.dummy_cls is not None:
            return np.ones(len(X)) * self.dummy_cls
        return self.model.predict(X)

    def score(self, X, y):
        self.model.score(X, y)

    #def get_params(self, deep=True):
    #    return self.model.get_params(deep=deep)

    #def set_params(self, **params):
    #    return self.model.set_params(**params)