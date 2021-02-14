from functools import partial

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures

from autovar.base import RegisteringChoiceType, register_var, VariableClass

class PreprocessorVarClass(VariableClass, metaclass=RegisteringChoiceType):
    var_name = 'preprocessor'
    
    @register_var()
    @staticmethod
    def none(auto_var, X):
        return lambda x: x

    @register_var()
    @staticmethod
    def rminmax(auto_var, X):
        scaler = MinMaxScaler().fit(X[:, 1:])
        def transformer(x):
            return np.hstack((x[:, :1], scaler.transform(x[:, 1:])))
        return transformer

    @register_var()
    @staticmethod
    def minmax(auto_var, X):
        scaler = MinMaxScaler().fit(X)
        return partial(scaler.transform)

    @register_var()
    @staticmethod
    def standard(auto_var, X):
        scaler = StandardScaler().fit(X)
        return partial(scaler.transform)

    @register_var()
    @staticmethod
    def poly2(auto_var, X):
        scaler = PolynomialFeatures(degree=2, include_bias=True).fit(X)
        return partial(scaler.transform)

    @register_var()
    @staticmethod
    def poly2minmax(auto_var, X):
        fets = PolynomialFeatures(degree=2, include_bias=True).fit(X)
        fetX = fets.transform(X)
        scaler = MinMaxScaler().fit(fetX)
        ret = lambda x: scaler.transform(fets.transform(X))
        return ret