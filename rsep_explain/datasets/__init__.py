import os
from autovar import base

import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

from autovar.base import RegisteringChoiceType, register_var, VariableClass


def load_risk_data(name, base_dir=None):
    from riskslim import load_data_from_csv
    if base_dir is None:
        base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "rsep_explain/datasets/risk_dsets")
    filepath = os.path.join(data_dir, f"{name}_data.csv")
    data = load_data_from_csv(dataset_csv_file=filepath)
    return data

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"risk_adult")
    @staticmethod
    def risk_adult(auto_var, base_dir=None):
        data = load_risk_data("adult", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        return data['X'], data['Y'].reshape(-1)

    @register_var(argument=r"risk_breastcancer")
    @staticmethod
    def risk_breastcancer(auto_var, base_dir=None):
        data = load_risk_data("breastcancer", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        return data['X'], data['Y'].reshape(-1)

    @register_var(argument=r"risk_mushroom")
    @staticmethod
    def risk_mushroom(auto_var, base_dir=None):
        data = load_risk_data("mushroom", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var(argument=r"risk_mammo")
    @staticmethod
    def risk_mammo(auto_var, base_dir=None):
        data = load_risk_data("mammo", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var(argument=r"risk_spambase")
    @staticmethod
    def risk_spambase(auto_var, base_dir=None):
        data = load_risk_data("spambase", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_bank(auto_var, base_dir=None):
        data = load_risk_data("bank", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_heart(auto_var, base_dir=None):
        data = load_risk_data("proc-heart", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_bank_2(auto_var, base_dir=None):
        data = load_risk_data("proc-bank", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_careval(auto_var, base_dir=None):
        data = load_risk_data("careval", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_diabetes(auto_var, base_dir=None):
        data = load_risk_data("proc-diabetes", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_ionosphere(auto_var, base_dir=None):
        data = load_risk_data("proc-ionosphere", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_compasbin(auto_var, base_dir=None):
        data = load_risk_data("compasbin", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var()
    @staticmethod
    def risk_ficobin(auto_var, base_dir=None):
        data = load_risk_data("ficobin", base_dir=base_dir)
        auto_var.inter_var['data'] = data
        X, y = data['X'], data['Y'].reshape(-1).astype(np.int)
        assert len(np.unique(y)) == 2, y
        return X, y

    @register_var(argument=r"risk_twogauss-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def risk_twogauss(auto_var, inter_var, n_samples, noisy_level):
        """
        Returns the two spirals dataset.
        """
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        n_dims = 1000
        noisy_level = float(noisy_level)
        noisy_level = np.eye(n_dims) * noisy_level
        lcenter = -np.ones(n_dims)
        rcenter = np.ones(n_dims)

        def sample():
            l = random_state.multivariate_normal(lcenter, noisy_level, size=n_samples)
            r = random_state.multivariate_normal(rcenter, noisy_level, size=n_samples)
            X = np.hstack((np.ones((2*n_samples, 1)), np.sign(np.vstack((l, r)))))
            y = np.concatenate((-np.ones(n_samples), np.ones(n_samples)))
            return X, y.astype(np.int)

        X, y = sample()
        data = {
            'X': X, 'Y': y, 'variable_names': ['(Intercept)'] + [f't_{i}' for i in range(n_dims)],
            'outcome_name': "out", 'sample_weights': np.ones(n_samples),
        }
        auto_var.inter_var['data'] = data
        return X, y

    @register_var(argument=r"risk_twogauss_cont-(?P<n_samples>\d+)-(?P<noisy_level>\d+\.\d+|\d+)", shown_name="threegauss")
    @staticmethod
    def risk_twogauss_cont(auto_var, inter_var, n_samples, noisy_level):
        """
        Returns the two spirals dataset.
        """
        random_state = np.random.RandomState(auto_var.get_var("random_seed"))
        n_samples = int(n_samples)
        n_dims = 1000
        noisy_level = float(noisy_level)
        noisy_level = np.eye(n_dims) * noisy_level
        lcenter = -np.ones(n_dims)
        rcenter = np.ones(n_dims)

        def sample():
            l = random_state.multivariate_normal(lcenter, noisy_level, size=n_samples)
            r = random_state.multivariate_normal(rcenter, noisy_level, size=n_samples)
            fet = np.vstack((l, r))
            X = np.hstack((np.ones((2*n_samples, 1)), fet))
            y = np.concatenate((-np.ones(n_samples), np.ones(n_samples)))
            return X, y.astype(np.int)

        X, y = sample()
        data = {
            'X': X, 'Y': y, 'variable_names': ['(Intercept)'] + [f't_{i}' for i in range(n_dims)],
            'outcome_name': "out", 'sample_weights': np.ones(n_samples),
        }
        auto_var.inter_var['data'] = data
        return X, y
