from utils import RSepExperiments

__all__ = [
    'CalcSepExperiments', 'CalcLinSepExperiments',
    'DTInterpretRobExperiments3', 'XgboostRobDTInterpretRobExperiments', 
    'RiskSlimExperiments3', 'LinSepBBMRobExperiments3',
]


random_seed = list(range(10))

DATASETS = [
    "risk_ionosphere", "risk_diabetes",
    "risk_breastcancer", "risk_adult", "risk_mushroom", "risk_mammo",
    "risk_spambase", "risk_bank", "risk_careval", "risk_compasbin",
    "risk_ficobin", "risk_bank_2", "risk_heart",
]
PREPROCESSOR = ['rminmax']


class XgboostRobDTInterpretRobExperiments(RSepExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = ""
        cls.experiment_fn = 'xgboostrobdt_interpret_rob'
        grid_params = []
        grid_params.append({
            'random_seed': random_seed, "dataset": DATASETS, "preprocessor": PREPROCESSOR,
            'rsep': [0., 0.05, 0.1, 0.15, 0.2, 0.25],
        })
        cls.grid_params = grid_params
        return RSepExperiments.__new__(cls, *args, **kwargs)

class LinSepBBMRobExperiments3(XgboostRobDTInterpretRobExperiments):
    def __new__(cls, *args, **kwargs):
        return XgboostRobDTInterpretRobExperiments.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'lin_sep_bbm_rob_3'

class DTInterpretRobExperiments3(RSepExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = ""
        cls.experiment_fn = 'dt_interpret_rob_3'
        grid_params = []
        grid_params.append({
            'random_seed': random_seed,
            "dataset": DATASETS,
            "preprocessor":  PREPROCESSOR,
        })
        cls.grid_params = grid_params
        return RSepExperiments.__new__(cls, *args, **kwargs)

class RiskSlimExperiments3(RSepExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = ""
        cls.experiment_fn = 'risk_slim_3'
        grid_params = []
        grid_params.append({
            'random_seed': random_seed,
            "dataset": DATASETS,
            "preprocessor": PREPROCESSOR,
        })
        cls.grid_params = grid_params
        return RSepExperiments.__new__(cls, *args, **kwargs)

class CalcSepExperiments(RSepExperiments):
    def __new__(cls, *args, **kwargs):
        cls.experiment_fn = 'calc_separation'
        cls.name = "sample experiment"
        grid_params = []
        grid_params.append({
            'random_seed': [0],
            "dataset": DATASETS,
            "preprocessor": PREPROCESSOR,
        })
        cls.grid_params = grid_params
        return RSepExperiments.__new__(cls, *args, **kwargs)

class CalcLinSepExperiments(RSepExperiments):
    def __new__(cls, *args, **kwargs):
        cls.experiment_fn = 'calc_lin_separation'
        cls.name = "sample experiment"
        grid_params = []
        grid_params.append({
            'random_seed': [0],
            "dataset": DATASETS,
            "preprocessor": PREPROCESSOR,
        })
        cls.grid_params = grid_params
        return RSepExperiments.__new__(cls, *args, **kwargs)