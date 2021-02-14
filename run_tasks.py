import os
import logging

from rsep_explain.variables import auto_var, get_file_name
from params import *
from utils import setup_experiments

logging.basicConfig(level=logging.DEBUG)

DEBUG = True if os.environ.get('DEBUG', False) else False

def main(auto_var):
    experiments = [
        CalcSepExperiments(),
        CalcLinSepExperiments(),

        XgboostRobDTInterpretRobExperiments(),
        LinSepBBMRobExperiments3(),
        RiskSlimExperiments3(),
        DTInterpretRobExperiments3(),
    ]
    grid_params = []
    for exp in experiments:
        exp_fn, _, grid_params, run_param = exp()

        if DEBUG:
            run_param['n_jobs'] = 1
            run_param['allow_failure'] = False
        else:
            run_param['n_jobs'] = 1
            run_param['allow_failure'] = False

        auto_var.run_grid_params(exp_fn, grid_params, **run_param)
    #auto_var.run_grid_params(delete_file, grid_params, n_jobs=1,
    #                          with_hook=False, allow_failure=False)

def delete_file(auto_var):
    os.unlink(get_file_name(auto_var) + '.json')

if __name__ == "__main__":
    setup_experiments(auto_var)
    main(auto_var)
