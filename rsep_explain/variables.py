import logging
from functools import partial

import numpy as np
from autovar import AutoVar
from autovar.base import RegisteringChoiceType, register_var, VariableClass
from autovar.hooks import check_result_file_exist, save_result_to_file
from autovar.hooks import create_placeholder_file, remove_placeholder_if_error
from autovar.hooks import default_get_file_name as get_file_name

from .datasets import DatasetVarClass
from .preprocessors import PreprocessorVarClass

auto_var = AutoVar(
    logging_level=logging.INFO,
    before_experiment_hooks=[
        partial(check_result_file_exist, get_name_fn=get_file_name),
        partial(create_placeholder_file, get_name_fn=get_file_name),
    ],
    after_experiment_hooks=[
         partial(save_result_to_file, get_name_fn=get_file_name),
         partial(remove_placeholder_if_error, get_name_fn=get_file_name),
    ],
    settings={
        'file_format': 'pickle',
        'server_url': '',
        'result_file_dir': './results/'
    }
)

auto_var.add_variable_class(DatasetVarClass())
auto_var.add_variable_class(PreprocessorVarClass())
auto_var.add_variable('random_seed', int)
auto_var.add_variable('rsep', float, default=float(-1.))
