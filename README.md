# Connecting Interpretability and Robustness in Decision Trees through Separation

This repository contains the code of the experiments in the paper

[Connecting Interpretability and Robustness in Decision Trees through Separation](https://arxiv.org/abs/2102.07048)

Authors: [Michal Moshkovitz](https://sites.google.com/view/michal-moshkovitz), [Yao-Yuan Yang](https://github.com/yangarbiter/), [Kamalika Chaudhuri](http://cseweb.ucsd.edu/~kamalika/)

## Abstract

Recent research has recognized interpretability and robustness as essential properties of trustworthy classification. Curiously, a connection between robustness and interpretability was empirically observed, but the theoretical reasoning behind it remained elusive. In this paper, we rigorously investigate this connection. Specifically, we focus on interpretation using decision trees and robustness to $l_\infty$-perturbation. Previous works defined the notion of $r$-separation as a sufficient condition for robustness. We prove upper and lower bounds on the tree size in case the data is $r$-separated. We then show that a tighter bound on the size is possible when the data is linearly separated. We provide the first algorithm with provable guarantees both on robustness, interpretability, and accuracy in the context of decision trees. Experiments confirm that our algorithm yields classifiers that are both interpretable and robust and have high accuracy.

## Installation

```
pip install -r requirements.txt
```

#### Install LP, QP Solvers

- Install gurobi: https://www.cvxpy.org/install/index.html#install-with-gurobi-support
- Install GLPK: https://www.gnu.org/software/glpk/
- Install CVXOPT with GLPK support:
```
CVXOPT_BUILD_GLPK=1
CVXOPT_GLPK_LIB_DIR=/path/to/glpk-X.X/lib
CVXOPT_GLPK_INC_DIR=/path/to/glpk-X.X/include
pip install --upgrade cvxopt
```

### submodules

```
git submodule init
git submodule update
```

#### Install LCPA

```
cd risk-slim
pip install -r requirements.txt
pip install .
```

For more LCPA installation instructions, please visit https://github.com/ustunb/risk-slim


#### Install robust decision tree (RobDT)

```
cd RobustTrees
git submodule update --init --recursive
./build.sh
cd python-package
python setup.py install
```

For more RobDT installation instructions, please visit https://github.com/chenhongge/RobustTrees

## Scripts

### Dataset process scripts

- [scripts/proc_bank.py](scripts/proc_bank.py)
- [scripts/proc_data.py](scripts/proc_data.py)
- [scripts/proc_heart.py](scripts/proc_heart.py)
- [scripts/proc_svmlight.py](scripts/proc_svmlight.py)

### Experiment scripts

- [params.py](params.py): listed all parameters run
- [rsep_explain/datasets/__init__.py](rsep_explain/datasets/__init__.py): load datasets
- [experiments/lin_sep_bbm_rob_3.py](experiments/lin_sep_bbm_rob_3.py): run experiment for BBM-RS
- [experiments/dt_interpret_rob_3.py](experiments/dt_interpret_rob_3.py): run experiment for DT (Breiman et al., 1984)
- [experiments/xgboostrobdt_interpret_rob.py](experiments/xgboostrobdt_interpret_rob.py): run experiment for RobDT (Chen et al., 2019)
- [experiments/risk_slim_3.py](experiments/risk_slim_3.py): run experiment for LCAP (Ustun & Rudin, 2019)
- [experiments/calc_lin_separation.py](experiments/calc_lin_separation.py): estimating the linear separateness of each dataset
- [experiments/calc_separation.py](experiments/calc_separation.py): estimating the $r$- separateness of each dataset

### Figure/Table generation scripts
- [notebooks/case_study.ipynb](notebooks/case_study.ipynb): generate Table 1
- [notebooks/separation.ipynb](notebooks/separation.ipynb): generate Table 2
- [notebooks/risk_score_3.ipynb](notebooks/risk_score_3.ipynb): generate Table 3 and 4
- [notebooks/tradeoff.ipynb](notebooks/tradeoff.ipynb): generate images in Figure 1 and 4
- [notebooks/plot_bbm.ipynb](notebooks/plot_bbm.ipynb): generate images in Figure 5

### Parameters

```
usage: main.py [-h] [--no-hooks] --experiment
               {lin_sep_bbm_rob_3,risk_slim_3,dt_interpret_rob_3,xgboostrobdt_interpret_rob,calc_lin_separation,calc_separation}
               --dataset DATASET --preprocessor PREPROCESSOR --random_seed
               RANDOM_SEED --rsep RSEP
```

Datasets: {risk_ionosphere, risk_diabetes, risk_breastcancer, risk_adult, risk_mushroom, risk_mammo,
risk_spambase, risk_bank, risk_careval, risk_compasbin, risk_ficobin, risk_bank_2, risk_heart}


#### Algorithm implementations

- [Boosting by majority (BBM)](rsep_explain/models/boosting/boosting_by_majority.py)


## Examples

The result of each example is outputed as a joblib pickle file named temp.pkl.

Run BBM-RS with $\tau = 0.05$ on the bank dataset.
```
python main.py --no-hooks --experiment lin_sep_bbm_rob_3 \
  --dataset risk_bank --preprocessor rminmax \
  --rsep 0.05 \
  --random_seed 0
```

Run RobDT with robust radius $ = 0.1$ on the mammo dataset.
```
python main.py --no-hooks --experiment xgboostrobdt_interpret_rob \
  --dataset risk_mammo --preprocessor rminmax \
  --rsep 0.1 \
  --random_seed 0
```

Run LCAP on the mammo dataset.
```
python main.py --no-hooks --experiment risk_slim_3 \
  --dataset risk_mammo --preprocessor rminmax \
  --random_seed 0
```

Run DT on the heart dataset.
```
python main.py --no-hooks --experiment dt_interpret_rob_3 \
  --dataset risk_heart --preprocessor rminmax \
  --random_seed 0
```

Calculate the r-separateness of the heart dataset.
```
python main.py --no-hooks --experiment calc_separation \
  --dataset risk_heart --preprocessor rminmax \
  --random_seed 0
```

Calculate the linear separateness of the heart dataset.
```
python main.py --no-hooks --experiment calc_lin_separation \
  --dataset risk_heart --preprocessor rminmax \
  --random_seed 0
```
