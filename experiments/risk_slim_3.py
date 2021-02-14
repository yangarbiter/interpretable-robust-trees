import joblib
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

import riskslim
from rsep_explain.attacks.linear import optimal_attack_linear


def run_risk_slim_3(auto_var):
    X, y = auto_var.get_var("dataset")
    y = y.reshape(-1, 1)
    random_seed = auto_var.get_var("random_seed")
    trnX, tstX, trny, tsty = train_test_split(X, y, test_size=0.33, random_state=random_seed)
    preprocess_fn = auto_var.get_var("preprocessor", X=trnX)
    trnX, tstX = preprocess_fn(trnX), preprocess_fn(tstX)
    trnX[:, 0] = 1
    tstX[:, 0] = 1
    data = auto_var.inter_var['data']

    results = {"linsep_results": [], "n_samples": len(X)}

    max_l0_candidates = [5, 10, 15, 20, 25, 30]
    kf = KFold(n_splits=5)
    cv_results = {}
    for max_l0 in max_l0_candidates:
        if trnX.shape[1] <= max_l0:
            cv_results[max_l0] = {'split_results': [], 'avg_tst_acc': 0.}
            continue
        cv_results[max_l0] = {'split_results': []}
        for tr_idx, ts_idx in kf.split(trnX):
            tX, ty = trnX[tr_idx], trny[tr_idx]
            tsX, tsy = trnX[ts_idx], trny[ts_idx]
            data['X'], data['Y'], data['sample_weights'] = tX, ty, np.ones(len(tX))
            w = train_risk_slim(data, max_L0_value=max_l0)
            t_preds = 1. / (1. + np.exp(-(np.dot(tX, w))))
            ts_preds = 1. / (1. + np.exp(-(np.dot(tsX, w))))

            cv_results[max_l0]['split_results'].append({
                'trn_acc': ((t_preds > 0.5) == ty).mean(),
                'tst_acc': ((ts_preds > 0.5) == tsy).mean(),
            })
        cv_results[max_l0]['avg_tst_acc'] = np.mean([
            d['tst_acc'] for d in cv_results[max_l0]['split_results']])
    best_max_l0 = max_l0_candidates[np.argmax([
        cv_results[l0]['avg_tst_acc'] for l0 in max_l0_candidates])]

    w = train_risk_slim(data, max_L0_value=best_max_l0)
    trn_preds = 1. / (1. + np.exp(-(np.dot(trnX, w))))
    tst_preds = 1. / (1. + np.exp(-(np.dot(tstX, w))))

    ttrny, ttsty = (trny + 1) // 2, (tsty + 1) // 2
    ttrny, ttsty = ttrny.reshape(-1), ttsty.reshape(-1)

    try:
        adv_tstX = optimal_attack_linear(tstX, tsty.reshape(-1), w)
        adv_tst_dist = np.linalg.norm(adv_tstX - tstX, ord=np.inf, axis=1)

        inds = np.where((tst_preds > 0.5) == ttsty)[0]
        subsample = np.random.RandomState(random_seed).choice(
            inds, size=min(len(inds), 100), replace=False)
        adv_tstX = optimal_attack_linear(tstX[subsample], tsty.reshape(-1)[subsample], w)
        er_dist = np.linalg.norm(adv_tstX - tstX[subsample], ord=np.inf, axis=1)
    except:
        raise ValueError()
        adv_tst_dist = None

    results["linsep_results"].append({
        'w': w,
        'trn acc': ((trn_preds > 0.5) == ttrny).mean(),
        'tst acc': ((tst_preds > 0.5) == ttsty).mean(),
        'trn auc': roc_auc_score(ttrny, trn_preds),
        'tst auc': roc_auc_score(ttsty, tst_preds),
        'adv tst dist': adv_tst_dist,
        'er dist': er_dist,
        'cv_results': cv_results,
        'best_max_l0': best_max_l0,
    })

    joblib.dump(results, "./temp.pkl")

    return results

def train_risk_slim(data, max_coefficient=5, max_L0_value=5, max_offset=50, c0_value=1e-6):
    # problem parameters
    #max_coefficient = 5     # value of largest/smallest coefficient
    #max_L0_value = 5        # maximum model size (set as float(inf))
    #max_offset = 50         # maximum value of offset parameter (optional)
    #c0_value = 1e-6         # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms

    # create coefficient set and set the value of the offset parameter
    coef_set = riskslim.CoefficientSet(variable_names=data['variable_names'], lb=-max_coefficient, ub=max_coefficient, sign=0)
    coef_set.update_intercept_bounds(X=data['X'], y=data['Y'], max_offset=max_offset)

    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }

    # major settings (see riskslim_ex_02_complete for full set of options)
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        #
        # LCPA Settings
        'max_runtime': 30.0,                               # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': True,                     # print CPLEX progress on screen
        'loss_computation': 'fast',                         # how to compute the loss function ('normal','fast','lookup')
        #
        # LCPA Improvements
        'round_flag': True,                                # round continuous solutions with SeqRd
        'polish_flag': True,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': True,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        #
        # Initialization
        'initialization_flag': True,                       # use initialization procedure
        'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,
        #
        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }

    # train model using lattice_cpa
    model_info, _, _ = riskslim.run_lattice_cpa(data, constraints, settings)

    #print model contains model
    #riskslim.print_model(model_info['solution'], data)

    w = model_info['solution']
    return w
