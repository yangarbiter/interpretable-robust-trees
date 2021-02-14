import json

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

from rsep_explain.attacks.xgb import XGBoost_optimal_attack


def get_xgb_depth_leaves_nodes(model):
    bb = model.get_booster()
    tree = json.loads(bb.get_dump(dump_format='json')[0])

    def _dfs(node, d, data):
        data['nodes'] += 1
        if 'leaf' in node:
            data['leaves'] += 1
            data['depth'] = max(data['depth'], d)
        else:
            for child in node['children']:
                _dfs(child, d+1, data)
    data = {'depth': 0, 'leaves': 0, 'nodes': 0}
    _dfs(tree, 1, data)
    return data['depth'], data['leaves'], data['nodes']

def run_xgboostrobdt_interpret_rob(auto_var):
    random_seed = auto_var.get_var("random_seed")

    X, y = auto_var.get_var("dataset")
    trnX, tstX, trny, tsty = train_test_split(X, y, test_size=0.33, random_state=random_seed)
    preprocess_fn = auto_var.get_var("preprocessor", X=X)
    trnX = preprocess_fn(trnX)
    tstX = preprocess_fn(tstX)

    params = {
        'max_depth': [5, 10, 15, 20, 25, 30],
        'random_state': [0],
    }
    model = GridSearchCV(
        XGBClassifier(n_estimators=1, booster="gbtree",
                      tree_method="robust_exact", objective="binary:logistic",
                      robust_eps=auto_var.get_var("rsep"), random_state=0),
        params, cv=5, n_jobs=-1,
    )
    model.fit(trnX, trny)

    trn_preds = model.predict(trnX)
    tst_preds = model.predict(tstX)

    subsample = np.random.RandomState(random_seed).choice(
        np.arange(len(tstX)), size=min(len(tstX), 100), replace=False)
    advX = XGBoost_optimal_attack(model.best_estimator_, tstX[subsample], tsty[subsample])
    adv_tst_dist = np.linalg.norm(advX - tstX[subsample], ord=np.inf, axis=1)

    inds = np.where(tst_preds == tsty)[0]
    subsample = np.random.RandomState(random_seed).choice(
        inds, size=min(len(inds), 100), replace=False)
    advX = XGBoost_optimal_attack(model.best_estimator_, tstX[subsample], tsty[subsample])
    ER_dist = np.linalg.norm(advX - tstX[subsample], ord=np.inf, axis=1)

    depth, leaves, nodes = get_xgb_depth_leaves_nodes(model.best_estimator_)

    results = {
        'cv_results': model.cv_results_,
        'trn acc': (trn_preds == trny).mean(),
        'tst acc': (tst_preds == tsty).mean(),
        #'adv trn dist': adv_trn_dist.mean(),
        'adv tst dist': adv_tst_dist,
        'er dist': ER_dist,
        'depth': depth,
        'leaves': leaves,
        'nodes': nodes,
        'best_params': model.best_params_,
        'best_clf': model.best_estimator_,
    }

    joblib.dump(results, "./temp.pkl")

    return results
