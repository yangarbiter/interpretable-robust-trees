import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from rsep_explain.attacks.dt import DTOptAttack


def run_dt_interpret_rob_3(auto_var):
    X, y = auto_var.get_var("dataset")
    random_seed = auto_var.get_var("random_seed")
    trnX, tstX, trny, tsty = train_test_split(X, y, test_size=0.33, random_state=random_seed)
    preprocess_fn = auto_var.get_var("preprocessor", X=X)
    trnX = preprocess_fn(trnX)
    tstX = preprocess_fn(tstX)

    params = {
        'criterion': ['entropy'],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'random_state': [0],
    }
    model = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=0),
        params, cv=5, n_jobs=4,
    )
    model.fit(trnX, trny)

    attack = DTOptAttack(clf=model.best_estimator_, norm=np.inf)
    trn_preds = model.predict(trnX)
    tst_preds = model.predict(tstX)

    subsample = np.random.RandomState(random_seed).choice(
        np.arange(len(tstX)), size=min(len(tstX), 100), replace=False)
    adv_tst_dist = np.linalg.norm(
        attack.perturb(tstX[subsample], tsty[subsample]), ord=np.inf, axis=1)

    inds = np.where(tst_preds == tsty)[0]
    subsample = np.random.RandomState(random_seed).choice(
        inds, size=min(len(inds), 100), replace=False)
    ER_dist = np.linalg.norm(attack.perturb(tstX[subsample], tsty[subsample]), ord=np.inf, axis=1)

    results = {
        'cv_results': model.cv_results_,
        'trn acc': (trn_preds == trny).mean(),
        'tst acc': (tst_preds == tsty).mean(),
        'adv tst dist': adv_tst_dist.mean(),
        'er dist': ER_dist.mean(),
        'depth': model.best_estimator_.get_depth(),
        'leaves': model.best_estimator_.get_n_leaves(),
        'best_params': model.best_params_,
        'best_clf': model.best_estimator_,
    }

    joblib.dump(results, "./temp.pkl")

    return results
