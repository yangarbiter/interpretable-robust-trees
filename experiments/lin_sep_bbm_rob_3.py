import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

from rsep_explain.models.boosting.boosting_by_majority import BoostingByMajority
from rsep_explain.attacks.bbm import optimal_attack_bbm

def run_lin_sep_bbm_rob_3(auto_var):
    X, y = auto_var.get_var("dataset")
    random_seed = auto_var.get_var("random_seed")
    trnX, tstX, trny, tsty = train_test_split(X, y, test_size=0.33, random_state=random_seed)
    preprocess_fn = auto_var.get_var("preprocessor", X=trnX)
    trnX, tstX = preprocess_fn(trnX), preprocess_fn(tstX)

    clf = BoostingByMajority(n_estimators=30, gamma=10.0, direction=1).fit(trnX, trny)
    print((clf.predict(tstX) == tsty).mean())

    results = {"n_samples": len(trnX)}
    params = {
        'n_estimators': [5, 10, 15, 20, 25, 30],
        'gamma': [0.01],
        'direction': [1],
    }
    clf = GridSearchCV(
        BoostingByMajority(n_estimators=10, gamma=0.1, direction=1),
        params, cv=5, n_jobs=4,
    )
    eps = auto_var.get_var("rsep")
    clf.fit(trnX - eps * trny.reshape(len(trny), 1), trny)
    results["cv_results"] = clf.cv_results_
    results['best_params'] = clf.best_params_
    clf = clf.best_estimator_
    ttrny, ttsty = (trny + 1) // 2, (tsty + 1) // 2

    dif_depth = [None]
    for i in range(1, len(clf.estimators)):
        trn_preds = clf.predict_proba(trnX, n_estimators=i)
        tst_preds = clf.predict_proba(tstX, n_estimators=i)

        dif_depth.append({
            "n_estimators": i,
            'trn acc': ((trn_preds > 0.5) == ttrny).mean(),
            'tst acc': ((tst_preds > 0.5) == ttsty).mean(),
            'trn auc': roc_auc_score(ttrny, trn_preds),
            'tst auc': roc_auc_score(ttsty, tst_preds),
            'adv tst dist': None,
        })

    trn_preds = clf.predict_proba(trnX)
    tst_preds = clf.predict_proba(tstX)

    adv_tstX = optimal_attack_bbm(tstX, tsty.reshape(-1), clf)
    adv_tst_dist = np.linalg.norm(adv_tstX - tstX, ord=np.inf, axis=1)

    inds = np.where((tst_preds > 0.5) == ttsty)[0]
    subsample = np.random.RandomState(random_seed).choice(
        inds, size=min(len(inds), 100), replace=False)
    adv_tstX = optimal_attack_bbm(tstX[subsample], tsty.reshape(-1)[subsample], clf)
    ER_dist = np.linalg.norm(adv_tstX - tstX[subsample], ord=np.inf, axis=1)

    results["bbm_results"] = {
        'depth': len(clf.estimators),
        'trn acc': ((trn_preds > 0.5) == ttrny).mean(),
        'tst acc': ((tst_preds > 0.5) == ttsty).mean(),
        'trn auc': roc_auc_score(ttrny, trn_preds),
        'tst auc': roc_auc_score(ttsty, tst_preds),
        'dif_depth': dif_depth,
        'direction': clf.direction,
        'learned_model': [m.model for m in clf.estimators],
        'adv tst dist': adv_tst_dist,
        'er dist': ER_dist,
    }

    joblib.dump(results, "./temp.pkl")
    
    return results
