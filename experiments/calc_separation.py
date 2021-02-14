import joblib
import numpy as np
from rsep_explain.separation import get_rseparation, adversarial_pruning 

def run_calc_separation(auto_var):
    X, y = auto_var.get_var("dataset")
    preprocess_fn = auto_var.get_var("preprocessor", X=X)
    X = preprocess_fn(X)
    
    r, tst_r = get_rseparation(X, y, norm=2)
    results = {
        'n_samples': X.shape[0], 'n_features': X.shape[1],
        "trn_rsep": r, "tst_rsep": tst_r,
    }

    augX, augy = adversarial_pruning(X, y, 1e-5, 1)
    r, _ = get_rseparation(augX, augy, norm=np.inf)
    results.update({'aug_n_samples': augX.shape[0], 'aug_rsep': r})

    joblib.dump(results, "./temp.pkl")

    return results
