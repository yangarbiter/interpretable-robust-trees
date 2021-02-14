import joblib
import numpy as np
from sklearn.svm import LinearSVC

from rsep_explain.attacks.linear import optimal_attack_linear

def solve_l1_svm(X, y, C):
    import cvxpy as cp
    n, m = X.shape
    w = cp.Variable((m,1))
    loss = cp.sum(cp.pos(1 - cp.multiply(y, X @ w)))
    reg = cp.norm(w, 1)
    prob = cp.Problem(cp.Minimize(loss/n + C*reg))
    prob.solve(solver="GUROBI")
    return w.value


def run_calc_lin_separation(auto_var):
    X, y = auto_var.get_var("dataset")
    preprocess_fn = auto_var.get_var("preprocessor", X=X)
    X = preprocess_fn(X)

    results = {"svm_results": [], "l1svm_results": [], "n_samples": len(X)}
    for c in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0, 1e2, 1e4, 1e6, 1e8, 1e10]:
        temp_results = {}
        # the bias term is already in the feature
        svm = LinearSVC(C=c, random_state=0, tol=1e-5, max_iter=1e4)
        svm.fit(X[:, 1:], y)

        decision_function = svm.decision_function(X[:, 1:])
        support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
        #support_vectors = X[support_vector_indices]

        w = np.concatenate((np.array(svm.intercept_), svm.coef_.reshape(-1)))
        try:
            advX = optimal_attack_linear(X, y.reshape(-1), w)
            adv_dists = np.linalg.norm(advX - X, ord=np.inf, axis=1)
        except:
            print(f"Failed to attack for {c}")
            adv_dists = np.array([-1])

        temp_results['adv dists'] = adv_dists
        temp_results['w'] = w
        temp_results['c'] = c
        temp_results['num svs'] = len(support_vector_indices)
        support_vector_indices = np.where((2 * y - 1) * decision_function < 1)[0]
        temp_results['num to rm'] = len(support_vector_indices)
        temp_results['acc'] = svm.score(X[:, 1:], y)
        results["svm_results"].append(temp_results)

        w = solve_l1_svm(X, y.reshape(-1, 1), c)
        pred = np.sign(X @ w).reshape(-1)

        advX = optimal_attack_linear(X, y.reshape(-1), w.reshape(-1))
        adv_dists = np.linalg.norm(advX - X, ord=np.inf, axis=1)

        results["l1svm_results"].append({
            'c': c,
            'w': w,
            'wl1': np.linalg.norm(w, ord=1),
            'acc': (pred == y).mean(),
            'adv dists': adv_dists,
        })
    
    joblib.dump(results, "./temp.pkl")

    return results
