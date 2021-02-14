
import cvxpy as cp
import numpy as np
from sklearn.svm import LinearSVC

def linear_pruning(X, y, c=1e-0, random_state=0):
    # y = {-1, 1}
    y = y.reshape(-1)
    svm = LinearSVC(C=c, random_state=random_state, tol=1e-5, max_iter=1e4)
    svm.fit(X, y)
    #trn_acc = svm.score(X, y)

    decision = svm.decision_function(X)
    #support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
    indices = ((2 * y - 1) * decision < 1)
    #indices = (y * decision < 1)
    #indices = (y * decision < 0)
    augX, augy = X[np.logical_not(indices)], y[np.logical_not(indices)]

    return augX, augy

def get_lin_sep(X, y, w):

    def hard_margin_svm():
        _, m = X.shape
        w = cp.Variable((m,1))
        b = cp.Variable()
        loss = cp.sum(1 / 2 * w.T @ w)
        constraints = [cp.multiply(y, X @ w - b) >= 1]
        prob = cp.Problem(cp.Minimize(loss), constraints)
        return prob
