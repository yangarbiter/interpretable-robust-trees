from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
from scipy.special import comb
import numpy as np
from numba import njit


def go_through_itervals(sortedX, y, weights, intervals, i, res):
    pred = (sortedX >= intervals[0]).astype(np.int32) * 2 - 1
    yf = y * pred * weights
    cur_acc = yf.sum()
    if cur_acc > res[1]:
        res = ((i, intervals[0]), cur_acc)
    now = 0
    while pred[now] > 0 and now < len(pred):
        now += 1

    for j in intervals[1:]:
        while sortedX[now] <= j and now < len(pred):
            pred[now] = -1
            now += 1
        cur_acc += 2 * weights[j+1]
        if cur_acc > res[1]:
            res = ((i, j), cur_acc)

@njit
def stump_fit(X, y, weights, max_search_per_feature):
    _, m = X.shape
    res = (None, -1)
    for i in range(m):
        temp = np.unique(np.sort(X[:, i]))
        #temp = np.concatenate((temp, np.array([np.inf])))
        temp = np.array([temp[0]] + [(temp[i+1] + temp[i]) / 2 for i in range(len(temp)-1)])
        if max_search_per_feature != -1:
            if len(temp) > max_search_per_feature:
                temp = temp[::len(temp)//max_search_per_feature]
        for j in temp:
            pred = (X[:, i] >= j).astype(np.int32) * 2 - 1
            w_acc = ((y == pred) * weights).sum()
            if w_acc > res[1]:
                res = ((i, j), w_acc)
    return res

class DecisionStump(BaseEstimator, ClassifierMixin):

    def __init__(self, max_search_per_feature=-1):
        self.model = None
        self.max_search_per_feature = max_search_per_feature

    def fit(self, X, y, weights=None):
        # y ==> {-1, 1}
        X = X.astype(np.float64)
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        #y = y * 2 - 1
        n, m = X.shape
        if weights is None:
            weights = np.ones(len(X))

        #res = (None, -1)
        #for i in range(m):
        #    temp = np.unique(np.sort(X[:, i]))
        #    temp = np.concatenate((temp, np.array([np.inf])))
        #    temp = np.array([temp[0]] + [(temp[i+1] + temp[i]) / 2 for i in range(len(temp)-1)])
        #    if self.max_search_per_feature != -1:
        #        if len(temp) > self.max_search_per_feature:
        #            temp = temp[::len(temp)//self.max_search_per_feature]
        #    for j in temp:
        #        pred = (X[:, i] >= j).astype(np.int) * 2 - 1
        #        w_acc = ((y == pred) * weights).sum()
        #        if w_acc > res[1]:
        #            res = ((i, j), w_acc)
        res = stump_fit(X, y, weights, self.max_search_per_feature)
        self.model = res[0]
        ##### interface for attack
        self.b = res[0][1]
        self.coord = res[0][0]
        self.w_l = -1
        self.w_r = 2.
        #####
        return self

    def predict(self, X):
        #check_is_fitted(self)
        X = check_array(X)
        return (X[:, self.model[0]] >= self.model[1]).astype(np.int) * 2 - 1

@njit
def rob_stump_fit(X, y, weights, max_search_per_feature, epsilon):
    _, m = X.shape
    res = (None, -1)
    for i in range(m):
        temp = np.unique(np.sort(X[:, i]))
        #temp = np.concatenate((temp, np.array([np.inf])))
        temp = np.array([temp[0]] + [(temp[i+1] + temp[i]) / 2 for i in range(len(temp)-1)])
        if max_search_per_feature != -1:
            if len(temp) > max_search_per_feature:
                temp = temp[::len(temp)//max_search_per_feature]
        for j in temp:
            pred = (X[:, i] >= j).astype(np.int32) * 2 - 1
            w_acc = ((y == pred) * weights).sum()
            if w_acc > res[1]:
                res = ((i, j), w_acc)
    return res

class RobDecisionStump(BaseEstimator, ClassifierMixin):

    def __init__(self, epsilon, max_search_per_feature=-1):
        self.model = None
        self.max_search_per_feature = max_search_per_feature

    def fit(self, X, y, weights=None):
        # y ==> {-1, 1}
        X = X.astype(np.float64)
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        #y = y * 2 - 1
        n, m = X.shape
        if weights is None:
            weights = np.ones(len(X))

        res = rob_stump_fit(X, y, weights, self.max_search_per_feature)
        self.model = res[0]
        ##### interface for attack
        self.b = res[0][1]
        self.coord = res[0][0]
        self.w_l = -1
        self.w_r = 2.
        #####
        return self

    def predict(self, X):
        #check_is_fitted(self)
        X = check_array(X)
        return (X[:, self.model[0]] >= self.model[1]).astype(np.int) * 2 - 1

class BoostingByMajority(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, gamma=0.1, direction=1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.direction = direction
        self.gamma = gamma
        if base_estimator is None:
            self.base_estimator = DecisionStump()

    def _wt(self, t, s):
        T = self.n_estimators
        temp = (T-t-s+1)/2
        temp2 = (T-t+s-1)/2
        ret = 0.5 * comb(T-t, np.floor(temp)) * ((0.5 + self.gamma) ** np.floor(temp)) * ((0.5 - self.gamma) ** np.floor(temp2))
        return ret

    def preprocess_X(self, X):
        return X * self.direction

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = self.preprocess_X(X)
        self.classes_ = unique_labels(y)
        self.estimators = []

        ss = np.zeros(len(X))
        for t in range(1, self.n_estimators+1):
            weights = self._wt(t, ss) + 1e-6
            weights = weights / weights.sum()
            self.estimators.append(clone(self.base_estimator))
            self.estimators[-1].fit(X, y, weights)
            preds = self.estimators[-1].predict(X)
            acc = ((y == preds) * weights).sum()
            if acc < 0.5 + self.gamma:
                break
            ss = ss + y * preds

        return self

    def predict(self, X):
        X = self.preprocess_X(X)
        preds = self.predict_real(X)
        return np.sign(preds)

    def predict_real(self, X):
        #check_is_fitted(self)
        X = check_array(X)
        X = self.preprocess_X(X)

        preds = np.zeros(len(X))
        for clf in self.estimators:
            preds += clf.predict(X)
        return preds

    def predict_proba(self, X, n_estimators=-1):
        #check_is_fitted(self)
        X = check_array(X)
        X = self.preprocess_X(X)
        if n_estimators == -1:
            n_estimators = len(self.estimators)

        preds = np.zeros(len(X))
        for clf in self.estimators[:n_estimators]:
            preds += (clf.predict(X) + 1) // 2
        preds = preds / n_estimators
        return preds

#check_estimator(BoostingByMajority())