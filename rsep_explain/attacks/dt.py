from copy import deepcopy
import ipdb

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from tqdm import tqdm

from .solvers import solve_lp

def get_sol_linf(target_x, target_y, paths, tree, constraints):
    value = tree.value
    fet_dim = tree.n_features
    temp = (target_x, np.inf)

    for i, path in enumerate(paths):
        if np.argmax(value[path[-1]]) != target_y:
            G, h = constraints[i]

            c = np.concatenate((np.zeros(fet_dim), np.ones(1)))

            G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
            G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
            G = np.hstack((G, np.zeros((G.shape[0], 1))))
            G = np.vstack((G, G2, G3))
            h = np.concatenate((h, target_x, -target_x)).reshape(-1, 1)

            temph = h - 1e-4

            status, sol = solve_lp(c=c, G=G, h=temph, n_jobs=4,
                solver='GLPK', glpk={'msg_lev': 'GLP_MSG_OFF'})
            #status, sol = solve_lp(
            #    c=c, G=G, h=temph, n_jobs=4, solver='CVXOPT')
            #status, sol = solve_lp(
            #    c=c, G=G, h=temph, n_jobs=4, solver='GUROBI')
            if status == 'optimal':
                sol = np.array(sol).reshape(-1)[:-1]
                eps = np.linalg.norm(sol - target_x, ord=np.inf)
                if eps < temp[1]:
                    temp = (sol, eps)

    if temp [1] != np.inf:
        return temp[0] - target_x
    else:
        return np.zeros_like(target_x)

def _get_path_constraints(clf, path, direction):
    direction = np.asarray(direction)
    path = np.asarray(path)[:-1]

    tree = clf.tree_
    threshold = tree.threshold
    feature = tree.feature

    h = threshold[path]
    G = np.zeros((len(path), tree.n_features), np.float64)
    G[np.arange(len(path)), feature[path]] = 1

    h = h * direction
    G = G * direction.reshape((-1, 1))

    return G, h

def get_tree_constraints(clf: DecisionTreeClassifier):
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    paths: list = []
    directions: list = []

    path = []
    direction = []

    def _dfs(node_id):
        path.append(node_id)

        if children_left[node_id] != children_right[node_id]:
            direction.append(1)
            _dfs(children_left[node_id])
            direction[-1] = -1
            _dfs(children_right[node_id])
            direction.pop()
        else:
            paths.append(deepcopy(path))
            directions.append(deepcopy(direction))

        path.pop()
    _dfs(0)

    constraints = Parallel(n_jobs=1)(
        delayed(_get_path_constraints)(clf, p, d) for p, d in zip(paths, directions))
    return paths, constraints


class DTOptAttack():
    def __init__(self, clf: DecisionTreeClassifier, norm):
        self.norm = norm
        self.clf = clf
        self.paths, self.constraints = get_tree_constraints(clf)

    def fit(self, X, y):
        pass

    def perturb(self, X, y):
        X = X.astype(np.float64)
        pert_X = np.zeros_like(X, np.float64)
        if len(self.clf.tree_.feature) == 1 and self.clf.tree_.feature[0] == -2:
            # only root and root don't split
            return pert_X

        pred_y = self.clf.predict(X)

        if self.norm == np.inf:
            get_sol_fn = get_sol_linf
        else:
            raise ValueError("norm %s not supported", self.norm)

        #def _helper(sample_id):
        #    if pred_y[sample_id] != y[sample_id]:
        #       return np.zeros_like(X[sample_id])
        #    target_x = X[sample_id]
        #    internal_y = np.where(self.clf.classes_ == y[sample_id])[0][0]
        #    pert_x = get_sol_fn(target_x, internal_y, self.paths,
        #                        self.clf.tree_, self.constraints)
        #    if np.linalg.norm(pert_x) != 0:
        #        assert self.clf.predict([X[sample_id] + pert_x])[0] != y[sample_id]
        #        return pert_x
        #    else:
        #        raise ValueError("shouldn't happen")
        #pert_X = Parallel(n_jobs=2, verbose=1)(
        #    delayed(_helper)(sample_id) for sample_id in range(len(X)))
        #pert_X = np.asarray(pert_X)


        for sample_id in tqdm(range(len(X)), desc="[Attacking DT]"):
            if pred_y[sample_id] != y[sample_id]:
               continue
            target_x = X[sample_id]
            internal_y = np.where(self.clf.classes_ == y[sample_id])[0][0]
            pert_x = get_sol_fn(target_x, internal_y, self.paths,
                                self.clf.tree_, self.constraints)
            if np.linalg.norm(pert_x) != 0:
                assert self.clf.predict([X[sample_id] + pert_x])[0] != y[sample_id]
                pert_X[sample_id, :] = pert_x
            else:
                import ipdb; ipdb.set_trace()
                raise ValueError("shouldn't happen")
            
        assert (self.clf.predict(X + pert_X) == y).sum() == 0

        return pert_X