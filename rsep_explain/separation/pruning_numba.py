
import numpy as np
from numba.typed import List
from numba import njit

@njit
def get_l2_dist(a, b):
    return np.linalg.norm(a-b, ord=2)

@njit
def get_l1_dist(a, b):
    return np.linalg.norm(a-b, ord=1)

@njit
def get_linf_dist(a, b):
    return np.linalg.norm(a-b, ord=np.inf)

@njit
def collision_graph_numba(X, y, eps, dist_fn):
    n = len(X)
    adj_lst = List()
    graph = List()
    
    for i in range(n-1):
        adj_lst.append(List())
        graph.append(List())
        for j in range(i+1, n):
            if y[i] != y[j]:
                dist = dist_fn(X[i], X[j])
                if dist <= 2 * eps:
                    adj_lst[i].append(j)
                    adj_lst[j].append(i)
                    if y[i] == 1:
                        graph[i].append(j)
                    else:
                        graph[j].append(i)
    return adj_lst, graph

def get_dist_fn(sep_measure):
    if sep_measure == 1:
        return get_l1_dist
    elif sep_measure == 2:
        return get_l2_dist
    elif sep_measure == np.inf:
        return get_linf_dist
    #elif sep_measure == 'min_measure':
    #    dist = min(d)
    else:
        raise ValueError("Not supported measure %s for collision",
                            str(sep_measure))

def build_collision_graph_numba(X, y, eps, sep_measure):
    adj_lst, graph = collision_graph_numba(X, y, eps, get_dist_fn(sep_measure))
    adj_list = {i: list(l) for i, l in enumerate(adj_lst)}
    graph = {i: list(l) for i, l in enumerate(graph) if len(l) > 0}
    return adj_list, graph