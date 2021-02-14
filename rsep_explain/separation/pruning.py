import numpy as np
from sklearn.metrics import pairwise_distances

from .hopcroftkarp import HopcroftKarp

def adversarial_pruning(X, y, eps, sep_measure):
    if len(np.unique(y)) != 2:
        raise ValueError("Can only deal with number of classes = 2"
                         "got %d", len(np.unique(y)))
    y = y.astype(int)*2-1
    augX, augy = find_eps_separated_set(X, eps, y, ord=sep_measure)
    augy = (augy+1)//2
    return augX, augy

def build_collision_graph_sk(X, y, eps, sep_measure):
    dists = pairwise_distances(X, metric='minkowski', p=sep_measure, n_jobs=12)
    dists = (dists <= 2*eps) * (y.reshape(-1, 1) != y.reshape(1, -1))

    adj_lst = {i: np.where(l)[0] for i, l in enumerate(dists)}
    graph = {i: set(np.where(l)[0]) for i, l in enumerate(dists) if y[i] == 1}
    return adj_lst, graph

def build_collision_graph(eps, pts, y, ord):
    '''
        This function builds the collision/conflict graph for the max-matching algorithm.
    '''
    n = len(pts)
    graph = dict()
    
    adj_lst = dict()
    for i in range(n):
        adj_lst[i] = set([])
    
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i] != y[j]:
                p1 = pts[i]
                p2 = pts[j]
                d = p1 - p2
                if ord in [1, 2, np.inf]:
                    dist = np.linalg.norm(d, ord=ord)
                elif ord == 'min_measure':
                    dist = min(d)
                else:
                    raise ValueError("Not supported measure %s for collision",
                                     str(ord))
                if (dist <= 2*eps):
                    adj_lst[i].add(j)
                    adj_lst[j].add(i)
                    if y[i] == 1:
                        if i in graph:
                            graph[i].add(j)
                        else: 
                            graph[i] = set([j])
                    else:
                        if j in graph:
                            graph[j].add(i)
                        else:
                            graph[j] = set([i])
    #print 'Done: find collision graph'
    return adj_lst, graph

def find_matching(graph):
    match = HopcroftKarp(graph).maximum_matching()
    #print 'Done: find matching'
    return match

def find_Z(graph, matching, U, adj_lst):
    # a simple bfs finding Z, set of vertices reachable from U via alternating path.
    visited_via_matched_edge = set([p for p in U])
    visited_via_unmatched_edge = set([p for p in U])
    while True:
        set1 = set([item for item in visited_via_unmatched_edge])
        set2 = set([item for item in visited_via_matched_edge])
        for u in visited_via_matched_edge:
            if u in matching:
                for v in adj_lst[u]:
                    if v != matching[u]:
                        set1.add(v)
            else:
                for v in adj_lst[u]:
                    set1.add(v)
                    
        for u in visited_via_unmatched_edge:
            if u in matching:
                for v in adj_lst[u]:
                    if v == matching[u]:
                        set2.add(v)
                        
        if (len(set1) == len(visited_via_unmatched_edge)) and (len(set2) == len(visited_via_matched_edge)):
            return set1.union(set2)
        
        visited_via_matched_edge = set2
        visited_via_unmatched_edge = set1

def find_min_cover(graph, adj_lst, y_pts):
    '''
        This function finds the min-cover of a bipartite graph from max-matching.
        graph: the collision graph dictionary, each key is a vertex while the value is the set of vertices adjacent to it. Note than a vertex can only appear in either the key or the value list,never both.
        adj_lst: similar to graph, except that the key set contains all vertices.
        has_collison: the set of points that has collision with other points
        y_pts: label of points in the graph. 
    ''' 
    matching = find_matching(graph)
    L = set([i for i in graph if y_pts[i] > 0])
    R = set([i for i in graph if y_pts[i] <= 0])
    U = set([i for i in graph if (i not in matching) and (i in L)])
    Z = find_Z(graph, matching, U, adj_lst)
    K = (L.difference(Z)).union(R.intersection(Z))
    return [matching, K]

def find_eps_separated_set(pts, eps, y_pts, ord):
    '''
        This function finds the epsilon-separated subset with the largest cardinality.
        Args:
            pts: the set of input points
            y_pts: the label of pts
            eps: the epsilon value
        Output:
            good_pts: the desired epsilon-separated set
            good_y:   the label of good_pts
        Brief description of steps:
            1) find the set of points that have conflict with other points
            2) build the conflict adjacency graph, conflict points are adjacent
            3) find the min-cover of the adjacency graph
            4) remove the min-cover from the original set pts.
    '''
    y_pts = np.array([1 if i>0 else -1 for i in y_pts])
    #adj_lst, graph = build_collision_graph(eps, pts, y_pts, ord)
    #adj_lst, graph = build_collision_graph_numba(pts, y_pts, eps, ord)
    adj_lst, graph = build_collision_graph_sk(pts, y_pts, eps, ord)
    matching, min_cover = find_min_cover(graph, adj_lst, y_pts)
    min_cover = np.array(list(min_cover), dtype=np.int)
    good_pts = np.delete(pts, min_cover, axis=0)
    good_y = np.delete(y_pts, min_cover, axis=0)
    #print('size of min_cover is: ', len(min_cover), 'size of matching is: ', len(matching))
    return good_pts, good_y