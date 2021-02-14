"""
https://github.com/max-andr/provably-robust-boosting/blob/master/attacks.py
"""
import numpy as np

def exact_attack_stumps(model, X, y):
    """ Fast exact adv. examples for boosted stumps.
        `f` is a StumpEnsemble object.
    """
    min_val = 1e-4
    num, dim = X.shape
    deltas = np.zeros([num, dim])
    db_dists = np.full(num, np.inf)

    for i in range(num):
        # 0.0 means we just check whether the point is originally misclassified; if yes  =>  db_dist=0
        #eps_all_i = [0.0]
        eps_all_i = [min_val]
        for wl in model.estimators:
            eps_all_i.append(np.abs(wl.b - X[i, wl.coord] + min_val*np.sign(wl.b - X[i, wl.coord])))
        eps_all_i = np.unique(np.asarray(eps_all_i))
        eps_sorted = np.sort(eps_all_i)
        #print(eps_sorted)
        for k, eps in enumerate(eps_sorted):
            if eps < min_val or (k > 0 and abs(eps_sorted[k-1] - eps_sorted[k]) < min_val):
                continue
            # Clear unvectorized version
            yf_min = 0.0
            delta = np.zeros(dim)
            for coord, trees_current_coord in model.coords_trees.items():
                yf_min_coord_base, yf_orig_pt = 0.0, 0.0
                for tree in trees_current_coord:
                    yf_min_coord_base += y[i] * tree.predict(X[None, i] - eps)[0]
                    yf_orig_pt += y[i] * tree.predict(X[None, i])[0]

                unstable_thresholds, unstable_wr_values = [X[i, coord] - eps], [0.0]
                for tree in trees_current_coord:
                    # excluding the left equality since we have already evaluated it
                    if X[i, coord] - eps < tree.b <= X[i, coord] + eps: # changed
                        unstable_thresholds.append(tree.b)
                        unstable_wr_values.append(tree.w_r)
                unstable_thresholds = np.array(unstable_thresholds)
                unstable_wr_values = np.array(unstable_wr_values)
                idx = np.argsort(unstable_thresholds)
                unstable_thresholds = unstable_thresholds[idx]

                sorted_y_wr = (y[i] * np.array(unstable_wr_values))[idx]
                yf_coord_interval_vals = np.cumsum(sorted_y_wr)
                yf_min_coord = yf_min_coord_base + yf_coord_interval_vals.min()
                yf_min += yf_min_coord
                #if np.isclose(eps, 1.2551) and coord == 53 and np.isclose(X[i, coord], 0.033):
                #    import ipdb; ipdb.set_trace()
                #if np.isclose(eps, 2.5001) and i == 10:
                #    print("========================")
                #    print(eps, coord, X[i, coord])
                #    print(yf_coord_interval_vals)
                #    print(yf_min_coord_base)
                #    print(unstable_thresholds, unstable_wr_values)
                #    print(yf_min, yf_min_coord)

                i_opt_threshold = yf_coord_interval_vals.argmin()
                # if the min value is attained at the point itself, take it instead; so that we do not take
                # unnecessary -eps deltas (which would not anyway influence Linf size, but would bias the picture)
                if yf_min_coord == yf_orig_pt:
                    opt_threshold = X[i, coord]  # i.e. the final delta is 0.0
                else:
                    opt_threshold = unstable_thresholds[i_opt_threshold]
                delta[coord] = opt_threshold - X[i, coord] - y[i] * 1e-8
                #print(coord, X[i, coord], delta[coord], eps, y[i])
                #print([(t.b, t.coord, X[i, coord] >= t.b, (X[i, coord] - y[i] * delta[coord]) >= t.b) for t in model.coords_trees[coord]])
                #print(sum([(X[i, coord] - y[i] * delta[coord]) >= t.b for t in model.coords_trees[coord]]))

            yf = float(y[i] * model.predict_real(X[None, i] + delta[None])[0])
            #print('eps_max={:.3f}, eps_delta={:.3f}, yf={:.3f}, yf_min={:.3f}, nnz={}'.format(
            #    eps, np.abs(delta).max(), yf, yf_min, (delta != 0.0).sum()))
            #print()
            if np.abs(yf) != np.abs(yf_min):
                import ipdb; ipdb.set_trace()
            if yf_min < 0:
                if yf != yf_min:
                    print(yf, yf_min)
                    continue
                db_dists[i] = eps
                deltas[i] = delta
                break
        #print()
        yf = y[i] * model.predict(X[None, i] + deltas[None, i])
        #if yf >= 0.0:
        if yf > 0.0:
            print('The class was not changed! Some bug apparently!') # There may be numarical errors some times
            print(i, model.predict_real(X[None, i] + deltas[None, i]), model.predict_real(X[None, i]), y[i])
            import ipdb; ipdb.set_trace()
            raise ValueError("The class was not changed! Some bug apparently!")
            #import ipdb; ipdb.set_trace()
    return deltas


def optimal_attack_bbm(X, y, model):
    X = X.astype(np.float64)
    X = X * model.direction

    model.coords_trees = {}
    for wl in model.estimators:
        model.coords_trees.setdefault(wl.model[0], []).append(wl)

    deltas = exact_attack_stumps(model, X, y)
    advX = X + deltas
    assert (model.predict(advX) == y).sum() == 0
    return advX