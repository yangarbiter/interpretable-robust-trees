import numpy as np
from .solvers import solve_lp

def get_sol_linf(target_x, G, h):
    fet_dim = target_x.shape[0]
    c = np.concatenate((np.zeros(fet_dim), np.ones(1)))

    G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G = np.hstack((G, np.zeros((G.shape[0], 1))))
    G = np.vstack((G, G2, G3))
    h = np.concatenate((h, target_x, -target_x))

    temph = h - 1e-4

    status, sol = solve_lp(c=c, G=G, h=temph, solver='GLPK')
    if status == 'optimal':
        return np.array(sol).reshape(-1)[:-1]
    else:
        print(status)
        raise ValueError()

def optimal_attack_linear(X, y, w):
    adv_X = np.copy(X).astype(np.float32)
    G = w[1:].reshape(1, -1)
    h = np.ones((1, 1)) * w[0]
    preds = np.sign(np.dot(X, w))
    for i, (x, yi) in enumerate(zip(X, y)):
        x = x.reshape(-1, 1)
        if preds[i] != yi:
            continue
        # Gx + h
        adv_x = get_sol_linf(x[1:, :], preds[i] * G, -1 * preds[i] * h)

        adv_X[i, 1:] = adv_x
        if np.sign(np.dot(adv_X[i], w)).astype(int) == yi:
            import ipdb; ipdb.set_trace()
        assert np.sign(np.dot(adv_X[i], w)).astype(int) != yi, (i, np.dot(adv_X[i], w), np.sign(np.dot(adv_X[i], w)).astype(int), yi)
    return adv_X