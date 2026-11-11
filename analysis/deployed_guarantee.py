"""
Deployed-classifier analysis for the MS revision (OR AE critique 2).

Runs the "ours" learner (exponential weights over groups + Robust PRC oracle with
per-round shift grid search), records every round's classifier h_t, and compares:
  * the uniform mixture over rounds (what the paper reports, an expectation),
  * the distribution of worst-group strategic test error of a single draw h_t,
  * an LP-sparsified mixture: min_{p in simplex} max_g sum_t p_t err_g(h_t) on train
    (a basic optimal solution has support <= G+1),
  * the best single iterate (chosen on train).
Usage: python analysis/deployed_guarantee.py DATASET TAU f1,f2,... [numsteps]
Run from the repo root.
"""
import sys, copy, json, os
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, os.getcwd())
import types
if 'torch' not in sys.modules:
    try:
        import torch  # noqa
    except ImportError:  # torch only needed for the MLP model; stub it
        _tw = types.ModuleType('src.torch_wrapper'); _tw.MLPClassifier = None
        sys.modules['src.torch_wrapper'] = _tw
from dataset_mapping import get_dataset_features
from src.setup_matrices import setup_matrices
from src.train_test_split import create_validation_split
from src.minmaxML import rescale_feature_matrix, shift_model, compute_model_errors
from src.paired_regression_classifier import PairedRegressionClassifier
from sklearn.linear_model import LinearRegression


def load(name):
    path, label, groups, feats, cats, drop, _ = get_dataset_features(name)
    X, y, gl, gnames, _, _ = setup_matrices(path, label, groups, usable_features=feats,
                                            drop_group_as_feature=True, categorical_columns=cats,
                                            groups_to_drop=drop, verbose=False, save_data=False)
    return rescale_feature_matrix(X).astype(np.float64), y, gl, np.array(gnames).flatten()


def group_err(e, gl, G):
    return np.array([e[gl == g].mean() for g in range(G)])


# The repo's compute_model_errors projects movers exactly onto the boundary, and floating-point
# rounding then leaves many of them at pred ~ -1e-18, so they are scored as rejected even though
# they manipulated. EXACT_RESPONSE=1 (default) scores the best response exactly instead:
# positive iff signed distance s >= 0, or s < 0 and -s < tau_i (same rule as the repo, no rounding).
EXACT = os.environ.get('EXACT_RESPONSE', '1') == '1'


def strat_err(model, X, y, tauv):
    if EXACT:
        w, b = model.regressor.coef_, model.regressor.intercept_
        s = (X @ w + b) / np.linalg.norm(w)
        pos = (s >= 0) | (-s < tauv)
        return (pos != (y == 1)).astype(float)
    E = np.zeros((1, len(y)))
    compute_model_errors(model, X, y, 0, E, '0/1 Loss', 'none', 1.0, True, tauv)
    return E[0]


def run(name, tau, fracs, numsteps=3000, seed=4235255, step=0.1):
    X, y, gl2, gnames = load(name)
    gl2 = np.atleast_2d(gl2)
    gl = gl2.flatten()
    G = len(np.unique(gl))
    fracs = np.array(fracs, dtype=float)
    tauv = fracs[gl] * tau
    Xtr, Xte, ytr, yte, gtr, gte, ttr, tte = create_validation_split(X, y, tauv, gl2, 0.3, seed)
    gtr, gte = gtr.flatten(), gte.flatten()
    n = len(ytr)
    sizes = np.array([np.sum(gtr == g) for g in range(G)])
    gw = sizes / n
    shifts = [tau * i * step for i in range(int(round(fracs.max() / step)) + 1)]
    TR, TE, SH = [], [], []
    for t in range(1, numsteps + 1):
        sw = gw[gtr] / sizes[gtr]
        m = PairedRegressionClassifier(regressor_class=LinearRegression).fit(Xtr, ytr, sw)
        best = None
        for s in shifts:
            c = shift_model(copy.deepcopy(m), s)
            ge = group_err(strat_err(c, Xtr, ytr, ttr), gtr, G)
            key = (ge.max(), ge.sum())
            if best is None or key < best[0]:
                best = (key, s, ge)
        _, s, ge_tr = best
        m = shift_model(m, s)
        ge_te = group_err(strat_err(m, Xte, yte, tte), gte, G)
        TR.append(ge_tr); TE.append(ge_te); SH.append(s)
        eta = t ** -0.5
        gw = gw * np.exp(eta * ge_tr); gw /= gw.sum()
    TR, TE = np.array(TR), np.array(TE)
    T = len(TR)
    # LP: min z s.t. TR^T p <= z, sum p = 1, p >= 0
    c = np.r_[np.zeros(T), 1.0]
    A_ub = np.c_[TR.T, -np.ones(G)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(G), A_eq=np.r_[np.ones(T), 0.0][None, :], b_eq=[1.0],
                  bounds=[(0, None)] * T + [(None, None)], method='highs-ds')
    p = np.maximum(res.x[:T], 0); p /= p.sum()
    supp = np.flatnonzero(p > 1e-9)
    per_draw_te = TE.max(axis=1)
    best_single = int(np.argmin(TR.max(axis=1)))
    out = dict(
        dataset=name, tau=tau, fracs=fracs.tolist(), G=G, group_names=gnames.tolist(), T=T,
        mixture_train_max=float(TR.mean(0).max()), mixture_test_max=float(TE.mean(0).max()),
        mixture_test_group=TE.mean(0).round(4).tolist(),
        draw_test_max_mean=float(per_draw_te.mean()),
        draw_test_max_q=[float(np.quantile(per_draw_te, q)) for q in (0.05, 0.5, 0.95)],
        draw_test_max_worst=float(per_draw_te.max()),
        frac_draws_above_mixture_plus_2pp=float(np.mean(per_draw_te > TE.mean(0).max() + 0.02)),
        lp_train_max=float(res.fun), lp_support=int(len(supp)),
        lp_test_max=float((p @ TE).max()),
        lp_draw_test_max=[float(TE[i].max()) for i in supp], lp_weights=p[supp].round(4).tolist(),
        best_single_train_max=float(TR[best_single].max()), best_single_test_max=float(TE[best_single].max()),
        distinct_shifts=sorted(set(np.round(SH, 4).tolist())),
    )
    return out


if __name__ == '__main__':
    name, tau = sys.argv[1], float(sys.argv[2])
    fracs = [float(v) for v in sys.argv[3].split(',')]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 4235255
    out = run(name, tau, fracs, steps, seed)
    out['seed'] = seed
    out['exact_response'] = EXACT
    print(json.dumps(out))
