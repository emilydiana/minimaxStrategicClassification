"""
Oracle-quality check for the MS revision (OR AE critique 3: heuristic oracle).

Runs Algorithm 1 (exponential weights over groups) twice on the same split:
  (a) with the Robust PRC heuristic used in the paper (PRC fit + shift grid search),
  (b) with a stronger oracle: for each candidate direction w (PRC direction, weighted
      logistic-regression direction, previous round's best, and a local random search
      around the best), the threshold b is chosen *exactly* by a sweep over the n
      critical values, minimizing the lambda-weighted strategic training error.
Reports, per round, the weighted objective each oracle attains at the same lambda,
and the final worst-group strategic error (train/test) of the uniform mixture.
Strategic response follows src/minmaxML.compute_model_errors: an agent with
signed distance s < 0 to the boundary moves iff -s < tau_i.
Usage: python analysis/oracle_gap.py DATASET TAU f1,f2,... [numsteps] [seed] [n_search]
"""
import sys, os, json, types
import numpy as np
sys.path.insert(0, os.getcwd())
try:
    import torch  # noqa
except ImportError:
    _tw = types.ModuleType('src.torch_wrapper'); _tw.MLPClassifier = None
    sys.modules['src.torch_wrapper'] = _tw
from sklearn.linear_model import LinearRegression, LogisticRegression
from analysis.deployed_guarantee import load
from src.train_test_split import create_validation_split


def strat_pos(w, b, X, tauv):
    s = (X @ w + b) / np.linalg.norm(w)
    return (s >= 0) | ((s < 0) & (-s < tauv))


def group_err(pos, y, gl, G):
    e = (pos != (y == 1))
    return np.array([e[gl == g].mean() for g in range(G)])


def best_threshold(u, y, tauv, sw):
    """u = X w with ||w|| = 1. Positive iff u + b > -tau (strictly) or u + b >= 0.
    Returns b minimizing sum_i sw_i * err_i, evaluated at midpoints between critical values."""
    c = -u - tauv  # agent i is positive iff b > c_i (tau_i > 0); for tau_i = 0, iff b >= c_i
    order = np.argsort(c)
    cs, ys, ws = c[order], y[order], sw[order]
    # b below all c: everyone negative -> error = weight of positives
    pos_w = np.where(ys == 1, ws, 0.0); neg_w = np.where(ys == 0, ws, 0.0)
    # after passing c_(k), points 0..k are positive
    err = pos_w.sum() - np.cumsum(pos_w) + np.cumsum(neg_w)
    errs = np.r_[pos_w.sum(), err]
    k = int(np.argmin(errs))
    if k == 0:
        b = cs[0] - 1.0
    elif k == len(cs):
        b = cs[-1] + 1.0
    else:
        b = 0.5 * (cs[k - 1] + cs[k])
    return b, errs[k]


def strong_oracle(X, y, tauv, sw, init_dirs, n_search, rng):
    best = (np.inf, None, None)
    for w in init_dirs:
        w = w / np.linalg.norm(w)
        b, e = best_threshold(X @ w, y, tauv, sw)
        if e < best[0]:
            best = (e, w, b)
    e0, w0, b0 = best
    sigma, fails = 0.3, 0
    for _ in range(n_search):
        w = w0 + sigma * rng.standard_normal(len(w0)); w /= np.linalg.norm(w)
        b, e = best_threshold(X @ w, y, tauv, sw)
        if e < e0 - 1e-12:
            e0, w0, b0, fails = e, w, b, 0
        else:
            fails += 1
            if fails >= 20:
                sigma, fails = sigma / 2, 0
    return w0, b0


def prc_oracle(X, y, tauv, sw, gl, G, tau, fracs, step=0.1):
    costs = y * sw - (1 - y) * sw
    reg = LinearRegression().fit(X, costs)
    w, b0 = reg.coef_.copy(), float(reg.intercept_)
    nw = np.linalg.norm(w)
    best = None
    for i in range(int(round(fracs.max() / step)) + 1):
        b = b0 - tau * i * step * nw
        ge = group_err(strat_pos(w, b, X, tauv), y, gl, G)
        key = (ge.max(), ge.sum())
        if best is None or key < best[0]:
            best = (key, b)
    return w, best[1]


def run_alg1(oracle, Xtr, ytr, gtr, ttr, Xte, yte, gte, tte, G, numsteps):
    n = len(ytr)
    sizes = np.array([np.sum(gtr == g) for g in range(G)])
    lam = sizes / n
    TR, TE, OBJ = [], [], []
    for t in range(1, numsteps + 1):
        sw = lam[gtr] / sizes[gtr]
        w, b = oracle(sw)
        ge_tr = group_err(strat_pos(w, b, Xtr, ttr), ytr, gtr, G)
        ge_te = group_err(strat_pos(w, b, Xte, tte), yte, gte, G)
        TR.append(ge_tr); TE.append(ge_te); OBJ.append(float(lam @ ge_tr))
        lam = lam * np.exp(t ** -0.5 * ge_tr); lam /= lam.sum()
    return np.array(TR), np.array(TE), np.array(OBJ)


def main(name, tau, fracs, numsteps, seed, n_search):
    X, y, gl2, _ = load(name)
    gl2 = np.atleast_2d(gl2); gl = gl2.flatten()
    G = len(np.unique(gl)); fracs = np.array(fracs, float); tauv = fracs[gl] * tau
    Xtr, Xte, ytr, yte, gtr, gte, ttr, tte = create_validation_split(X, y, tauv, gl2, 0.3, seed)
    gtr, gte = gtr.flatten(), gte.flatten()
    rng = np.random.default_rng(0)
    prc = lambda sw: prc_oracle(Xtr, ytr, ttr, sw, gtr, G, tau, fracs)
    state = {'w': None}

    def strong(sw):
        w_prc, _ = prc(sw)
        inits = [w_prc]
        try:
            lr = LogisticRegression(max_iter=500).fit(Xtr, ytr, sample_weight=sw * len(ytr))
            inits.append(lr.coef_.ravel())
        except Exception:
            pass
        if state['w'] is not None:
            inits.append(state['w'])
        w, b = strong_oracle(Xtr, ytr, ttr, sw, inits, n_search, rng)
        state['w'] = w
        return w, b

    # Gap at matched lambda: evaluate both oracles along the PRC run's lambda path
    TRp, TEp, OBJp = run_alg1(prc, Xtr, ytr, gtr, ttr, Xte, yte, gte, tte, G, numsteps)
    TRs, TEs, OBJs = run_alg1(strong, Xtr, ytr, gtr, ttr, Xte, yte, gte, tte, G, numsteps)
    out = dict(dataset=name, tau=tau, fracs=fracs.tolist(), G=G, T=numsteps, seed=seed, n_search=n_search,
               prc_mix_train_max=float(TRp.mean(0).max()), prc_mix_test_max=float(TEp.mean(0).max()),
               strong_mix_train_max=float(TRs.mean(0).max()), strong_mix_test_max=float(TEs.mean(0).max()),
               prc_obj_mean=float(OBJp.mean()), strong_obj_mean=float(OBJs.mean()),
               # per-round regret-style gap: at round 1 both oracles see the same lambda (= group proportions)
               round1_obj_prc=float(OBJp[0]), round1_obj_strong=float(OBJs[0]))
    # matched-lambda comparison on a few lambdas from the PRC path
    n = len(ytr); sizes = np.array([np.sum(gtr == g) for g in range(G)])
    lam = sizes / n; gaps = []
    for t in range(1, numsteps + 1):
        if t in (1, 10, 100, numsteps // 2, numsteps):
            sw = lam[gtr] / sizes[gtr]
            wp, bp = prc(sw); state['w'] = None; ws, bs = strong(sw)
            op = lam @ group_err(strat_pos(wp, bp, Xtr, ttr), ytr, gtr, G)
            os_ = lam @ group_err(strat_pos(ws, bs, Xtr, ttr), ytr, gtr, G)
            gaps.append((t, float(op), float(os_)))
        lam = lam * np.exp(t ** -0.5 * TRp[t - 1]); lam /= lam.sum()
    out['matched_lambda_obj'] = gaps
    return out


if __name__ == '__main__':
    name, tau = sys.argv[1], float(sys.argv[2])
    fracs = [float(v) for v in sys.argv[3].split(',')]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 4235255
    ns = int(sys.argv[6]) if len(sys.argv) > 6 else 200
    print(json.dumps(main(name, tau, fracs, steps, seed, ns)))
