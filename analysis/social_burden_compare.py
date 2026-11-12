"""
Social burden (Definition in the paper; computed as on the social_burden branch: for positives who move,
the distance moved; averaged over each group's positives; mixture = average over rounds) for
non-strategic, naive, and ours, with ours run under three variants:
  old    = repo's rounding in best-response scoring + max-group shift rule
  fixed  = exact best-response scoring + max-group shift rule
  werm   = exact best-response scoring + weighted shift rule
Non-strategic and naive train without strategic agents, so their models do not depend on the variant;
only their test error does (reported under both scorings).
Usage: python analysis/social_burden_compare.py DATASET f1,.. seedidx T out.jsonl
Env: SB_TAUS (comma list), SB_VARIANTS (subset of the five runs).
"""
import sys, os, copy, json
import numpy as np
sys.path.insert(0, os.getcwd())
import analysis.deployed_guarantee as dg
from analysis.deployed_guarantee import load, group_err
from src.train_test_split import create_validation_split
from src.minmaxML import shift_model, compute_model_errors
from src.paired_regression_classifier import PairedRegressionClassifier
from sklearn.linear_model import LinearRegression


def scores(m, X):
    w, b = m.regressor.coef_, m.regressor.intercept_
    return (X @ w + b) / np.linalg.norm(w)


def strat_err(m, X, y, tv, exact):
    if exact:
        s = scores(m, X); pos = (s >= 0) | (-s < tv)
        return (pos != (y == 1)).astype(float)
    E = np.zeros((1, len(y))); compute_model_errors(m, X, y, 0, E, '0/1 Loss', 'none', 1.0, True, tv)
    return E[0]


def burden(m, X, y, tv):
    s = scores(m, X); move = (s < 0) & (-s < tv)
    return np.where(move & (y == 1), -s, 0.0)


def gpos_mean(v, y, g, G):
    return np.array([v[(g == k) & (y == 1)].mean() if np.any((g == k) & (y == 1)) else 0.0 for k in range(G)])


name = sys.argv[1]; fracs = np.array([float(v) for v in sys.argv[2].split(',')])
seed = 4235255 + int(sys.argv[3]); T = int(sys.argv[4]); out = sys.argv[5]
X, y, gl2, _ = load(name); gl2 = np.atleast_2d(gl2); gl = gl2.flatten(); G = len(np.unique(gl))
TAUS = [float(v) for v in os.environ.get('SB_TAUS', '0,1,2,3,4,5,6').split(',')]
VARIANTS = os.environ.get('SB_VARIANTS', 'nonstrategic,naive,ours_old,ours_fixed,ours_werm').split(',')
for tau in TAUS:
    tauv = fracs[gl] * tau
    Xtr, Xte, ytr, yte, gtr, gte, ttr, tte = create_validation_split(X, y, tauv, gl2, 0.3, seed)
    gtr, gte = gtr.flatten(), gte.flatten()
    n = len(ytr); sizes = np.array([np.sum(gtr == k) for k in range(G)]); tmean = float(np.mean(tauv))
    shifts = [tau * i * 0.1 for i in range(int(round(fracs.max() / 0.1)) + 1)]
    rec = dict(dataset=name, fracs=fracs.tolist(), tau=tau, seed=seed, T=T)
    runs = [('nonstrategic', None, None), ('naive', None, None), ('ours_old', False, 'maxgroup'),
            ('ours_fixed', True, 'maxgroup'), ('ours_werm', True, 'weighted')]
    for mname, exact, rule in runs:
        if mname not in VARIANTS:
            continue
        lam = sizes / n; SB = []; E_old = []; E_new = []
        for t in range(1, T + 1):
            sw = lam[gtr] / sizes[gtr]
            m = PairedRegressionClassifier(regressor_class=LinearRegression).fit(Xtr, ytr, sw)
            if mname.startswith('ours'):
                best = None
                for sft in shifts:
                    e = strat_err(shift_model(copy.deepcopy(m), sft), Xtr, ytr, ttr, exact)
                    ge = group_err(e, gtr, G)
                    key = (float(sw @ e), ge.max()) if rule == 'weighted' else (ge.max(), ge.sum())
                    if best is None or key < best[0]:
                        best = (key, sft, ge)
                m = shift_model(m, best[1]); ge_tr = best[2]; mt = m
            else:
                ge_tr = group_err((m.predict(Xtr).ravel() != ytr).astype(float), gtr, G)
                mt = shift_model(copy.deepcopy(m), tmean) if mname == 'naive' else m
            SB.append(gpos_mean(burden(mt, Xte, yte, tte), yte, gte, G))
            E_old.append(group_err(strat_err(mt, Xte, yte, tte, False), gte, G))
            E_new.append(group_err(strat_err(mt, Xte, yte, tte, True), gte, G))
            lam = lam * np.exp(t ** -0.5 * ge_tr); lam /= lam.sum()
        rec[mname + '_sb'] = float(np.mean(SB, 0).max())
        rec[mname + '_err_oldscore'] = float(np.mean(E_old, 0).max())
        rec[mname + '_err_exact'] = float(np.mean(E_new, 0).max())
    with open(out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
