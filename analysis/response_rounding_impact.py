"""
How much does the boundary-rounding issue in src/minmaxML.compute_model_errors change the
paper's headline comparison? Reproduces the three methods of the experiments section
(non-strategic, naive strategic, ours) with the Robust PRC oracle and reports the test
worst-group error of the uniform mixture, scoring the agents' best response either
with the repo's code (EXACT_RESPONSE=0) or exactly (EXACT_RESPONSE=1).
Usage: EXACT_RESPONSE=0|1 python analysis/response_rounding_impact.py DATASET TAU f1,... [numsteps] [seed]
"""
import sys, os, copy, json
import numpy as np
sys.path.insert(0, os.getcwd())
import analysis.deployed_guarantee as dg
from analysis.deployed_guarantee import load, group_err, strat_err
from src.train_test_split import create_validation_split
from src.minmaxML import shift_model
from src.paired_regression_classifier import PairedRegressionClassifier
from sklearn.linear_model import LinearRegression


def nonstrat_err(model, X, y):
    return (model.predict(X).ravel() != y).astype(float)


def run(name, tau, fracs, numsteps, seed, step=0.1):
    X, y, gl2, _ = load(name)
    gl2 = np.atleast_2d(gl2); gl = gl2.flatten(); G = len(np.unique(gl))
    fracs = np.array(fracs, float); tauv = fracs[gl] * tau
    Xtr, Xte, ytr, yte, gtr, gte, ttr, tte = create_validation_split(X, y, tauv, gl2, 0.3, seed)
    gtr, gte = gtr.flatten(), gte.flatten()
    n = len(ytr); sizes = np.array([np.sum(gtr == g) for g in range(G)])
    tau_mean = float(np.mean(tauv))
    shifts = [tau * i * step for i in range(int(round(fracs.max() / step)) + 1)]
    res = {}
    for method in ('nonstrategic', 'naive', 'ours'):
        lam = sizes / n; TE = []
        for t in range(1, numsteps + 1):
            sw = lam[gtr] / sizes[gtr]
            m = PairedRegressionClassifier(regressor_class=LinearRegression).fit(Xtr, ytr, sw)
            if method == 'ours':
                best = None
                for s in shifts:
                    ge = group_err(strat_err(shift_model(copy.deepcopy(m), s), Xtr, ytr, ttr), gtr, G)
                    if best is None or (ge.max(), ge.sum()) < best[0]:
                        best = ((ge.max(), ge.sum()), s, ge)
                m = shift_model(m, best[1]); ge_tr = best[2]
                mt = m
            else:
                ge_tr = group_err(nonstrat_err(m, Xtr, ytr), gtr, G)
                mt = shift_model(copy.deepcopy(m), tau_mean) if method == 'naive' else m
            TE.append(group_err(strat_err(mt, Xte, yte, tte), gte, G))
            lam = lam * np.exp(t ** -0.5 * ge_tr); lam /= lam.sum()
        res[method] = float(np.mean(TE, 0).max())
    return dict(dataset=name, tau=tau, fracs=fracs.tolist(), seed=seed, T=numsteps,
                exact_response=dg.EXACT, **res)


if __name__ == '__main__':
    name, tau = sys.argv[1], float(sys.argv[2])
    fracs = [float(v) for v in sys.argv[3].split(',')]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 4235255
    print(json.dumps(run(name, tau, fracs, steps, seed)))
