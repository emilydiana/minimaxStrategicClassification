"""
Regenerates the paper's max-group-error-vs-tau curves by calling src.minmaxML.do_learning
with the same settings as main_driver.py (PRC, a=1, b=1/2, 0/1 loss, test_size=0.3,
learner_tau_step=0.1, split seeds 4235255, 4235256, ...), for the three methods
(non-strategic, ours, naive strategic). Uses whatever src/ is importable from the CWD,
so running it from a checkout of main gives the old (rounding) behaviour.
Usage: python analysis/regen_figures.py DATASET f1,f2,.. taus(comma) n_seeds numsteps out.jsonl
Pareto mode (Algorithm 2): set N_GAMMAS=k. For each method, an unrelaxed run gives the minimax value
gamma_hat; then k equally spaced gammas in [gamma_hat, 0.5] are run with relaxed=True (gammas above the
unrelaxed run's max group error are skipped only if SKIP_LOOSE=1, as main_driver.py does). One record per gamma.
"""
import sys, os, json, types, warnings
import numpy as np
sys.path.insert(0, os.getcwd())
warnings.filterwarnings('ignore')
try:
    import torch  # noqa
except ImportError:
    _tw = types.ModuleType('src.torch_wrapper'); _tw.MLPClassifier = None
    sys.modules['src.torch_wrapper'] = _tw
import matplotlib; matplotlib.use('Agg')
from dataset_mapping import get_dataset_features
from src.setup_matrices import setup_matrices
from src.minmaxML import do_learning

name = sys.argv[1]; fracs = np.array([float(v) for v in sys.argv[2].split(',')])
taus = [float(v) for v in sys.argv[3].split(',')]; nseeds = int(sys.argv[4]); steps = int(sys.argv[5]); out = sys.argv[6]
path, label, groups, feats, cats, drop, _ = get_dataset_features(name)
X, y, gl, gnames, gtypes, _ = setup_matrices(path, label, groups, usable_features=feats, drop_group_as_feature=True,
                                             categorical_columns=cats, groups_to_drop=drop, verbose=False, save_data=False)
METHODS = {'nonstrategic': [False, False, False], 'ours': [True, True, False], 'naive': [False, False, True]}
N_GAMMAS = int(os.environ.get('N_GAMMAS', '0'))


def run(spec, tau, tau_vector, seed, gamma=0.0, relaxed=False):
    sl = [spec[0], spec[2]]; sa = spec[1]
    idx = sl[0] * 4 + sl[1] * 2 + sa
    me, ae, vme, vae = [0] * 6, [0] * 6, [0] * 6, [0] * 6
    ret = do_learning(X.copy(), y, steps, gl, 1, 0.5, model_type='PairedRegressionClassifier',
                group_names=gnames, group_types=gtypes, data_name=name, error_type='0/1 Loss', extra_error_types={}, pop_error_type='',
                convergence_threshold=1e-12, test_size=0.3, random_split_seed=seed, gamma=gamma, relaxed=relaxed,
                display_plots=False, save_plots=True, dirname=os.environ.get('PLOTDIR', '/tmp/regen_plots'), use_input_commands=False,
                strategic_learner=sl, strategic_agent=sa, tau=tau, tau_vector=tau_vector,
                learner_tau_min_frac=fracs.min(), learner_tau_max_frac=fracs.max(),
                learner_tau_mean=float(np.mean(tau_vector)), learner_tau_step=0.1, curr_idx=idx,
                max_error=me, avg_error=ae, val_max_error=vme, val_avg_error=vae)
    return ret, float(np.ravel(vme[idx])[-1]), float(np.ravel(vae[idx])[-1]), float(np.ravel(me[idx])[-1]), float(np.ravel(ae[idx])[-1])


for s in range(nseeds):
    seed = 4235255 + s + int(os.environ.get('SEED_OFFSET', '0'))
    for tau in taus:
        tau_vector = fracs[gl.flatten()] * tau
        if N_GAMMAS == 0:
            rec = dict(dataset=name, fracs=fracs.tolist(), tau=tau, seed=seed, T=steps)
            for mname, spec in METHODS.items():
                _, vmax, vavg, _, _ = run(spec, tau, tau_vector, seed)
                rec[mname] = vmax; rec[mname + '_avg'] = vavg
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            continue
        for mname, spec in METHODS.items():
            ret, vmax, vavg, tmax, tavg = run(spec, tau, tau_vector, seed)
            gamma_hat, max_err = float(ret[0]), float(ret[1])
            recs = [dict(dataset=name, fracs=fracs.tolist(), tau=tau, seed=seed, T=steps, method=mname,
                         gamma=None, gamma_hat=gamma_hat, val_max=vmax, val_pop=vavg, train_max=tmax, train_pop=tavg)]
            for gamma in np.linspace(gamma_hat, 0.5, N_GAMMAS):
                if gamma > max_err and os.environ.get('SKIP_LOOSE', '0') == '1':
                    continue  # main_driver.py skips these; the paper text runs the full range up to 0.5
                _, vmax, vavg, tmax, tavg = run(spec, tau, tau_vector, seed, gamma=float(gamma), relaxed=True)
                recs.append(dict(dataset=name, fracs=fracs.tolist(), tau=tau, seed=seed, T=steps, method=mname,
                                 gamma=float(gamma), gamma_hat=gamma_hat, val_max=vmax, val_pop=vavg, train_max=tmax, train_pop=tavg))
            with open(out, 'a') as f:
                for r in recs:
                    f.write(json.dumps(r) + '\n')
