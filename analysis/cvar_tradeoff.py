"""
Diversified / capped-weight regulator experiment (after Balcan, Blum & Chen, AAMAS 2018).
For each alpha in CAP_ALPHAS (env, comma list; 'none' = original uncapped minimax), runs ours and the naive
strategic baseline with group weights capped at p_g / alpha, and records test (and train) max-group and
population error. alpha = 1 is plain error minimization, alpha <= min_g p_g recovers minimax.
Usage: python analysis/cvar_tradeoff.py DATASET f1,f2,.. taus(comma) n_seeds numsteps out.jsonl
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
import src.minmaxML as mm

name = sys.argv[1]; fracs = np.array([float(v) for v in sys.argv[2].split(',')])
taus = [float(v) for v in sys.argv[3].split(',')]; nseeds = int(sys.argv[4]); steps = int(sys.argv[5]); out = sys.argv[6]
alphas = os.environ.get('CAP_ALPHAS', 'none,1,0.75,0.5,0.3,0.2,0.1').split(',')
path, label, groups, feats, cats, drop, _ = get_dataset_features(name)
X, y, gl, gnames, gtypes, _ = setup_matrices(path, label, groups, usable_features=feats, drop_group_as_feature=True,
                                             categorical_columns=cats, groups_to_drop=drop, verbose=False, save_data=False)
METHODS = {'ours': [True, True, False], 'naive': [False, False, True]}


def run(spec, tau, tau_vector, seed):
    sl = [spec[0], spec[2]]; sa = spec[1]
    idx = sl[0] * 4 + sl[1] * 2 + sa
    me, ae, vme, vae = [0] * 6, [0] * 6, [0] * 6, [0] * 6
    mm.do_learning(X.copy(), y, steps, gl, 1, 0.5, model_type='PairedRegressionClassifier',
                   group_names=gnames, group_types=gtypes, data_name=name, error_type='0/1 Loss', extra_error_types={}, pop_error_type='',
                   convergence_threshold=1e-12, test_size=0.3, random_split_seed=seed, gamma=0.0, relaxed=False,
                   display_plots=False, save_plots=True, dirname=os.environ.get('PLOTDIR', '/tmp/cvar_plots'), use_input_commands=False,
                   strategic_learner=sl, strategic_agent=sa, tau=tau, tau_vector=tau_vector,
                   learner_tau_min_frac=fracs.min(), learner_tau_max_frac=fracs.max(),
                   learner_tau_mean=float(np.mean(tau_vector)), learner_tau_step=0.1, curr_idx=idx,
                   max_error=me, avg_error=ae, val_max_error=vme, val_avg_error=vae)
    return [float(np.ravel(v[idx])[-1]) for v in (vme, vae, me, ae)]


for s in range(nseeds):
    seed = 4235255 + s + int(os.environ.get('SEED_OFFSET', '0'))
    for tau in taus:
        tau_vector = fracs[gl.flatten()] * tau
        for a in alphas:
            mm.GROUP_CAP_ALPHA = None if a == 'none' else float(a)
            for mname, spec in METHODS.items():
                vmax, vpop, tmax, tpop = run(spec, tau, tau_vector, seed)
                with open(out, 'a') as f:
                    f.write(json.dumps(dict(dataset=name, fracs=fracs.tolist(), tau=tau, seed=seed, T=steps, alpha=a,
                                            method=mname, val_max=vmax, val_pop=vpop, train_max=tmax, train_pop=tpop)) + '\n')
