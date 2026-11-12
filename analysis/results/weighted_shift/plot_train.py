import json, os, sys, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]
CFG = [('Communities', (1, 1, 1, 1), 1.0), ('Credit', (1, 0.5, 0, 0), 3.0), ('COMPAS', (0.25, 1, 1, 0.25), 0.5), ('Heart', (0.5, 1), 0.5)]
M = [('nonstrategic', 'Non-strategic', '#7a7a7a'), ('naive', 'Naive strategic', '#d1495b'), ('ours', 'Ours', '#2e6fdb')]
def load(v):
    d = collections.defaultdict(list); p = f'{R}/{v}.jsonl'
    if not os.path.exists(p): return d
    rows = [json.loads(l) for l in open(p)]
    # index gammas within (config, method, seed) in file order: 0 = unrelaxed, 1.. = relaxed
    cnt = collections.Counter()
    for r in rows:
        key = (r['dataset'], tuple(float(x) for x in r['fracs']), r['tau'], r['method'], r['seed'])
        k = cnt[key]; cnt[key] += 1
        d[key[:4] + (k,)].append(r)
    return d
def pareto(pts):
    pts = sorted(pts); front = []; best = np.inf
    for x, y in pts:
        if y < best: front.append((x, y)); best = y
    return np.array(front)
D = {v: load(v) for v in ('old', 'new')}
fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
summary = []
for j, (d, f, tau) in enumerate(CFG):
    for i, (v, lab) in enumerate([('old', 'Old shift rule (max group)'), ('new', 'Weighted shift (WERM)')]):
        ax = axes[i, j]
        for m, ml, c in M:
            pts = []
            for k in range(0, 50):
                rs = D[v].get((d, tuple(float(x) for x in f), tau, m, k))
                if not rs: continue
                pts.append((np.mean([r['train_pop'] for r in rs]), np.mean([r['train_max'] for r in rs])))
            if not pts: continue
            pts = np.array(pts); fr = pareto([tuple(p) for p in pts])
            ax.scatter(pts[:, 0], pts[:, 1], s=10, color=c, alpha=0.35)
            ax.step(fr[:, 0], fr[:, 1], where='post', color=c, lw=2, label=ml)
            summary.append((d, v, m, fr[:, 1].min(), fr[:, 0].min(), len(pts), min(len(rs) for rs in [D[v][(d, tuple(float(x) for x in f), tau, m, 0)]])))
        fs = 'equal' if len(set(f)) == 1 else '(' + ','.join(f'{x:g}' for x in f) + ')'
        ax.set_title(f'{d}, τ={tau:g}, f={fs}\n{lab}', fontsize=10); ax.grid(alpha=0.3)
        ax.set_xlabel('Population train error'); 
        if j == 0: ax.set_ylabel('Max group train error')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Algorithm 2 trade-off curves on TRAINING data (Pareto frontier of per-γ averages over seeds): fixed code, old vs weighted shift rule')
fig.tight_layout(); fig.savefig(f'{OUT}/pareto_shiftrule_compare_train.png', dpi=150)
print(f"{'dataset':12s}{'ver':5s}{'method':14s}{'min max-grp':>12s}{'min pop':>9s}{'#pts':>6s}{'seeds':>6s}")
for s in summary: print(f"{s[0]:12s}{s[1]:5s}{s[2]:14s}{s[3]:12.3f}{s[4]:9.3f}{s[5]:6d}{s[6]:6d}")
