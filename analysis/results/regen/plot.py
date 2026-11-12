import json, sys, collections, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
data = collections.defaultdict(list)
for v in ('old', 'new'):
    p = f'{S}/{v}.jsonl'
    if not os.path.exists(p): continue
    for l in open(p):
        r = json.loads(l); data[(r['dataset'], tuple(r['fracs']), v, r['tau'])].append(r)
cfgs = []
for l in open(f'{S}/jobs.txt'):
    d, f = l.split(); cfgs.append((d, tuple(float(x) for x in f.split(','))))
M = [('nonstrategic', 'Non-strategic', '#7a7a7a'), ('naive', 'Naive strategic', '#d1495b'), ('ours', 'Ours', '#2e6fdb')]
def series(d, f, v, m):
    taus = sorted(t for (dd, ff, vv, t) in data if (dd, ff, vv) == (d, f, v))
    mu = [np.mean([r[m] for r in data[(d, f, v, t)]]) for t in taus]
    se = [1.96 * np.std([r[m] for r in data[(d, f, v, t)]]) / np.sqrt(len(data[(d, f, v, t)])) for t in taus]
    n = [len(data[(d, f, v, t)]) for t in taus]
    return np.array(taus), np.array(mu), np.array(se), n
def fig(cfg_list, fname, title):
    k = len(cfg_list); fig, axes = plt.subplots(2, k, figsize=(4.2 * k, 7), sharey='col', squeeze=False)
    for j, (d, f) in enumerate(cfg_list):
        for i, (v, lab) in enumerate([('old', 'Old code (rounding issue)'), ('new', 'Fixed code')]):
            ax = axes[i, j]
            for m, ml, c in M:
                t, mu, se, n = series(d, f, v, m)
                if len(t) == 0: continue
                ax.plot(t, mu, '-o', ms=3, color=c, label=ml); ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.15)
            fs = 'equal' if len(set(f)) == 1 else '(' + ','.join(f'{x:g}' for x in f) + ')'
            ax.set_title(f'{d}, f={fs}\n{lab}', fontsize=10); ax.set_xlabel('τ'); ax.grid(alpha=0.3)
            if j == 0: ax.set_ylabel('Max group test error')
    axes[0, 0].legend(fontsize=8, loc='best'); fig.suptitle(title); fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}', dpi=150); plt.close(fig)
fig(cfgs[:4], 'equal_budget_old_vs_fixed.png', 'Equal budgets: old vs fixed best-response scoring')
fig(cfgs[4:7], 'heart_unequal_old_vs_fixed.png', 'Heart, unequal budgets')
fig(cfgs[7:10], 'credit_unequal_old_vs_fixed.png', 'Credit, unequal budgets')
fig(cfgs[10:13], 'compas_unequal_old_vs_fixed.png', 'COMPAS, unequal budgets')
# summary: mean over tau>0 of (naive - ours), old vs new
print(f"{'config':28s} {'old naive-ours':>15s} {'new naive-ours':>15s} {'old nonstrat-ours':>18s} {'new nonstrat-ours':>18s} seeds")
for d, f in cfgs:
    row = []
    for v in ('old', 'new'):
        _, a, _, n = series(d, f, v, 'naive'); t, b, _, _ = series(d, f, v, 'ours'); _, c, _, _ = series(d, f, v, 'nonstrategic')
        if len(t) == 0: row += [np.nan, np.nan, 0]; continue
        msk = t > 0; row += [np.mean(a[msk] - b[msk]), np.mean(c[msk] - b[msk]), min(n)]
    print(f"{d+' '+str(f):28s} {row[0]:15.3f} {row[3]:15.3f} {row[1]:18.3f} {row[4]:18.3f} {row[2]}/{row[5]}")
