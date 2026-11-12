import json, os, sys, collections, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]
d = collections.defaultdict(list)
old = {}
if os.path.exists(f'{S}/sb_old.jsonl'):
    for l in open(f'{S}/sb_old.jsonl'):
        r = json.loads(l); old[(r['dataset'], tuple(r['fracs']), r['tau'], r['seed'])] = r['ours_old_sb']
for l in open(f'{S}/sb.jsonl'):
    r = json.loads(l)
    r['ours_old_sb'] = old.get((r['dataset'], tuple(r['fracs']), r['tau'], r['seed']), np.nan)
    d[(r['dataset'], tuple(r['fracs']))].append(r)
cfgs = [(l.split()[0], tuple(float(x) for x in l.split()[1].split(','))) for l in open(f'{S}/jobs.txt')]
SER = [('nonstrategic', 'Non-strategic', '#7a7a7a', '-'), ('naive', 'Naive strategic', '#d1495b', '-'),
       ('ours_old', 'Ours, old code', '#9bb8ec', '--'), ('ours_fixed', 'Ours, fixed', '#2e6fdb', '-'),
       ('ours_werm', 'Ours, fixed + weighted shift', '#173f86', ':')]
def ser(rs, key):
    taus = sorted(set(r['tau'] for r in rs))
    return np.array(taus), np.array([np.mean([r[key] for r in rs if r['tau'] == t]) for t in taus])
groups = [('Equal budgets', cfgs[:4]), ('Heart, unequal budgets', cfgs[4:7]), ('Credit, unequal budgets', cfgs[7:10]), ('COMPAS, unequal budgets', cfgs[10:13])]
fig, axes = plt.subplots(4, 4, figsize=(17, 16))
for i, (title, cl) in enumerate(groups):
    for j in range(4):
        ax = axes[i, j]
        if j >= len(cl): ax.axis('off'); continue
        dname, f = cl[j]; rs = d[(dname, f)]
        for key, lab, c, ls in SER:
            t, v = ser(rs, key + '_sb'); ax.plot(t, v, ls, marker='o', ms=3, color=c, label=lab, lw=2 if key == 'ours_fixed' else 1.5)
        fs = 'equal' if len(set(f)) == 1 else '(' + ','.join(f'{x:g}' for x in f) + ')'
        ax.set_title(f'{dname}, f={fs}', fontsize=10); ax.set_xlabel('τ'); ax.grid(alpha=0.3)
        if j == 0: ax.set_ylabel('Max group social burden (test)')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Max group social burden: non-strategic and naive are unchanged by the fixes; ours under old code, fixed code, and fixed + weighted shift')
fig.tight_layout(); fig.savefig(f'{OUT}/social_burden_compare.png', dpi=130)
print(f"{'config':30s} {'naive':>7s} {'old':>7s} {'fixed':>7s} {'werm':>7s} {'nonst':>7s}  (mean over tau>0)")
for dname, f in cfgs:
    rs = [r for r in d[(dname, f)] if r['tau'] > 0]
    m = lambda k: np.mean([r[k + '_sb'] for r in rs])
    print(f"{dname+' '+str(f):30s} {m('naive'):7.3f} {m('ours_old'):7.3f} {m('ours_fixed'):7.3f} {m('ours_werm'):7.3f} {m('nonstrategic'):7.3f}")
