"""Aggregate analysis/results/deployed_*.jsonl into a table (means over seeds)."""
import json, glob, sys
from collections import defaultdict
import numpy as np
rows = defaultdict(list)
for f in sorted(glob.glob('analysis/results/deployed_*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        rows[(r['dataset'], tuple(r['fracs']), r['tau'])].append(r)
keys = ['mixture_test_max', 'draw_test_max_mean', 'draw_q05', 'draw_q95', 'draw_test_max_worst',
        'lp_support', 'lp_test_max', 'best_single_test_max']
fmt = sys.argv[1] if len(sys.argv) > 1 else 'text'
order = ['Heart', 'Credit', 'COMPAS', 'Communities']
out = []
for k in sorted(rows, key=lambda k: (order.index(k[0]), k[1], k[2])):
    rs = rows[k]
    v = {}
    for r in rs:
        r['draw_q05'], r['draw_q95'] = r['draw_test_max_q'][0], r['draw_test_max_q'][2]
    for key in keys:
        v[key] = np.mean([r[key] for r in rs])
    v['lp_support_max'] = max(r['lp_support'] for r in rs)
    v['n'] = len(rs)
    out.append((k, v))
if fmt == 'tex':
    for (d, f, t), v in out:
        fr = '(' + ','.join(f'{x:g}' for x in f) + ')'
        print(f"{d} & ${fr}$ & {t:g} & {v['mixture_test_max']:.3f} & {v['draw_test_max_mean']:.3f} & "
              f"[{v['draw_q05']:.3f}, {v['draw_q95']:.3f}] & {v['lp_support_max']} & {v['lp_test_max']:.3f} & "
              f"{v['best_single_test_max']:.3f} \\\\")
else:
    print(f"{'dataset':12s}{'fracs':20s}{'tau':>4s} {'mix':>6s} {'draw':>6s} {'q05':>6s} {'q95':>6s} {'worst':>6s} {'supp':>4s} {'lp':>6s} {'best1':>6s} n")
    for (d, f, t), v in out:
        print(f"{d:12s}{str(f):20s}{t:4g} {v['mixture_test_max']:6.3f} {v['draw_test_max_mean']:6.3f} {v['draw_q05']:6.3f} "
              f"{v['draw_q95']:6.3f} {v['draw_test_max_worst']:6.3f} {v['lp_support_max']:4d} {v['lp_test_max']:6.3f} {v['best_single_test_max']:6.3f} {v['n']}")
