"""Stricter no-address threshold on US/India (G4 keeps v6.1's 0.50): remove G4 matches of no-address records whose best
stage-2 probability (x_p2a, v6.1 pass 1) is in [0.50, t). Expected LB change from G4's per-S1 structure, each removed
pair TRUE w.p. q (exact binomial), N = 1,732,544. Writes G6_na<t>.tsv.
  python noaddr_thr_vm.py G4.tsv FRAME OUTDIR"""
import collections, math, sys
import numpy as np, pandas as pd
G4, FRAME, OUT = sys.argv[1:4]
g = pd.read_csv(G4, sep='\t', dtype=str, keep_default_na=False)
cols = list(g.columns)
base = {a: [x for x in v.split(',') if x] for a, v in zip(g.iloc[:, 0], g.iloc[:, 1])}
owner = {x: a for a, v in base.items() for x in v}
f = pd.read_parquet(FRAME, columns=['record_id', 's1_id', 'slice', 'x_p2a'])
f = f[f.slice == 'noaddr']
best = f.sort_values('x_p2a', ascending=False).drop_duplicates('record_id')
best = best[best.record_id.map(owner).notna()]
best['g4_s1'] = best.record_id.map(owner)
print(f'no-address US/India records matched in G4: {len(best):,} | G4 S1 = argmax-p2a S1: {np.mean(best.g4_s1 == best.s1_id):.1%}')
print('their best p2a:', np.round(np.quantile(best.x_p2a, [0.01, 0.05, 0.1, 0.25, 0.5]), 3), '| share < 0.6/0.7/0.8/0.9:',
      [f'{np.mean(best.x_p2a < t):.1%}' for t in (0.6, 0.7, 0.8, 0.9)])
N = len(base)
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    P, R = tp / n, tp / T; return 1.25 * P * R / (0.25 * P + R)
def gain_remove(ch, p):
    return sum(c * math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(kk, kk, kk + t) - f05(kk + m, kk + t, kk + t))
               for (kk, m), c in ch.items() for t in range(m + 1)) / N
for t in (0.6, 0.7, 0.8, 0.9):
    s = best[best.x_p2a < t]
    ch = collections.Counter((len(base[a]) - m, m) for a, m in s.g4_s1.value_counts().items())
    emp = sum(c for (kk, m), c in ch.items() if kk == 0)
    print(f'noaddr threshold 0.50 -> {t}: remove {len(s):,} matches from {s.g4_s1.nunique():,} S1 ({emp:,} become empty) | if TRUE share = '
          + ' | '.join(f'{q:.0%}: {gain_remove(ch, q):+.5f}' for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)), flush=True)
    rm = set(s.record_id)
    with open(f'{OUT}/G6_na{int(t * 100)}.tsv', 'w', encoding='utf-8', newline='') as fh:
        fh.write(f'{cols[0]}\t{cols[1]}\n'); fh.writelines(f'{a}\t{",".join(x for x in v if x not in rm)}\n' for a, v in base.items())
print('DONE')
