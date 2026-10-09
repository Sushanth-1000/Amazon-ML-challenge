"""Expected LB change of adding X1/X2 pairs to G4, from G4's real per-S1 structure: each add is true with prob p
(exact binomial per S1, existing G4 matches assumed correct), N = all 1,732,544 S1."""
import collections, math, sys
import pandas as pd
G4, *adds = sys.argv[1:]
g = pd.read_csv(G4, sep='\t', dtype=str, keep_default_na=False)
k = {a: len([x for x in v.split(',') if x]) for a, v in zip(g.iloc[:, 0], g.iloc[:, 1])}
N = len(k)
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    P, R = tp / n, tp / T; return 1.25 * P * R / (0.25 * P + R)
def gain(ch, p):
    return sum(c * math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(kk + m, kk + t, kk + t) - f05(kk, kk, kk + t))
               for (kk, m), c in ch.items() for t in range(m + 1)) / N
for f in adds:
    a = pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)
    ch = collections.Counter((k[s], m) for s, m in a.s1_id.value_counts().items())
    emp = sum(c for (kk, m), c in ch.items() if kk == 0)
    print(f'{f}: {len(a):,} adds to {a.s1_id.nunique():,} S1 ({emp:,} empty in G4) | ' +
          ' | '.join(f'{p:.0%} true: {gain(ch, p):+.5f}' for p in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5)), flush=True)
