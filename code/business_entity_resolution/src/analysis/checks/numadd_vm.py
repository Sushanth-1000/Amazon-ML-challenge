"""Build France add-back files from the house-number signature and compute their exact expected LB change.
Adds: identical names, different house number, relation prefix/other (NOT small shift, NOT one-digit edit), record
unassigned in B, and the record has exactly ONE such candidate S1 (unambiguous).
  tier1: street jaccard >= 0.6  (held-out: prefix 94% / other 86% truly same)
  tier2: tier1 + street 0.3-0.6 (held-out: prefix 94% / other 78%)
  python numadd_vm.py B.tsv fr_numsig.parquet OUT"""
import math, sys
import pandas as pd

BEST, SIG, OUT = sys.argv[1:4]
f = pd.read_parquet(SIG)
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
cur = {a: [x for x in v.split(',') if x] for a, v in zip(b.source1_entity_id, b.matched_entity_ids)}
N = len(cur)
out = open(f'{OUT}/numadd.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    P, R = tp / n, tp / T; return 1.25 * P * R / (0.25 * P + R)

base = f[f.free & ~f.acc & f.rel.isin(['prefix', 'other'])]
for tier, streets in (('tier1', ['street>=0.6']), ('tier2', ['street>=0.6', 'street 0.3-0.6'])):
    c = base[base.street.isin(streets)]
    one = c.groupby('rid').sid.transform('size') == 1           # unambiguous records only
    c = c[one]
    add = c.groupby('sid').rid.apply(list).to_dict()
    ch = [(len(cur[s]), len(xs)) for s, xs in add.items()]
    empty = sum(k == 0 for k, _ in ch)
    say(f'{tier}: add {len(c):,} matches to {len(add):,} French S1 ({empty:,} currently EMPTY) | by relation {c.rel.value_counts().to_dict()}')
    for p in (0.94, 0.86, 0.78, 0.6, 0.4):
        g = 0.0
        for k, m in ch:                       # k kept (assumed correct), m added; each added true w.p. p
            for t in range(m + 1):
                w = math.comb(m, t) * p ** t * (1 - p) ** (m - t)
                T = k + t
                g += w * (f05(k + m, k + t, T) - f05(k, k, T))
        say(f'    if {p:.0%} of added are true: {g / N:+.6f} LB')
    rows = []
    for s, v in cur.items():
        rows.append((s, ','.join(v + add.get(s, []))))
    with open(f'{OUT}/u8_{tier}.tsv', 'w', encoding='utf-8', newline='') as fh:
        fh.write('source1_entity_id\tmatched_entity_ids\n'); fh.writelines(f'{s}\t{m}\n' for s, m in rows)
say('DONE')
