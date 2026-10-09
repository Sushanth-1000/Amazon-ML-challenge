"""Second France add-back on top of G2 (LB 0.987342), using the France-adapted CE (v6.2b) as the gate.
LB calibration so far: add-back p1>=0.7 -> ~78% true overall; of those, v6.2b CE<0 -> ~37% true, CE>=0 -> ~95% true.
Tiers (records unmatched in G2, their best stage-1 candidate, not an R1 look-alike, cap 11 per S1):
  T2: p1 in [0.5, 0.7) and v6.2b CE > 0        T3: p1 in [0.3, 0.5) and v6.2b CE > 2
  python addback3_vm.py DATA G2.tsv SCORED V62B_DIR OUT"""
import collections, glob, math, sys
import numpy as np, pandas as pd
import france_rules as R
import france_lookalike_filter as F

DATA, G2, SC, V62B, OUT = sys.argv[1:6]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/addback3.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    P, Rr = tp / n, tp / T; return 1.25 * P * Rr / (0.25 * P + Rr)
N = 1732544
def gain_add(ch, p):
    return sum(math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(k + m, k + t, k + t) - f05(k, k, k + t)) for k, m in ch for t in range(m + 1)) / N
t1 = rd('test/test_source1.tsv'); tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')], ignore_index=True)
fr = set(t1.entity_id[t1.country == 'France'])
g = {a: [x for x in v.split(',') if x] for a, v in zip(*pd.read_csv(G2, sep='\t', dtype=str, keep_default_na=False).T.values)}
assigned = {x for a in fr for x in g[a]}
sc = pd.read_parquet(SC, columns=['q', 's', 'p1']); sc = sc[t1.country.values[sc.s.values] == 'France']
sc['sid'] = t1.entity_id.values[sc.s.values]; sc['rid'] = tr.entity_id.values[sc.q.values]
top = sc.sort_values('p1', ascending=False).drop_duplicates('rid')
free = top[~top.rid.isin(assigned)].copy()
ce = pd.concat([pd.read_parquet(f, columns=['q', 's', 'ce']) for f in glob.glob(f'{V62B}/*_test_ce.parquet')]).drop_duplicates(['q', 's']).set_index(['q', 's']).ce
free['ce'] = ce.reindex(list(zip(free.q.values, free.s.values))).values
T1, TR = t1.set_index('entity_id'), tr.set_index('entity_id')
freq = collections.Counter(t for n in T1.business_name[T1.country == 'France'].values for t in set(F.toks(n)))
say(f'French records unmatched in G2 with a stage-1 candidate: {len(free):,} | with v6.2b CE: {free.ce.notna().sum():,}')
tiers = {'T2': ((0.5, 0.7), 0.0), 'T3': ((0.3, 0.5), 2.0)}
chosen = {}
for name, ((lo, hi), cet) in tiers.items():
    c = free[(free.p1 >= lo) & (free.p1 < hi) & (free.ce > cet)].copy()
    c = c[[R.classify(T1.business_name[a], T1.business_address[a], TR.business_name[x], TR.business_address[x], freq) is None for a, x in zip(c.sid, c.rid)]]
    c = c[c.sid.map(lambda a: len(g[a])) < 11]
    chosen[name] = c
    chs = [(len(g[a]), m) for a, m in c.sid.value_counts().items()]
    emp = sum(1 for a in c.sid.unique() if len(g[a]) == 0)
    say(f'{name}: p1 in [{lo},{hi}), v6.2b CE > {cet}: {len(c):,} adds to {c.sid.nunique():,} S1 ({emp:,} empty) | '
        + ' | '.join(f'{p:.0%} true: {gain_add(chs, p):+.6f}' for p in (0.95, 0.9, 0.8, 0.7, 0.6)))
for label, parts in (('G3', ['T2']), ('G4', ['T2', 'T3'])):
    add = pd.concat([chosen[k] for k in parts]).groupby('sid').rid.apply(list).to_dict()
    with open(f'{OUT}/{label}.tsv', 'w', encoding='utf-8', newline='') as fh:
        fh.write('source1_entity_id\tmatched_entity_ids\n')
        fh.writelines(f'{a}\t{",".join(v + add.get(a, [])[:max(0, 11 - len(v))])}\n' for a, v in g.items())
say('DONE')
