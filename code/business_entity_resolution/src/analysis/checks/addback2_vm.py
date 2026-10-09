"""France add-back, calibrated on the LB: E's add-back (20,026 French matches, unmatched records with stage-1 p1 >= 0.7,
look-alike filters re-applied) gained +0.000335 (D2 0.98656 -> E 0.986895, only French rows differ).
1) implied true-rate of those adds; 2) split by France-adapted CE (v6.2b); 3) second tier: records still unmatched in
E with best p1 in [0.5, 0.7) (and CE agreement), look-alike filter applied; expected LB for each variant.
  python addback2_vm.py DATA E.tsv D2.tsv SCORED V62B_DIR OUT"""
import collections, glob, math, sys
import numpy as np, pandas as pd
import france_rules as R
import france_lookalike_filter as F

DATA, E, D2, SC, V62B, OUT = sys.argv[1:7]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/addback2.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    P, Rr = tp / n, tp / T; return 1.25 * P * Rr / (0.25 * P + Rr)
N = 1732544
def gain_add(ch, p):          # ch = [(kept, added)] ; kept assumed correct, each add true w.p. p
    return sum(math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(k + m, k + t, k + t) - f05(k, k, k + t)) for k, m in ch for t in range(m + 1)) / N
def gain_remove(ch, p):       # ch = [(kept_after, removed)] ; each removed pair is TRUE w.p. p
    return sum(math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(k, k, k + t) - f05(k + m, k + t, k + t)) for k, m in ch for t in range(m + 1)) / N

t1 = rd('test/test_source1.tsv'); tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')], ignore_index=True)
fr = set(t1.entity_id[t1.country == 'France'])
load = lambda f: {a: [x for x in v.split(',') if x] for a, v in zip(*pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False).T.values)}
e, d2 = load(E), load(D2)
adds = [(a, x) for a in fr for x in set(e[a]) - set(d2[a])]
say(f'E add-back: {len(adds):,} French matches added vs D2')
ch = [(len(d2[a]), len(set(e[a]) - set(d2[a]))) for a in fr if set(e[a]) != set(d2[a])]
for p in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6):
    say(f'   if {p:.0%} of adds true: expected {gain_add(ch, p):+.6f} (measured +0.000335)')
# CE for adds
si = pd.Series(np.arange(len(t1)), index=t1.entity_id.values); qi = pd.Series(np.arange(len(tr)), index=tr.entity_id.values)
ce62 = pd.concat([pd.read_parquet(f, columns=['q', 's', 'ce']) for f in glob.glob(f'{V62B}/*_test_ce.parquet')]).drop_duplicates(['q', 's']).set_index(['q', 's']).ce
A = pd.DataFrame(adds, columns=['sid', 'rid']); A['ce'] = ce62.reindex(list(zip(qi.reindex(A.rid).values, si.reindex(A.sid).values))).values
say(f'adds with v6.2b CE < 0: {int((A.ce < 0).sum()):,} | CE >= 0: {int((A.ce >= 0).sum()):,} | missing {int(A.ce.isna().sum()):,}')
neg = A[A.ce < 0]
chn = [(len(e[a]) - m, m) for a, m in neg.sid.value_counts().items()]
for p in (0.2, 0.4, 0.5, 0.6, 0.8):
    say(f'   remove CE<0 adds, if {p:.0%} of them are TRUE: {gain_remove(chn, p):+.6f}')
# second tier
sc = pd.read_parquet(SC, columns=['q', 's', 'p1'])
sc = sc[t1.country.values[sc.s.values] == 'France']
sc['sid'] = t1.entity_id.values[sc.s.values]; sc['rid'] = tr.entity_id.values[sc.q.values]
assigned = {x for a in fr for x in e[a]}
top = sc.sort_values('p1', ascending=False).drop_duplicates('rid')
free = top[~top.rid.isin(assigned)]
T1, TR = t1.set_index('entity_id'), tr.set_index('entity_id')
freq = collections.Counter(t for n in T1.business_name[T1.country == 'France'].values for t in set(F.toks(n)))
for lo, hi in ((0.6, 0.7), (0.5, 0.7)):
    c = free[(free.p1 >= lo) & (free.p1 < hi)].copy()
    c['ce'] = ce62.reindex(list(zip(c.q.values, c.s.values))).values
    c['lk'] = [R.classify(T1.business_name[a], T1.business_address[a], TR.business_name[x], TR.business_address[x], freq) is not None for a, x in zip(c.sid, c.rid)]
    for cet in (None, 0.0, 2.0):
        s = c[~c.lk] if cet is None else c[~c.lk & (c.ce > cet)]
        cap = s[s.sid.map(lambda a: len(e[a])) < 11]
        chs = [(len(e[a]), m) for a, m in cap.sid.value_counts().items()]
        say(f'tier2 p1 in [{lo},{hi}) not look-alike{"" if cet is None else f", v6.2b CE > {cet}"}: {len(cap):,} adds | '
            + ' | '.join(f'{p:.0%}: {gain_add(chs, p):+.6f}' for p in (0.8, 0.7, 0.6, 0.5)))
        if lo == 0.6 and cet == 0.0:
            cap.to_parquet(f'{OUT}/tier2_adds.parquet')
say('DONE')
