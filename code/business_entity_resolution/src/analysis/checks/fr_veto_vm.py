"""France veto: remove ACCEPTED French pairs of G4 (LB 0.98743) that the France-adapted CE (v6.2b) scores very negative.
Calibration: (1) train pairs scored by v6.2b -> odds of a true match per CE band, relative to the CE<0 band;
(2) LB: E's French add-backs with v6.2b CE<0 were ~37% true when all add-backs were ~78% true (G2 vs E) -> LR(CE<0)=0.166.
Accepted-pair truth per band = prior odds x 0.166 x rel_band (prior = precision of accepted French pairs, assumed).
  python fr_veto_vm.py DATA G4.tsv V62B_TEST_DIR TRAIN_CE_DIR V61_TEST_CE OUT"""
import collections, glob, math, sys
import numpy as np, pandas as pd

DATA, G4, V62B, TRCE, V61CE, OUT = sys.argv[1:7]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
def say(*a): print(' '.join(map(str, a)), flush=True)
BANDS = [-99, -8, -6, -4, -2, 0, 2, 99]
# ---- (1) train calibration of v6.2b CE
s1 = rd('train/train_source1.tsv', usecols=['entity_id']).entity_id.values
r = pd.concat([rd('train/train_source2.tsv', usecols=['entity_id']), rd('train/train_source3.tsv', usecols=['entity_id'])]).entity_id.values
gt = rd('train/train_ground_truth.tsv'); r2s = {m: a for a, v in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in v.split(',') if m}
for f in sorted(glob.glob(f'{TRCE}/*.parquet')):
    t = pd.read_parquet(f); say(f, list(t.columns), len(t))
    if not {'q', 's', 'ce'} <= set(t.columns): continue
    t = t.drop_duplicates(['q', 's'])
    y = np.fromiter((r2s.get(a) == b for a, b in zip(r[t.q.values], s1[t.s.values])), bool, len(t))
    t['y'] = y; t['b'] = pd.cut(t.ce, BANDS)
    tab = t.groupby('b', observed=True).y.agg(['size', 'mean'])
    o = lambda p: p / max(1e-9, 1 - p)
    neg = o(t.y[t.ce < 0].mean())
    tab['rel_odds_vs_CEneg'] = [o(p) / neg for p in tab['mean']]
    say(tab.round(4).to_string()); rel = tab['rel_odds_vs_CEneg'].to_dict(); break
# ---- (2) G4 accepted French pairs
t1 = rd('test/test_source1.tsv', usecols=['entity_id', 'country'])
tr = pd.concat([rd('test/test_source2.tsv', usecols=['entity_id']), rd('test/test_source3.tsv', usecols=['entity_id'])], ignore_index=True)
si = pd.Series(np.arange(len(t1)), index=t1.entity_id.values); qi = pd.Series(np.arange(len(tr)), index=tr.entity_id.values)
fr = set(t1.entity_id[t1.country == 'France'])
g = {a: [x for x in v.split(',') if x] for a, v in zip(*pd.read_csv(G4, sep='\t', dtype=str, keep_default_na=False).T.values)}
P = pd.DataFrame([(a, x) for a in fr for x in g[a]], columns=['sid', 'rid'])
k = list(zip(qi.reindex(P.rid).values, si.reindex(P.sid).values))
c62 = pd.concat([pd.read_parquet(f, columns=['q', 's', 'ce']) for f in glob.glob(f'{V62B}/*_test_ce.parquet')]).drop_duplicates(['q', 's']).set_index(['q', 's']).ce
P['ce'] = c62.reindex(k).values
P['ce61'] = pd.read_parquet(V61CE, columns=['q', 's', 'ce']).drop_duplicates(['q', 's']).set_index(['q', 's']).ce.reindex(k).values
P['k'] = P.sid.map(lambda a: len(g[a]))
say(f'G4 French accepted pairs {len(P):,} | v6.2b CE covered {P.ce.notna().mean():.1%} | v6.1 CE covered {P.ce61.notna().mean():.1%}')
say(pd.cut(P.ce, BANDS).value_counts().sort_index().to_string())
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    Pp, R = tp / n, tp / T; return 1.25 * Pp * R / (0.25 * Pp + R)
N = len(g)
def gain_remove(ch, p):   # ch = {(kept_after, removed): count}; each removed pair TRUE w.p. p
    return sum(c * math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(kk, kk, kk + t) - f05(kk + m, kk + t, kk + t))
               for (kk, m), c in ch.items() for t in range(m + 1)) / N
for name, sel in [(f'v6.2b CE < {t}', P.ce < t) for t in (-8, -6, -4, -2)] + \
                 [(f'v6.2b CE < {t} and v6.1 CE < {t}', (P.ce < t) & (P.ce61 < t)) for t in (-6, -4, -2)]:
    s = P[sel]; ch = collections.Counter((len(g[a]) - m, m) for a, m in s.sid.value_counts().items())
    only = sum(c for (kk, m), c in ch.items() if kk == 0)
    say(f'{name:34s}: {len(s):>7,} pairs, {s.sid.nunique():>7,} S1 ({only:,} would become empty) | if TRUE share = ' +
        ' | '.join(f'{p:.0%}: {gain_remove(ch, p):+.5f}' for p in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)))
P.to_parquet(f'{OUT}/fr_acc_ce.parquet')
say('DONE')
