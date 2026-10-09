"""Cross-encoder veto: remove ACCEPTED pairs the cross-encoder scores strongly negative.
Held-out (v6.1, train labels): precision of accepted pairs by CE bucket; cut-off t chosen on half A, paired per-S1 F0.5
change on half B (1000 bootstrap). France (file D2/B France): counts of accepted pairs with v6.2b and v6.1 CE below t.
  python ce_veto_vm.py DATA V61_STAGE2 V61_TRAIN_CE BEST.tsv V62B_DIR V61_TEST_CE OUT"""
import glob, re, sys, zlib
import numpy as np, pandas as pd

DATA, S2P, TRCE, BEST, V62B, V61CE, OUT = sys.argv[1:8]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/ce_veto.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
gt = rd('train/train_ground_truth.tsv'); truth = {a: set(v.split(',')) - {''} for a, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
bucket = lambda ids: np.fromiter((zlib.crc32(i.encode()) % 100 for i in ids), np.int16, len(ids))
H = s1[bucket(s1.entity_id) < 2].reset_index(drop=True)
half_a = bucket(np.array([i[::-1] for i in H.entity_id])) < 50
p = pd.read_parquet(S2P, columns=['q', 's', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
na = r.business_address.values[best.q.values] == ''
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
nat = np.array([bool(nonlat.search(x + ' ' + y)) for x, y in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
acc = best[best.p2a.values >= np.where(na, 0.50, np.where(nat, 0.50, 0.625))].copy()
ce = pd.read_parquet(TRCE); say('train CE columns', list(ce.columns), len(ce))
ce = ce.drop_duplicates(['q', 's']).set_index(['q', 's']).ce
acc['ce'] = ce.reindex(list(zip(acc.q.values, acc.s.values))).values
acc['sid'] = s1.entity_id.values[acc.s.values]; acc['rid'] = r.entity_id.values[acc.q.values]
acc['country'] = s1.country.values[acc.s.values]
say(f'held-out accepted pairs {len(acc):,} | CE available {np.isfinite(acc.ce).mean():.1%} | precision overall {acc.y.mean():.4f}')
acc['ceb'] = pd.cut(acc.ce, [-99, -6, -4, -2, 0, 2, 99])
say(acc.groupby(['ceb'], observed=True).y.agg(['size', 'mean']).round(4).to_string())
def f05(P, T):
    if not T: return float(not P)
    tp = len(P & T)
    if not tp: return 0.0
    pr, rc = tp / len(P), tp / len(T); return 1.25 * pr * rc / (0.25 * pr + rc)
def per(sel, ids):
    d = sel.groupby('sid').rid.apply(set).to_dict(); return np.array([f05(d.get(i, set()), truth[i]) for i in ids])
ida, idb = H.entity_id.values[half_a], H.entity_id.values[~half_a]
base_a, base_b = per(acc, ida), per(acc, idb)
res = {}
for t in (-8, -6, -4, -2, 0):
    keep = acc[~(acc.ce < t)]
    res[t] = per(keep, ida).mean() - base_a.mean()
    say(f'  half A: veto CE < {t}: removes {int((acc.ce < t).sum()):,} | change {res[t]:+.5f}')
tb = max(res, key=res.get)
keep = acc[~(acc.ce < tb)]; nb = per(keep, idb); d = nb - base_b
rng = np.random.default_rng(0); idx = [rng.integers(0, len(d), len(d)) for _ in range(1000)]
ci = np.percentile([d[i].mean() for i in idx], [2.5, 97.5])
say(f'CHOSEN on half A: CE < {tb} -> half B paired {d.mean():+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]')
cb = H.country.values[~half_a]
for c in ('US', 'India'):
    m = cb == c; dc = d[m]; i2 = [rng.integers(0, m.sum(), m.sum()) for _ in range(1000)]
    cc = np.percentile([dc[i].mean() for i in i2], [2.5, 97.5]); say(f'   {c}: {dc.mean():+.5f} [{cc[0]:+.5f}, {cc[1]:+.5f}]')
# France
t1 = rd('test/test_source1.tsv', usecols=['entity_id', 'country'])
tr = pd.concat([rd('test/test_source2.tsv', usecols=['entity_id']), rd('test/test_source3.tsv', usecols=['entity_id'])], ignore_index=True)
si = pd.Series(np.arange(len(t1)), index=t1.entity_id.values); qi = pd.Series(np.arange(len(tr)), index=tr.entity_id.values)
fr = set(t1.entity_id[t1.country == 'France'])
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
fp = pd.DataFrame([(a, x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids) if a in fr for x in v.split(',') if x], columns=['sid', 'rid'])
k = list(zip(qi.reindex(fp.rid).values, si.reindex(fp.sid).values))
c62 = pd.concat([pd.read_parquet(f, columns=['q', 's', 'ce']) for f in glob.glob(f'{V62B}/*_test_ce.parquet')]).drop_duplicates(['q', 's']).set_index(['q', 's']).ce.reindex(k).values
c61 = pd.read_parquet(V61CE, columns=['q', 's', 'ce']).drop_duplicates(['q', 's']).set_index(['q', 's']).ce.reindex(k).values
say(f'FRANCE accepted {len(fp):,} | v6.2b CE covered {np.isfinite(c62).mean():.1%} | v6.1 CE covered {np.isfinite(c61).mean():.1%}')
for t in (-8, -6, -4, -2, 0):
    say(f'  France: v6.1 CE < {t}: {int(np.nansum(c61 < t)):,} | v6.2b CE < {t}: {int(np.nansum(c62 < t)):,} | BOTH < {t}: {int(np.nansum((c61 < t) & (c62 < t))):,}')
fp.assign(c62=c62, c61=c61).to_parquet(f'{OUT}/fr_ce_accepted.parquet')
say('DONE')
