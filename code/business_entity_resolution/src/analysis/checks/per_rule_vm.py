"""VM: per-rule held-out check (train labels). For each rule: how many v4-accepted held-out pairs it flags, how many of
those are truly the same business, and the paired per-S1 F0.5 change if ONLY that rule is applied.
  /root/venv/bin/python per_rule_vm.py DATA V4S2 OUT"""
import collections, re, sys, zlib
import numpy as np, pandas as pd
import france_lookalike_filter as F
import france_rules as R

DATA, V4S2, OUT = sys.argv[1:4]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
gt = rd('train/train_ground_truth.tsv'); truth = {a: set(v.split(',')) - {''} for a, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
H = s1[np.fromiter((zlib.crc32(i.encode()) % 100 for i in s1.entity_id), np.int16, len(s1)) < 2].reset_index(drop=True)
p = pd.read_parquet(V4S2, columns=['q', 's', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
na = r.business_address.values[best.q.values] == ''
nat = np.array([bool(nonlat.search(a + ' ' + b)) for a, b in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
acc = best[best.p2a.values >= np.where(na, 0.40, np.where(nat, 0.525, 0.60))].copy()
R.FR_FILLERS = {'center', 'services', 'service', 'partners', '5ervices', '5ervice'}
freq = {c: collections.Counter(t for n in s1.business_name[s1.country == c] for t in set(F.toks(n))) for c in ('US', 'India')}
sn, sa, sc = s1.business_name.values, s1.business_address.values, s1.country.values
rn, ra = r.business_name.values, r.business_address.values
acc['rule'] = [R.classify(sn[s], sa[s], rn[q], ra[q], freq[sc[s]]) for q, s in zip(acc.q.values, acc.s.values)]
acc['sid'] = s1.entity_id.values[acc.s.values]; acc['rid'] = r.entity_id.values[acc.q.values]
def f05(P, T):
    if not T: return float(not P)
    tp = len(P & T)
    if not tp: return 0.0
    pr, rc = tp / len(P), tp / len(T); return 1.25 * pr * rc / (0.25 * pr + rc)
def per(sel):
    d = sel.groupby('sid').rid.apply(set).to_dict(); return np.array([f05(d.get(i, set()), truth[i]) for i in H.entity_id])
b = per(acc); rng = np.random.default_rng(0); idx = [rng.integers(0, len(b), len(b)) for _ in range(1000)]
out = open(f'{OUT}/per_rule.txt', 'w')
for rule in ('R1', 'R2', 'R3'):
    f = acc[acc.rule == rule]; n = per(acc[acc.rule != rule]); d = n - b
    ci = np.percentile([d[i].mean() for i in idx], [2.5, 97.5])
    s = (f'{rule}: flags {len(f):,} of {len(acc):,} held-out accepted pairs ({len(f) / len(acc):.4%}) | truly same business '
         f'{f.y.mean() if len(f) else float("nan"):.3f} | paired per-S1 F0.5 change {d.mean():+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]')
    print(s, flush=True); out.write(s + '\n')
out.close()
