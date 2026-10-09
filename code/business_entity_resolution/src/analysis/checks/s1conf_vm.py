"""Stage-1-confident pairs that the final decision rejected, record left unassigned.
France: v6.2a stage-1 p1 (test_scored_v62a) vs file B.  Held-out US/India: v6.1 p1 vs v6.1 decision, with labels.
  python s1conf_vm.py DATA SCORED_V62A B.tsv V61_STAGE2 OUT"""
import re, sys
import numpy as np, pandas as pd

DATA, SC, BEST, S2P, OUT = sys.argv[1:6]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/s1conf.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
TH = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)

# ---- France
t1 = rd('test/test_source1.tsv', usecols=['entity_id', 'country'])
tr = pd.concat([rd('test/test_source2.tsv', usecols=['entity_id', 'business_address']), rd('test/test_source3.tsv', usecols=['entity_id', 'business_address'])], ignore_index=True)
sc = pd.read_parquet(SC, columns=['q', 's', 'p1'])
fr = t1.country.values[sc.s.values] == 'France'
sc = sc[fr].copy()
sc['sid'] = t1.entity_id.values[sc.s.values]; sc['rid'] = tr.entity_id.values[sc.q.values]
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
acc = set((a, x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids) for x in v.split(',') if x)
assigned = {x for _, x in acc}
sc['acc'] = [(a, x) in acc for a, x in zip(sc.sid, sc.rid)]
chk = sc[sc.acc]; say(f'France: B pairs found in v6.2a scored pairs {len(chk):,} of {len(acc):,} B pairs (all countries); their p1 median {chk.p1.median():.3f}')
top = sc.sort_values('p1', ascending=False).drop_duplicates('rid')
top['noaddr'] = tr.business_address.values[top.q.values] == ''
free = top[~top.rid.isin(assigned)]
n_acc = int(sc.acc.sum())
for t in TH:
    m = free[free.p1 >= t]
    say(f'  France: record unassigned, top p1 >= {t}: {len(m):,} pairs ({len(m) / n_acc:.3%} of accepted) | no-address {m.noaddr.mean():.2%}')
free.to_parquet(f'{OUT}/fr_s1conf_free.parquet')

# ---- held-out (v6.1)
s1 = rd('train/train_source1.tsv', usecols=['entity_id']); r = pd.concat([rd('train/train_source2.tsv', usecols=['business_name', 'business_address']), rd('train/train_source3.tsv', usecols=['business_name', 'business_address'])], ignore_index=True)
p = pd.read_parquet(S2P, columns=['q', 's', 'p1', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
na = r.business_address.values[best.q.values] == ''
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
nat = np.array([bool(nonlat.search(x + ' ' + y)) for x, y in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
accq = best[best.p2a.values >= np.where(na, 0.50, np.where(nat, 0.50, 0.625))]
assigned_h = set(accq.q.values)
top_h = p.sort_values('p1', ascending=False).drop_duplicates('q')
free_h = top_h[~top_h.q.isin(assigned_h)]
for t in TH:
    m = free_h[free_h.p1 >= t]
    say(f'  HELD-OUT: record unassigned, top p1 >= {t}: {len(m):,} pairs ({len(m) / len(accq):.3%} of accepted) | truly same business {m.y.mean() if len(m) else float("nan"):.3f}')
say('DONE')
