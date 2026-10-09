"""Type breakdown of stage-1-confident rejected pairs (record unassigned): France vs held-out (with truth per type).
Also: is the pair's S1 empty in the final file (adding a true match there turns F 0 -> 1)?
  python s1conf_types_vm.py DATA SCORED_V62A B.tsv V61_STAGE2 FREE_PARQUET OUT"""
import re, sys
import numpy as np, pandas as pd
from ptypes import ptype

DATA, SC, BEST, S2P, FREE, OUT = sys.argv[1:7]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/s1conf_types.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
T = 0.9
t1 = rd('test/test_source1.tsv').set_index('entity_id'); tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')]).set_index('entity_id')
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
nmatch = dict(zip(b.source1_entity_id, b.matched_entity_ids.map(lambda v: len([x for x in v.split(',') if x]))))
f = pd.read_parquet(FREE); f = f[f.p1 >= T].copy()
f[['name', 'addr']] = pd.DataFrame([ptype(t1.business_name[a], t1.business_address[a], tr.business_name[x], tr.business_address[x]) for a, x in zip(f.sid, f.rid)], index=f.index)
f['s1_empty'] = f.sid.map(nmatch).eq(0)
f.to_parquet(f'{OUT}/fr_s1conf_typed.parquet')
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
p = pd.read_parquet(S2P, columns=['q', 's', 'p1', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
na = r.business_address.values[best.q.values] == ''
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
nat = np.array([bool(nonlat.search(x + ' ' + y)) for x, y in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
accq = best[best.p2a.values >= np.where(na, 0.50, np.where(nat, 0.50, 0.625))]
top = p.sort_values('p1', ascending=False).drop_duplicates('q')
h = top[~top.q.isin(set(accq.q.values)) & (top.p1 >= T)].copy()
sn, sa, rn, ra = s1.business_name.values, s1.business_address.values, r.business_name.values, r.business_address.values
h[['name', 'addr']] = pd.DataFrame([ptype(sn[s], sa[s], rn[q], ra[q]) for q, s in zip(h.q.values, h.s.values)], index=h.index)
ft = f.groupby(['name', 'addr']).agg(fr_n=('rid', 'size'), fr_s1_empty=('s1_empty', 'mean'))
ht = h.groupby(['name', 'addr']).y.agg(['size', 'mean']).rename(columns={'size': 'ho_n', 'mean': 'ho_truly_same'})
t = pd.concat([ft, ht], axis=1).fillna(0).sort_values('fr_n', ascending=False)
pd.set_option('display.width', 220)
say(f'stage-1 p1 >= {T}, final decision rejected, record unassigned | France {len(f):,} (S1 empty in B: {f.s1_empty.mean():.1%}) | held-out {len(h):,} (truly same {h.y.mean():.3f})')
say(t.round(3).head(25).to_string())
say('DONE')
