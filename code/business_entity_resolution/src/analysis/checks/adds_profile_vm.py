"""What did the 0.986895 file add in France (vs B)? Pair types, no-address share, cross-encoder opinions, stage-1 p1,
and how many MORE pairs of the same kind are still unassigned (the extension candidates).
  python adds_profile_vm.py DATA NEW.tsv B.tsv V62B_DIR V61_CE SCORED_V62A OUT"""
import glob, sys
import numpy as np, pandas as pd
from ptypes import ptype

DATA, NEW, BEST, V62B, V61CE, SC, OUT = sys.argv[1:8]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/adds_profile.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
t1 = rd('test/test_source1.tsv'); tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')], ignore_index=True)
fr = set(t1.entity_id[t1.country == 'France'])
load = lambda f: {(a, x) for a, v in zip(*pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False).T.values) if a in fr for x in v.split(',') if x}
new, old = load(NEW), load(BEST)
adds = pd.DataFrame(sorted(new - old), columns=['sid', 'rid'])
say(f'France added {len(adds):,} | removed {len(old - new):,}')
T1, TR = t1.set_index('entity_id'), tr.set_index('entity_id')
adds[['name', 'addr']] = pd.DataFrame([ptype(T1.business_name[a], T1.business_address[a], TR.business_name[x], TR.business_address[x]) for a, x in zip(adds.sid, adds.rid)])
say(adds.groupby(['name', 'addr']).size().sort_values(ascending=False).head(15).to_string())
si = pd.Series(np.arange(len(t1)), index=t1.entity_id.values); qi = pd.Series(np.arange(len(tr)), index=tr.entity_id.values)
adds['s'] = si.reindex(adds.sid).values; adds['q'] = qi.reindex(adds.rid).values
def ce(path_or_df):
    d = path_or_df.drop_duplicates(['q', 's']).set_index(['q', 's']).ce
    return d.reindex(list(zip(adds.q, adds.s))).values
c62 = ce(pd.concat([pd.read_parquet(f, columns=['q', 's', 'ce']) for f in glob.glob(f'{V62B}/*_test_ce.parquet')]))
c61 = ce(pd.read_parquet(V61CE, columns=['q', 's', 'ce']))
p1 = pd.read_parquet(SC, columns=['q', 's', 'p1']); p1 = p1[p1.s.isin(set(adds.s))].drop_duplicates(['q', 's']).set_index(['q', 's']).p1
adds['p1'] = p1.reindex(list(zip(adds.q, adds.s))).values
say(f'v6.2b CE covered {np.isfinite(c62).mean():.1%} median {np.nanmedian(c62):+.2f} >0 {np.nanmean(c62 > 0):.1%} | '
    f'v6.1 CE covered {np.isfinite(c61).mean():.1%} median {np.nanmedian(c61):+.2f} >0 {np.nanmean(c61 > 0):.1%}')
say('stage-1 p1 (v6.2a) of added pairs: quantiles ' + str(np.round(np.nanquantile(adds.p1, [0.05, 0.25, 0.5, 0.75, 0.95]), 3)))
import collections
bcnt = collections.Counter(a for a, _ in old)
emp = [a for a in adds.sid.unique() if bcnt[a] == 0]
say(f'French S1 receiving adds: {adds.sid.nunique():,} | of these EMPTY in B: {len(emp):,} | adds per S1: {len(adds) / max(1, adds.sid.nunique()):.2f}')
adds.to_parquet(f'{OUT}/fr_adds.parquet')
say('DONE')
