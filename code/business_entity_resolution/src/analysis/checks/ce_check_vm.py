"""France-aware second opinion on the house-number add-backs: v6.2b (France-adapted) and v6.1 cross-encoder scores for
(a) the add-back pairs, (b) accepted French pairs of the same shape (identical name, same number), (c) the removed R1
look-alikes (LB-confirmed ~91% wrong). Then build a CE-filtered add-back file.
  python ce_check_vm.py DATA B.tsv fr_numsig.parquet V62B_DIR V61_CE LOOKALIKE_PARQUET OUT"""
import glob, math, sys
import numpy as np, pandas as pd

DATA, BEST, SIG, V62B, V61CE, LK, OUT = sys.argv[1:8]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
out = open(f'{OUT}/ce_check.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()
t1 = rd('test/test_source1.tsv', usecols=['entity_id'])
tr = pd.concat([rd('test/test_source2.tsv', usecols=['entity_id']), rd('test/test_source3.tsv', usecols=['entity_id'])], ignore_index=True)
def ids(d):
    return d.assign(sid=t1.entity_id.values[d.s.values], rid=tr.entity_id.values[d.q.values])
b62 = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{V62B}/*_test_ce.parquet')]); say('v6.2b CE columns', list(b62.columns), len(b62))
b62 = ids(b62).drop_duplicates(['sid', 'rid']).set_index(['sid', 'rid'])
b61 = ids(pd.read_parquet(V61CE)).drop_duplicates(['sid', 'rid']).set_index(['sid', 'rid']); say('v6.1 CE columns', list(b61.columns), len(b61))
col = 'ce' if 'ce' in b62.columns else b62.columns[0]
f = pd.read_parquet(SIG)
adds = f[f.free & ~f.acc & f.rel.isin(['prefix', 'other']) & f.street.isin(['street>=0.6', 'street 0.3-0.6'])]
adds = adds[adds.groupby('rid').sid.transform('size') == 1]
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
acc = pd.DataFrame([(a, x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids) for x in v.split(',') if x], columns=['sid', 'rid'])
lk = pd.read_parquet(LK)[['sid', 'rid']]
def stats(name, d):
    k = list(zip(d.sid, d.rid))
    c62 = b62[col].reindex(k).values; c61 = b61[col].reindex(k).values if col in b61 else b61.iloc[:, 0].reindex(k).values
    say(f'{name:45s} n {len(d):>8,} | v6.2b CE covered {np.isfinite(c62).mean():.1%}, median {np.nanmedian(c62):+.2f}, >0 {np.nanmean(c62 > 0):.1%} '
        f'| v6.1 CE covered {np.isfinite(c61).mean():.1%}, median {np.nanmedian(c61):+.2f}, >0 {np.nanmean(c61 > 0):.1%}')
    return c62, c61
fracc = acc[acc.sid.isin(set(f.sid)) | True]
fr_ids = set(pd.read_csv(f'{DATA}/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, quoting=3).query("country=='France'").entity_id)
fracc = acc[acc.sid.isin(fr_ids)]
stats('(b) accepted French pairs (all)', fracc)
same_num = f[f.acc]; stats('(b2) accepted French, identical name diff number', same_num)
stats('(c) removed R1 look-alikes (~91% wrong on LB)', lk)
c62, c61 = stats('(a) house-number add-backs (tier 2)', adds)
adds = adds.assign(ce62=c62)
for thr in (0.0, 2.0):
    keep = adds[adds.ce62 > thr]
    say(f'  add-backs with v6.2b CE > {thr}: {len(keep):,} ({keep.street.value_counts().to_dict()})')
    cur = {a: [x for x in v.split(',') if x] for a, v in zip(b.source1_entity_id, b.matched_entity_ids)}
    add = keep.groupby('sid').rid.apply(list).to_dict()
    with open(f'{OUT}/u8_ce{int(thr)}.tsv', 'w', encoding='utf-8', newline='') as fh:
        fh.write('source1_entity_id\tmatched_entity_ids\n'); fh.writelines(f'{s}\t{",".join(v + add.get(s, []))}\n' for s, v in cur.items())
    N = len(cur); ch = [(len(cur[s]), len(xs)) for s, xs in add.items()]
    def f05(n, tp, T):
        if T == 0: return float(n == 0)
        if tp == 0: return 0.0
        P, R = tp / n, tp / T; return 1.25 * P * R / (0.25 * P + R)
    for p in (0.95, 0.9, 0.8, 0.7):
        g = sum(math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(k + m, k + t, k + t) - f05(k, k, k + t)) for k, m in ch for t in range(m + 1))
        say(f'     if {p:.0%} true: {g / N:+.6f} LB')
say('DONE')
