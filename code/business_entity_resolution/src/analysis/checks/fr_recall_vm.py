"""France recall check (VM, full data). Pairs whose business names are identical (after cleaning) and whose house numbers
match only after normalising the number (leading zeros, letter/bis/ter suffix), same street word, record not assigned.
  France: v6.1 candidates vs file B (0.986638) -> how many such pairs are rejected.
  Held-out US/India (train labels): the same pair type among v6.1 held-out candidates -> share truly the same business,
  split by accepted / rejected by v6.1.
  python fr_recall_vm.py DATA B.tsv V61CAND.tsv V61_STAGE2 OUT"""
import re, sys
import numpy as np, pandas as pd
import france_lookalike_filter as F

DATA, BEST, CAND, S2P, OUT = sys.argv[1:6]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
NUM = re.compile(r'^0*(\d+)[a-z]*$')


def numkey(addr):
    t = re.sub(r'[^a-z0-9 ]', ' ', F.fold(addr)).split()
    for i, x in enumerate(t):
        if x[:1].isdigit():
            m = NUM.match(x)
            return (x, m.group(1) if m else x)
    return ('', '')


def kind(sn, sa, rn, ra):
    if ra == '' or frozenset(F.toks(sn)) != frozenset(F.toks(rn)):
        return None
    (a_raw, a_n), (b_raw, b_n) = numkey(sa), numkey(ra)
    if not a_n or not b_n or not (F.addr_key(sa)[1] & F.addr_key(ra)[1]):
        return None
    if a_raw == b_raw:
        return 'same number (raw)'
    return 'same number after normalising' if a_n == b_n else 'different number'


out = open(f'{OUT}/fr_recall.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()

# ---- France (test)
t1 = rd('test/test_source1.tsv'); t1 = t1[t1.country == 'France'].set_index('entity_id')
tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')]); tr = tr[tr.country == 'France'].set_index('entity_id')
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
acc = set((a, x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids) if a in t1.index for x in v.split(',') if x)
owner = {x for _, x in acc}
cand = pd.read_csv(CAND, sep='\t', dtype=str, keep_default_na=False)
cand = cand[cand.source1_entity_id.isin(t1.index)]
rows = []
for a, v in zip(cand.source1_entity_id, cand.candidate_entity_ids):
    for x in v.split(','):
        if not x:
            continue
        k = kind(t1.business_name[a], t1.business_address[a], tr.business_name[x], tr.business_address[x])
        if k:
            rows.append((a, x, k, (a, x) in acc, x in owner))
f = pd.DataFrame(rows, columns=['sid', 'rid', 'kind', 'acc', 'rec_assigned']); f['rej_free'] = ~f.acc & ~f.rec_assigned
say('FRANCE (v6.1 candidates, file B): identical names, shared street word')
say(f.groupby('kind').agg(pairs=('acc', 'size'), accepted=('acc', 'sum'), rejected_record_unassigned=('rej_free', 'sum')).to_string())
m = f[(f.kind == 'same number after normalising') & ~f.acc & ~f.rec_assigned]
m.to_parquet(f'{OUT}/fr_norm_candidates.parquet')
say(m.head(15).assign(s=lambda d: [t1.business_address[a][:45] for a in d.sid], r=lambda d: [tr.business_address[x][:45] for x in d.rid])[['s', 'r']].to_string())

# ---- held-out US/India (train labels), v6.1
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
p = pd.read_parquet(S2P, columns=['q', 's', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
na = r.business_address.values[best.q.values] == ''
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
nat = np.array([bool(nonlat.search(x + ' ' + y)) for x, y in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
accq = best[best.p2a.values >= np.where(na, 0.50, np.where(nat, 0.50, 0.625))]      # v6.1 pass-1 decision (stage2.log)
accset = set(zip(accq.q.values, accq.s.values)); assigned = set(accq.q.values)
sn, sa, rn, ra = s1.business_name.values, s1.business_address.values, r.business_name.values, r.business_address.values
p['kind'] = [kind(sn[s], sa[s], rn[q], ra[q]) for q, s in zip(p.q.values, p.s.values)]
h = p[p.kind.notna()].copy()
h['acc'] = [(q, s) in accset for q, s in zip(h.q, h.s)]; h['rec_assigned'] = h.q.isin(assigned)
h['country'] = s1.country.values[h.s.values]
say('\nHELD-OUT US/India (v6.1 candidates, v6.1 decision), identical names, shared street word:')
say(h.groupby(['kind', 'acc']).y.agg(['size', 'mean']).rename(columns={'mean': 'share truly same'}).round(4).to_string())
g = h[~h.acc & ~h.rec_assigned]
say('rejected with record unassigned:\n' + g.groupby(['kind', 'country']).y.agg(['size', 'mean']).round(4).to_string())
say('DONE')
