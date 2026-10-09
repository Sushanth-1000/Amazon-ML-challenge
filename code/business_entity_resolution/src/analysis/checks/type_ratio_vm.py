"""Which kinds of ACCEPTED French matches are over-represented vs held-out US/India, and how accurate that kind is on
held-out (train labels)?  Same method that found the look-alike rule (France ~40x held-out, LB-confirmed ~91% wrong).
  python type_ratio_vm.py DATA B.tsv V61_STAGE2 OUT"""
import re, sys
from difflib import SequenceMatcher
import numpy as np, pandas as pd
import france_lookalike_filter as F

DATA, BEST, S2P, OUT = sys.argv[1:5]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
NUM = re.compile(r'^0*(\d+)[a-z]*$')


def num(addr):
    t = re.sub(r'[^a-z0-9 ]', ' ', F.fold(addr)).split()
    x = next((x for x in t if x[:1].isdigit()), '')
    m = NUM.match(x)
    return m.group(1) if m else x


def close(a, b):
    return SequenceMatcher(None, a, b).ratio() >= 0.6 or a[:3] == b[:3] or F.is_abbrev(a, b)


def ptype(sn, sa, rn, ra):
    ta, tb = set(F.toks(sn)), set(F.toks(rn))
    wa, wb = ta - tb, tb - ta
    if not wa and not wb:
        nm = 'exact'
    elif not (ta & tb):
        nm = 'no shared word'
    elif all(any(close(a, b) for b in wb) for a in wa) and all(any(close(a, b) for a in wa) for b in wb):
        nm = 'typo only'
    elif not wa:
        nm = 'record adds word(s)'
    elif not wb:
        nm = 'record drops word(s)'
    elif len(wa) == 1 and len(wb) == 1:
        nm = 'one word swapped'
    else:
        nm = 'several words differ'
    if ra == '':
        ad = 'no address'
    else:
        na, nb = num(sa), num(ra)
        street = bool(F.addr_key(sa)[1] & F.addr_key(ra)[1])
        ad = ('no number' if not (na and nb) else 'same number' if na == nb else 'diff number') + (' +street' if street else ' -street')
    return nm, ad


out = open(f'{OUT}/type_ratio.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()

# France accepted (file B)
t1 = rd('test/test_source1.tsv'); t1 = t1[t1.country == 'France'].set_index('entity_id')
tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')]); tr = tr[tr.country == 'France'].set_index('entity_id')
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
fp = [(a, x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids) if a in t1.index for x in v.split(',') if x]
f = pd.DataFrame([ptype(t1.business_name[a], t1.business_address[a], tr.business_name[x], tr.business_address[x]) for a, x in fp], columns=['name', 'addr'])
f[['sid', 'rid']] = pd.DataFrame(fp, index=f.index)
f.to_parquet(f'{OUT}/fr_types.parquet')
# held-out accepted (v6.1 pass-1 decision) + labels
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
p = pd.read_parquet(S2P, columns=['q', 's', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
na_ = r.business_address.values[best.q.values] == ''
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
nat = np.array([bool(nonlat.search(x + ' ' + y)) for x, y in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
acc = best[best.p2a.values >= np.where(na_, 0.50, np.where(nat, 0.50, 0.625))]
sn, sa, rn, ra = s1.business_name.values, s1.business_address.values, r.business_name.values, r.business_address.values
h = pd.DataFrame([ptype(sn[s], sa[s], rn[q], ra[q]) for q, s in zip(acc.q.values, acc.s.values)], columns=['name', 'addr'])
h['y'] = acc.y.values
say(f'accepted pairs: France (file B) {len(f):,} | held-out US+India (v6.1) {len(h):,} (held-out precision {h.y.mean():.4f})')
ft = f.groupby(['name', 'addr']).size().rename('fr_n')
ht = h.groupby(['name', 'addr']).y.agg(['size', 'mean']).rename(columns={'size': 'ho_n', 'mean': 'ho_precision'})
t = pd.concat([ft, ht], axis=1).fillna(0)
t['fr_share'] = t.fr_n / len(f); t['ho_share'] = t.ho_n / len(h)
t['ratio'] = t.fr_share / t.ho_share.replace(0, np.nan)
t['fr_excess_pairs'] = (t.fr_share - t.ho_share).clip(lower=0) * len(f)
pd.set_option('display.width', 220)
say(t.sort_values('fr_excess_pairs', ascending=False).round({'fr_share': 4, 'ho_share': 4, 'ratio': 2, 'ho_precision': 4, 'fr_excess_pairs': 0}).head(30).to_string())
say('DONE')
