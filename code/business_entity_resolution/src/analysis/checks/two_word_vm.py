"""VM job: two-word look-alikes. France (u4 accepted) vs v4 held-out accepted (train labels), same definition.
  /root/venv/bin/python two_word_vm.py DATA U4 V4_STAGE2 OUT"""
import collections, os, re, sys, zlib
import numpy as np, pandas as pd
import france_lookalike_filter as F
from france_rules import FR_FILLERS

DATA, U4, V4S2, OUT = sys.argv[1:5]
TRAIN_FILLERS = {'center', 'services', 'service', 'partners', '5ervices', '5ervice'}


def two_word(sn, sa, rn, ra, freq, fillers):
    ta, tb = set(F.toks(sn)), set(F.toks(rn))
    wa, wb = ta - tb, tb - ta
    if len(wa) != 2 or len(wb) != 2 or len(ta & tb) < 1:
        return False
    for b in wb:
        if b in fillers | F.KEEP_WORDS or freq[b] < F.MIN_FREQ:
            return False
        if any(F.SequenceMatcher(None, a, b).ratio() >= 0.6 or a[:3] == b[:3] or F.is_abbrev(a, b) for a in wa):
            return False
    ka, kb = F.addr_key(sa), F.addr_key(ra)
    return bool(ka[0]) and ka[0] == kb[0] and bool(ka[1] & kb[1])


os.makedirs(OUT, exist_ok=True)
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
# France
t1 = rd('test/test_source1.tsv'); t1 = t1[t1.country == 'France'].set_index('entity_id')
tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')]); tr = tr[tr.country == 'France'].set_index('entity_id')
ff = collections.Counter(t for n in t1.business_name.values for t in set(F.toks(n)))
u4 = pd.read_csv(U4, sep='\t', dtype=str, keep_default_na=False)
fp = [(a, x) for a, v in zip(u4.source1_entity_id, u4.matched_entity_ids) if a in t1.index for x in v.split(',') if x]
fh = [(a, x) for a, x in fp if two_word(t1.business_name[a], t1.business_address[a], tr.business_name[x], tr.business_address[x], ff, FR_FILLERS)]
print(f'FRANCE u4: accepted {len(fp):,} | two-word look-alikes {len(fh):,} ({len(fh) / len(fp):.4%})', flush=True)
fr = pd.DataFrame(fh, columns=['sid', 'rid'])
fr['s'] = [t1.business_name[a] for a in fr.sid]; fr['r'] = [tr.business_name[x] for x in fr.rid]
print(fr.sample(min(20, len(fr)), random_state=0)[['s', 'r']].to_string(), flush=True)
fr.to_parquet(f'{OUT}/fr_two_word.parquet')
# held-out (v4 accepted, labels)
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
p = pd.read_parquet(V4S2, columns=['q', 's', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
na = r.business_address.values[best.q.values] == ''
nat = np.array([bool(nonlat.search(a + ' ' + b)) for a, b in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
acc = best[best.p2a.values >= np.where(na, 0.40, np.where(nat, 0.525, 0.60))]
for c in ('US', 'India'):
    fq = collections.Counter(t for n in s1.business_name[s1.country == c] for t in set(F.toks(n)))
    a = acc[s1.country.values[acc.s.values] == c]
    hit = [two_word(s1.business_name.values[s], s1.business_address.values[s], r.business_name.values[q], r.business_address.values[q], fq, TRAIN_FILLERS)
           for q, s in zip(a.q.values, a.s.values)]
    h = a[np.array(hit)]
    print(f'HELD-OUT {c}: accepted {len(a):,} | two-word look-alikes {len(h):,} ({len(h) / len(a):.4%}) | truly same business {h.y.mean() if len(h) else float("nan"):.3f}', flush=True)
print('DONE', flush=True)
