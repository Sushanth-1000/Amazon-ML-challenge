"""Identical names, different house numbers: what separates a TRUE copy from a decoy (held-out labels), and how many
rejected French pairs (record unassigned) carry the true-copy signature.
  python numsig_vm.py DATA B.tsv V61CAND.tsv V61_STAGE2 OUT"""
import re, sys
from difflib import SequenceMatcher
import numpy as np, pandas as pd
import france_lookalike_filter as F

DATA, BEST, CAND, S2P, OUT = sys.argv[1:6]
rd = lambda f, **k: pd.read_csv(f'{DATA}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
NUM = re.compile(r'^0*(\d+)[a-z]*$')
out = open(f'{OUT}/numsig.txt', 'w')
def say(*a):
    s = ' '.join(map(str, a)); print(s, flush=True); out.write(s + '\n'); out.flush()


def parts(addr):
    t = re.sub(r'[^a-z0-9 ]', ' ', F.fold(addr)).split()
    i = next((i for i, x in enumerate(t) if x[:1].isdigit()), None)
    if i is None:
        return '', set()
    m = NUM.match(t[i]); n = m.group(1) if m else t[i]
    words = {x for x in t if x.isalpha() and len(x) > 2}
    return n, words


def lev1(a, b):
    """True if a and b differ by exactly one digit substituted, inserted or deleted."""
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if abs(len(a) - len(b)) != 1:
        return False
    lo, sh = (a, b) if len(a) > len(b) else (b, a)
    return any(lo[:i] + lo[i + 1:] == sh for i in range(len(lo)))


def sig(ta, sa, tb, ra):
    if ra == '' or ta != tb:
        return None
    (na, wa), (nb, wb) = parts(sa), parts(ra)
    if not na or not nb or na == nb or not na.isdigit() or not nb.isdigit():
        return None
    jac = len(wa & wb) / max(1, len(wa | wb))
    street = 'street>=0.6' if jac >= 0.6 else ('street 0.3-0.6' if jac >= 0.3 else 'street<0.3')
    d = abs(int(na) - int(nb))
    rel = ('prefix' if (na.startswith(nb) or nb.startswith(na)) else 'one digit edit' if lev1(na, nb) else
           'shift<=20' if d <= 20 else 'other')
    return rel, street

# ---- held-out US/India (labels)
s1 = rd('train/train_source1.tsv'); r = pd.concat([rd('train/train_source2.tsv'), rd('train/train_source3.tsv')], ignore_index=True)
p = pd.read_parquet(S2P, columns=['q', 's', 'p2a', 'y', 'split']); p = p[p.split == 'H']
best = p.sort_values('p2a', ascending=False).drop_duplicates('q')
na_ = r.business_address.values[best.q.values] == ''
nonlat = re.compile(r'[^\x00-ɏ\s\W\d]')
nat = np.array([bool(nonlat.search(x + ' ' + y)) for x, y in zip(r.business_name.values[best.q.values], r.business_address.values[best.q.values])])
accq = best[best.p2a.values >= np.where(na_, 0.50, np.where(nat, 0.50, 0.625))]
accset = set(zip(accq.q.values, accq.s.values)); assigned = set(accq.q.values)
sn, sa, rn, ra = s1.business_name.values, s1.business_address.values, r.business_name.values, r.business_address.values
TS = {i: frozenset(F.toks(sn[i])) for i in np.unique(p.s.values)}; TQ = {i: frozenset(F.toks(rn[i])) for i in np.unique(p.q.values)}
rows = [(q, s, *g) for q, s in zip(p.q.values, p.s.values) if TS[s] == TQ[q] and (g := sig(TS[s], sa[s], TQ[q], ra[q]))]
h = pd.DataFrame(rows, columns=['q', 's', 'rel', 'street']).merge(p[['q', 's', 'y']], on=['q', 's'])
h['acc'] = [(q, s) in accset for q, s in zip(h.q, h.s)]; h['free'] = ~h.q.isin(assigned)
say('HELD-OUT US/India, identical names, different house numbers (all v6.1 candidates):')
say(h.groupby(['rel', 'street']).agg(n=('y', 'size'), truly_same=('y', 'mean'), accepted=('acc', 'mean'),
                                      rejected_free_truly_same=('y', lambda s: h.loc[s.index][~h.loc[s.index].acc & h.loc[s.index].free].y.mean())).round(3).to_string())
# ---- France
t1 = rd('test/test_source1.tsv'); t1 = t1[t1.country == 'France'].set_index('entity_id')
tr = pd.concat([rd('test/test_source2.tsv'), rd('test/test_source3.tsv')]); tr = tr[tr.country == 'France'].set_index('entity_id')
b = pd.read_csv(BEST, sep='\t', dtype=str, keep_default_na=False)
acc = set((a, x) for a, v in zip(b.source1_entity_id, b.matched_entity_ids) if a in t1.index for x in v.split(',') if x)
owner = {x for _, x in acc}
cand = pd.read_csv(CAND, sep='\t', dtype=str, keep_default_na=False); cand = cand[cand.source1_entity_id.isin(t1.index)]
TS = {i: frozenset(F.toks(n)) for i, n in t1.business_name.items()}; TR = {i: frozenset(F.toks(n)) for i, n in tr.business_name.items()}
SA, RA = t1.business_address.to_dict(), tr.business_address.to_dict()
fr = []
for a, v in zip(cand.source1_entity_id, cand.candidate_entity_ids):
    for x in v.split(','):
        if x and TS[a] == TR[x] and (g := sig(TS[a], SA[a], TR[x], RA[x])):
            fr.append((a, x, *g, (a, x) in acc, x not in owner))
f = pd.DataFrame(fr, columns=['sid', 'rid', 'rel', 'street', 'acc', 'free'])
f.to_parquet(f'{OUT}/fr_numsig.parquet')
say('\nFRANCE, identical names, different house numbers (v6.1 candidates, file B):')
say(f.groupby(['rel', 'street']).agg(n=('acc', 'size'), accepted=('acc', 'sum'), rejected_record_free=('free', lambda s: int((s & ~f.loc[s.index].acc).sum()))).to_string())
say('DONE')
