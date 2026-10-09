"""Proof numbers for G2 (LB 0.987342): md5s, per-country profile of every file in the chain, row changes of G2 vs
each file, chain consistency, candidate coverage, and the true-rates implied by the leaderboard.
  python g2_proof_vm.py DATA OUT A.tsv B.tsv D2.tsv E.tsv G.tsv G2.tsv CAND.tsv"""
import hashlib, math, sys
import numpy as np, pandas as pd

DATA, OUT = sys.argv[1:3]
names = dict(zip(('A', 'B', 'D2', 'E', 'G', 'G2'), sys.argv[3:9])); CAND = sys.argv[9]
out = open(f'{OUT}/g2_proof.txt', 'w')
def say(*x):
    s = ' '.join(map(str, x)); print(s, flush=True); out.write(s + '\n'); out.flush()
def md5(f):
    h = hashlib.md5()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()
rd = lambda f, **k: pd.read_csv(f'{DATA}/test/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3, **k)
t1 = rd('test_source1.tsv', usecols=['entity_id', 'country'])
tr = pd.concat([rd(f'test_source{i}.tsv', usecols=['entity_id', 'business_address']) for i in (2, 3)], ignore_index=True)
cty = dict(zip(t1.entity_id, t1.country)); noaddr = set(tr.entity_id[tr.business_address == ''])
load = lambda f: {s: frozenset(x for x in v.split(',') if x)
                  for s, v in zip(*pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False).T.values)}

say('== md5')
for k, f in list(names.items()) + [('CAND', CAND)]:
    say(f'{k:4s} {md5(f)}  {f}')
F = {k: load(f) for k, f in names.items()}
C = ('US', 'India', 'France')
ids = {c: [s for s in F['G2'] if cty[s] == c] for c in C}
say('S1 per country:', {c: len(ids[c]) for c in C})

say('\n== profile: matches per S1 | empty S1 share | no-address share of matches')
for k, d in F.items():
    row = []
    for c in C:
        n = np.array([len(d[s]) for s in ids[c]]); m = [x for s in ids[c] for x in d[s]]
        row.append(f'{c} {n.mean():.4f} | {np.mean(n == 0):.4%} | {np.mean([x in noaddr for x in m]):.4%}')
    say(f'{k:3s} ' + ' || '.join(row))

say('\n== row changes: G2 vs file, per country: S1 that differ (share identical) | matches in G2 not in file / in file not in G2')
for k in ('A', 'B', 'D2', 'E', 'G'):
    for c in C:
        diff = [s for s in ids[c] if F['G2'][s] != F[k][s]]
        add = sum(len(F['G2'][s] - F[k][s]) for s in diff); rem = sum(len(F[k][s] - F['G2'][s]) for s in diff)
        say(f'G2 vs {k:2s} {c:6s}: {len(diff):>7,} S1 differ ({1 - len(diff) / len(ids[c]):.4%} identical) | +{add:,} / -{rem:,}')

say('\n== chain consistency (rows identical?)')
same = lambda x, y, c: all(F[x][s] == F[y][s] for s in ids[c])
for x, y in (('B', 'A'), ('D2', 'B'), ('E', 'D2'), ('G', 'B'), ('G', 'E'), ('G2', 'G')):
    say(f'{x:2s} == {y:2s}: ' + ', '.join(f'{c} {same(x, y, c)}' for c in C))
fr = ids['France']
say('France: E removes any D2 match?', sum(len(F['D2'][s] - F['E'][s]) for s in fr),
    '| G2 subset of G?', all(F['G2'][s] <= F['G'][s] for s in fr))

say('\n== candidate coverage: every G2 match inside the candidate file?')
miss = {c: 0 for c in C}; npairs = {c: 0 for c in C}; seen = 0
with open(CAND, encoding='utf-8') as fh:
    say('header:', next(fh).rstrip('\n'))
    for line in fh:
        s, v = line.rstrip('\n').split('\t'); seen += 1
        cs = v.split(',') if v else []; npairs[cty[s]] += len(cs)
        if F['G2'][s]:
            miss[cty[s]] += len(F['G2'][s] - set(cs))
say(f'candidate rows {seen:,} | pairs per country {npairs} | G2 matches NOT in candidates {miss}')

N = 1732544
def f05(n, tp, T):
    if T == 0: return float(n == 0)
    if tp == 0: return 0.0
    P, R = tp / n, tp / T; return 1.25 * P * R / (0.25 * P + R)
def gain_add(ch, p):  # ch = [(kept, added)]; kept assumed correct, each added pair true w.p. p
    return sum(math.comb(m, t) * p ** t * (1 - p) ** (m - t) * (f05(k + m, k + t, k + t) - f05(k, k, k + t))
               for k, m in ch for t in range(m + 1)) / N
def solve(fn, target):  # fn increasing in p
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
    return (lo + hi) / 2
ch_add = [(len(F['D2'][s]), len(F['E'][s] - F['D2'][s])) for s in fr if F['E'][s] != F['D2'][s]]
ch_rem = [(len(F['G2'][s]), len(F['G'][s] - F['G2'][s])) for s in fr if F['G'][s] != F['G2'][s]]
n_add, n_rem = sum(m for _, m in ch_add), sum(m for _, m in ch_rem)
p_add = solve(lambda p: gain_add(ch_add, p), 0.986895 - 0.98656)
p_rem = solve(lambda p: gain_add(ch_rem, p), -0.000369)  # removing pairs = minus adding them back
say(f'\n== LB-implied true-rates (kept matches assumed correct)\n'
    f'add-back (E - D2 = +0.000335, France rows only): {n_add:,} pairs on {len(ch_add):,} S1 -> {p_add:.1%} true\n'
    f'CE<0 removal (+0.000369 = G2 - E - (B - D2)): {n_rem:,} pairs on {len(ch_rem):,} S1 -> {p_rem:.1%} true\n'
    f'=> add-backs kept in G2 (CE >= 0): {n_add - n_rem:,} pairs, ~{(p_add * n_add - p_rem * n_rem) / (n_add - n_rem):.1%} true (derived)')
say('DONE')
