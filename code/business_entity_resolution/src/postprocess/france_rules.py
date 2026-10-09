"""All French precision fixes in one pass, applied DIRECTLY to the accepted French matches of any matching_results.tsv.

Rule applied (learned on TRAIN labels; France has none):
  R1 look-alike      names differ by one swapped word, same house number + street, the swapped-in word is a common French
                     business word (>= MIN_FREQ S1 names) and not a filler / abbreviation / typo   (train: 0.19-0.75% same business)
Leaderboard check: R1 removals were ~91% truly wrong (0.982746 -> 0.983297). On held-out US/India the models already
reject these (R1 flags 0.022% of accepted pairs there vs ~0.9% in France), so R1 is for France only.

Checked and NOT applied (paired held-out test on v4's accepted pairs, 44,210 S1, train labels):
  R2 one-word swap at a different address: 89% of flagged accepted pairs are true matches; paired -0.00003; unproven in France
  R3 name + one extra word at a different address: 98.6% true matches; paired -0.00519 [-0.00558, -0.00483] -> harmful

  python scripts/france_rules.py --matching in.tsv --out out.tsv --test-dir dataset/test
"""
import argparse
import collections

import pandas as pd

import france_lookalike_filter as F

FR_FILLERS = {'associes', 'cie', 'developpement', 'fils', 'france', 'groupe', 'services'}  # label-free (filler log)


def classify(sn, sa, rn, ra, freq):
    """Returns 'R1' if a matched pair is a same-address one-word look-alike, else None."""
    ta, tb = set(F.toks(sn)), set(F.toks(rn))
    wa, wb = ta - tb, tb - ta
    if len(wa) != 1 or len(wb) != 1:
        return None
    a, b = next(iter(wa)), next(iter(wb))
    if (F.SequenceMatcher(None, a, b).ratio() >= 0.6 or a[:3] == b[:3] or F.is_abbrev(a, b)
            or b in FR_FILLERS | F.KEEP_WORDS or freq[b] < F.MIN_FREQ):
        return None
    ka, kb = F.addr_key(sa), F.addr_key(ra)
    return 'R1' if bool(ka[0]) and ka[0] == kb[0] and bool(ka[1] & kb[1]) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matching', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test-dir', default='dataset/test')
    ap.add_argument('--country', default='France')
    a = ap.parse_args()
    rd = lambda f: pd.read_csv(f'{a.test_dir}/{f}', sep='\t', dtype=str, keep_default_na=False, quoting=3)
    s1 = rd('test_source1.tsv'); s1 = s1[s1.country == a.country].set_index('entity_id')
    r = pd.concat([rd('test_source2.tsv'), rd('test_source3.tsv')]); r = r[r.country == a.country].set_index('entity_id')
    freq = collections.Counter(t for n in s1.business_name.values for t in set(F.toks(n)))
    sub = pd.read_csv(a.matching, sep='\t', dtype=str, keep_default_na=False)
    removed, rows = collections.Counter(), []
    for sid, ids in zip(sub.source1_entity_id, sub.matched_entity_ids):
        keep = [x for x in ids.split(',') if x]
        if sid in s1.index:
            out = []
            for x in keep:
                rule = classify(s1.business_name[sid], s1.business_address[sid], r.business_name[x], r.business_address[x], freq)
                if rule:
                    removed[rule] += 1
                else:
                    out.append(x)
            keep = out
        rows.append((sid, ','.join(keep)))
    with open(a.out, 'w', encoding='utf-8', newline='') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        f.writelines(f'{s}\t{m}\n' for s, m in rows)
    print(f'removed {sum(removed.values()):,} {a.country} matches {dict(removed)} -> {a.out}')


if __name__ == '__main__':
    main()
