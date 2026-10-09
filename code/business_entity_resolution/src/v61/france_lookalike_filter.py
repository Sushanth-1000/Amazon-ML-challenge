"""Remove look-alike decoy matches in one country (used for France) from a matching_results.tsv.

Rule (learned on TRAIN labels, see docs below): a matched pair whose business names differ by exactly one whole word
(same house number, same street, not a typo, not an abbreviation) is a decoy when the swapped-in record word is a
common business word of that country (>= MIN_FREQ Source-1 names) and not a generator "filler" word.
Fillers are found without labels: words that sit right after a legal form (SAS, SARL, LLC...) in >= FILLER_SHARE of
the country's one-word pairs.

Train evidence (labels, 200k train S1): swapped-in common word -> same business in 4.6% of pairs; the same filler
detection recovers exactly {center, services, service, partners} in US and India; flagged pairs are 99.2% decoys in
India and 73% in US. Uses only the test files' text and token counts (unsupervised statistics), no test labels.

  python scripts/france_lookalike_filter.py --matching in.tsv --out out.tsv --test-dir dataset/test --country France
"""
import argparse
import collections
import re
import unicodedata
import zlib
from difflib import SequenceMatcher

import pandas as pd

LEGAL = set("inc llc ltd pvt private limited corp corporation co company pc pllc lp llp plc incorporated "
            "sas sarl sasu eurl sci sa snc ei scop selarl sca gie".split())
STOP = {"the", "and", "of", "de", "du", "la", "le", "des", "et"}
MIN_FREQ, FILLER_SHARE, FILLER_MIN_N, KEY_MAX_S1, KEY_MAX_R = 100, 0.30, 30, 20, 200
KEEP_WORDS = {"service", "services", "center", "centre", "partners", "societe"}  # filler-like / ambiguous: never flag


def fold(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def raw_toks(s):
    return re.sub(r"[^a-z0-9 ]", " ", fold(s).split("|")[0].replace("&", " and ")).split()


def toks(s):
    return [t for t in raw_toks(s) if t not in LEGAL and t not in STOP]


def addr_key(a):
    t = re.sub(r"[^a-z0-9 ]", " ", fold(a)).split()
    return next((x for x in t if x[:1].isdigit()), ""), {x for x in t if len(x) > 3 and not x[:1].isdigit()}


def is_abbrev(a, b):
    """b is an abbreviation of a (or the reverse): same first letter and letters in order (st/saint, cie/compagnie)."""
    short, long_ = sorted((a, b), key=len)
    if short[:1] != long_[:1]:
        return False
    it = iter(long_)
    return all(ch in it for ch in short)


def one_word_pairs(s1, r):
    """All (S1 row, record row) whose name word sets differ by exactly one word on each side (deletion join)."""
    def keys(names):
        out = []
        for i, n in enumerate(names):
            t = sorted(set(toks(n)))
            if 2 <= len(t) <= 6:
                out += [(i, zlib.crc32(" ".join(t[:j] + t[j + 1:]).encode())) for j in range(len(t))]
        return pd.DataFrame(out, columns=["i", "k"])
    ks, kr = keys(s1.business_name.values), keys(r.business_name.values)
    ns, nr = ks.k.value_counts(), kr.k.value_counts()
    ks = ks[ks.k.map(ns).le(KEY_MAX_S1).values & ks.k.map(nr).fillna(0).le(KEY_MAX_R).values]
    m = ks.merge(kr[kr.k.isin(ks.k)], on="k", suffixes=("_s", "_r")).drop_duplicates(["i_s", "i_r"])
    return m[["i_s", "i_r"]].reset_index(drop=True)


def describe(s1, r, m):
    sn, rn = s1.business_name.values, r.business_name.values
    rows = []
    for a, b in zip(m.i_s, m.i_r):
        ta, tb = toks(sn[a]), toks(rn[b])
        wa, wb = set(ta) - set(tb), set(tb) - set(ta)
        if len(wa) != 1 or len(wb) != 1:
            continue
        wa, wb = wa.pop(), wb.pop()
        rt = raw_toks(rn[b]); j = rt.index(wb) if wb in rt else 0
        after_legal = j > 0 and (rt[j - 1] in LEGAL or (rt[j - 1] == "and" and j > 1 and rt[j - 2] in LEGAL))
        rows.append((a, b, wa, wb, after_legal))
    d = pd.DataFrame(rows, columns=["i_s", "i_r", "wa", "wb", "after_legal"])
    d["typo"] = [SequenceMatcher(None, x, y).ratio() >= 0.6 or x[:3] == y[:3] for x, y in zip(d.wa, d.wb)]
    ak = {i: addr_key(s1.business_address.values[i]) for i in d.i_s.unique()}
    bk = {i: addr_key(r.business_address.values[i]) for i in d.i_r.unique()}
    d["same_num"] = [ak[a][0] != "" and ak[a][0] == bk[b][0] for a, b in zip(d.i_s, d.i_r)]
    d["street"] = [bool(ak[a][1] & bk[b][1]) for a, b in zip(d.i_s, d.i_r)]
    return d


def decoy_pairs(s1, r):
    d = describe(s1, r, one_word_pairs(s1, r))
    base = d[~d.typo & d.same_num & d.street]
    g = base.groupby("wb").after_legal.agg(["size", "mean"])
    fillers = set(g[(g["size"] >= FILLER_MIN_N) & (g["mean"] >= FILLER_SHARE)].index)
    freq = collections.Counter(t for n in s1.business_name.values for t in set(toks(n)))
    flag = (~base.wb.isin(fillers | KEEP_WORDS) & base.wb.map(lambda w: freq[w] >= MIN_FREQ)
            & ~pd.Series([is_abbrev(a, b) for a, b in zip(base.wa, base.wb)], index=base.index))
    f = base[flag]
    print(f"fillers (label-free): {sorted(fillers)} | flagged one-word decoy pairs: {len(f):,}")
    return set(zip(s1.entity_id.values[f.i_s], r.entity_id.values[f.i_r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--country", default="France")
    a = ap.parse_args()
    rd = lambda f: pd.read_csv(f"{a.test_dir}/{f}", sep="\t", dtype=str, keep_default_na=False, quoting=3)
    s1 = rd("test_source1.tsv"); r = pd.concat([rd("test_source2.tsv"), rd("test_source3.tsv")], ignore_index=True)
    s1 = s1[s1.country == a.country].reset_index(drop=True); r = r[r.country == a.country].reset_index(drop=True)
    bad = decoy_pairs(s1, r)

    sub = pd.read_csv(a.matching, sep="\t", dtype=str, keep_default_na=False)
    removed, rows = 0, []
    for sid, ids in zip(sub.source1_entity_id, sub.matched_entity_ids):
        keep = [x for x in ids.split(",") if x and (sid, x) not in bad]
        removed += len([x for x in ids.split(",") if x]) - len(keep)
        rows.append((sid, ",".join(keep)))
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(f"{s}\t{m}\n" for s, m in rows)
    print(f"removed {removed:,} matches from {a.country} -> {a.out}")


if __name__ == "__main__":
    main()
