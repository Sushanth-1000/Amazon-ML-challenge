"""E24: French decoy-rule mining (same principle as the teammates' look-alike filter: LB +0.00055 on France).
For each label-free rule on a (S1, record) pair, measure on TRAIN candidate pairs (v6.1 train_scored: q, s, y, p1) the
share of true matches among pairs where the rule fires, restricted to pairs a model would plausibly accept (p1 >= 0.3).
A rule qualifies if that share is <= 5% on US and on India separately (>= 200 firing pairs each). Then count how many
of the teammates' FRENCH matches (0.983297 file) each qualifying rule would remove. Nothing is applied here.
Usage: python e24_decoy_rules.py <w61> <dataset> <team matching_results.tsv>"""
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

W, DS, TEAM = sys.argv[1], sys.argv[2], sys.argv[3]
rd = lambda p: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
LEGAL = {"inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd", "corp": "corp",
         "corporation": "corp", "co": "co", "company": "co", "pvt": "pvt", "private": "pvt", "llp": "llp", "lp": "lp",
         "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc",
         "plc": "plc", "pc": "pc", "pllc": "pllc"}
GEN = {"the", "and", "of", "de", "du", "la", "le", "des", "et", "les"}


def fold(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def toks(s):
    return re.sub(r"[^a-z0-9 ]", " ", fold(s).replace("&", " and ")).split()


def feats(names_s, addr_s, names_r, addr_r):
    out = {k: [] for k in ("r1", "r2", "r3", "r4", "r5")}
    for ns, as_, nr, ar in zip(names_s, addr_s, names_r, addr_r):
        ts, tr = toks(ns), toks(nr)
        cs = [t for t in ts if t not in LEGAL and t not in GEN]; cr = [t for t in tr if t not in LEGAL and t not in GEN]
        a1, a2 = toks(as_), toks(ar)
        n1 = next((t for t in a1 if t[:1].isdigit()), ""); n2 = next((t for t in a2 if t[:1].isdigit()), "")
        st1 = {t for t in a1 if len(t) > 3 and not t[:1].isdigit()}; st2 = {t for t in a2 if len(t) > 3 and not t[:1].isdigit()}
        pc1 = {t for t in a1 if re.fullmatch(r"\d{5,6}", t)}; pc2 = {t for t in a2 if re.fullmatch(r"\d{5,6}", t)}
        same_name = set(cs) == set(cr) and len(cs) > 0
        # R1: same street, different leading house number, names not identical
        out["r1"].append(bool(n1 and n2 and n1 != n2 and (st1 & st2) and not same_name))
        # R2: same core name, both have postcodes, postcodes disjoint
        out["r2"].append(bool(same_name and pc1 and pc2 and not (pc1 & pc2)))
        # R3: two-word swap: same size, exactly two differing words on each side, >= 2 shared, same house number
        s1s, rrs = set(cs), set(cr)
        out["r3"].append(bool(len(s1s - rrs) == 2 and len(rrs - s1s) == 2 and len(s1s & rrs) >= 2 and n1 and n1 == n2))
        # R4: different legal forms, otherwise same core name
        l1 = {LEGAL[t] for t in ts if t in LEGAL}; l2 = {LEGAL[t] for t in tr if t in LEGAL}
        out["r4"].append(bool(l1 and l2 and not (l1 & l2) and same_name))
        # R5: a number inside the names on both sides, disagreeing, rest of the core name equal
        d1 = {t for t in cs if any(c.isdigit() for c in t)}; d2 = {t for t in cr if any(c.isdigit() for c in t)}
        out["r5"].append(bool(d1 and d2 and not (d1 & d2) and set(cs) - d1 == set(cr) - d2 and (set(cs) - d1)))
    return pd.DataFrame(out)


# ---- train: rule precision among plausible candidate pairs (p1 >= 0.3), per country
s1 = pd.read_parquet(f"{W}/train_s1.parquet", columns=["entity_id", "country", "raw_name", "raw_addr"])
r = pd.read_parquet(f"{W}/train_r.parquet", columns=["entity_id", "country", "raw_name", "raw_addr"])
t = pd.read_parquet(f"{W}/train_scored.parquet", columns=["q", "s", "p1", "y"])
t = t[t.p1.values >= 0.3].reset_index(drop=True)
F = feats(s1.raw_name.values[t.s.values], s1.raw_addr.values[t.s.values], r.raw_name.values[t.q.values], r.raw_addr.values[t.q.values])
cty = r.country.values[t.q.values]
res = {}
for rule in F.columns:
    row = {}
    for c in ("US", "India"):
        m = F[rule].values & (cty == c)
        row[c] = (int(m.sum()), float(t.y.values[m].mean()) if m.sum() else np.nan)
    ok = all(n >= 200 and rate <= 0.05 for n, rate in row.values())
    res[rule] = ok
    print(f"{rule}: US fires {row['US'][0]:,} match-rate {row['US'][1]:.3f} | India fires {row['India'][0]:,} "
          f"match-rate {row['India'][1]:.3f} -> {'QUALIFIES' if ok else 'no'}", flush=True)

# ---- France: how many of the teammates' French matches each rule would remove
ts1 = rd(f"{DS}/test/test_source1.tsv"); tr_ = pd.concat([rd(f"{DS}/test/test_source2.tsv"), rd(f"{DS}/test/test_source3.tsv")])
fs = ts1[ts1.country == "France"].set_index("entity_id"); fr = tr_[tr_.country == "France"].set_index("entity_id")
team = rd(TEAM)
pairs = [(sid, x) for sid, ms in zip(team.source1_entity_id, team.matched_entity_ids) if sid in fs.index for x in ms.split(",") if x]
P = pd.DataFrame(pairs, columns=["sid", "rid"])
FF = feats(fs.business_name.reindex(P.sid).values, fs.business_address.reindex(P.sid).values,
           fr.business_name.reindex(P.rid).values, fr.business_address.reindex(P.rid).values)
print(f"French matches in the 0.983297 file: {len(P):,} across {P.sid.nunique():,} S1")
for rule in F.columns:
    print(f"  {rule}: would remove {int(FF[rule].sum()):,} French matches ({'qualified' if res[rule] else 'not qualified'})")
P[[c for c in F.columns]] = FF.values
P.to_parquet(f"{W}/e24_france_flags.parquet")
