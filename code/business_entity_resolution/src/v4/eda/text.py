"""EDA: (a) token swaps mined from true pairs (what normalization should unify),
(b) noise-type rates per source/country, (c) France vocabulary vs train and normalization coverage."""
import os, re, sys, collections
import numpy as np, pandas as pd
from unidecode import unidecode
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from er import read_tsv
from normalize import LEGAL, ADDR_ABBR, ADDR_DROP

W = "work_eda"
tok = lambda s: re.findall(r"[a-z0-9]+", unidecode(s).lower())

def swaps(pairs, col_r, col_s, top=30):
    """Count (S1 token -> S2/S3 token) where token multisets differ by exactly one substitution."""
    c = collections.Counter()
    for a, b in zip(pairs[col_s], pairs[col_r]):
        ca, cb = collections.Counter(tok(a)), collections.Counter(tok(b))
        da, db = ca - cb, cb - ca
        if sum(da.values()) == 1 and sum(db.values()) == 1:
            c[(next(iter(da)), next(iter(db)))] += 1
    return pd.Series({f"{x} -> {y}": n for (x, y), n in c.most_common(top)})

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    s1 = pd.read_parquet(f"{W}/train_s1.parquet", columns=["entity_id", "business_name", "business_address", "name", "addr"])
    q = pd.read_parquet(f"{W}/train_q.parquet")
    gt = read_tsv("dataset/train/train_ground_truth.tsv")
    r2s = {m: s for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m}
    q["true_s1"] = q.entity_id.map(r2s)
    p = q[q.true_s1.notna()].merge(s1.add_prefix("s_"), left_on="true_s1", right_on="s_entity_id")
    p["source"] = p.entity_id.str[:2]
    print(f"true pairs in sample: {len(p):,}")

    for c in ("US", "India"):
        d = p[(p.country == c) & ~p.nonlatin]
        print(f"\n=== {c}: top NAME swaps (S1 -> noisy) ===\n", swaps(d, "business_name", "s_business_name").to_string())
        print(f"\n=== {c}: top ADDRESS swaps ===\n", swaps(d, "business_address", "s_business_address").to_string())
        # swaps our normalization still leaves unresolved
        print(f"\n=== {c}: ADDRESS swaps remaining AFTER normalization ===\n", swaps(d, "addr", "s_addr", 20).to_string())

    # noise-type rates on true pairs (multi-label), per source x country
    R = pd.DataFrame({
        "name_identical_raw": p.business_name == p.s_business_name,
        "name_identical_norm": p.name == p.s_name,
        "name_reordered": (p.name != p.s_name) & (p.name.str.split().map(sorted) == p.s_name.str.split().map(sorted)),
        "name_nonlatin": p.nonlatin,
        "name_domain": p.business_name.str.contains(r"\.(?:com|in|net|org|fr)\b", case=False),
        "name_trade_dba": p.business_name.str.contains(r"\b(?:dba|formerly|aka)\b", case=False),
        "name_junk_prefix": p.business_name.str.match(r"^[^\wऀ-෿]"),
        "name_brackets": p.business_name.str.contains(r"[\[\(]") & ~p.s_business_name.str.contains(r"[\[\(]"),
        "name_no_shared_word": [not (set(a.split()) & set(b.split())) for a, b in zip(p.name, p.s_name)],
        "addr_empty": p.business_address.str.strip() == "",
        "addr_identical_norm": p.addr == p.s_addr,
        "addr_reordered": (p.addr != p.s_addr) & (p.addr.str.split().map(sorted) == p.s_addr.str.split().map(sorted)),
        "addr_landmark_added": p.addr.str.contains(r"\b(?:nr|opp|behind|beside)\b") & ~p.s_addr.str.contains(r"\b(?:nr|opp|behind|beside)\b"),
        "addr_nonlatin": p.business_address.str.contains(r"[^\x00-ɏ]"),
        "num_conflict": [bool(a and b and not (set(a.split()) & set(b.split())))
                         for a, b in zip(p.nums, p.addr.str.findall(r"\d+").str.join(" ").where(False, None).fillna(""))],
    })
    snum = p.s_addr.str.findall(r"\d+").map(set); rnum = p.addr.str.findall(r"\d+").map(set)
    R["num_conflict"] = [bool(a and b and not (a & b)) for a, b in zip(rnum, snum)]
    R["num_partial"] = [bool(a and b and (a & b) and a != b) for a, b in zip(rnum, snum)]
    R["num_dropped"] = [bool(b and not a) for a, b in zip(rnum, snum)]
    print("\n=== noise-type rates on TRUE pairs (share of pairs) ===\n",
          R.groupby([p.source, p.country]).mean().T.round(3).to_string())

    # France (test only): vocabulary and normalization coverage
    t1 = pd.read_parquet(f"{W}/test_s1.parquet")
    tq = pd.read_parquet(f"{W}/test_q.parquet")
    fr1, frq = t1[t1.country == "France"], tq[tq.country == "France"]
    trn_vocab = collections.Counter(t for s in pd.concat([s1.business_name.sample(300000, random_state=0),
                                                         q.business_name]) for t in tok(s))
    for label, d in (("France S1 names", fr1.business_name), ("France S2/S3 names", frq.business_name),
                     ("France S1 addresses", fr1.business_address), ("France S2/S3 addresses", frq.business_address)):
        cnt = collections.Counter(t for s in d for t in tok(s))
        tot = sum(cnt.values())
        unseen = sum(n for t, n in cnt.items() if t not in trn_vocab) / tot
        print(f"\n=== {label}: {len(d):,} rows, token share unseen in train names = {unseen:.3f} ===")
        print("  top tokens:", ", ".join(f"{t}({n})" for t, n in cnt.most_common(40)))
    print("\nFrance S2/S3 non-Latin name share:", round(frq.nonlatin.mean(), 4))
    print("France S2/S3 empty address share:", round((frq.business_address.str.strip() == "").mean(), 4))
    legal_fr = collections.Counter(t for s in pd.concat([fr1.business_name, frq.business_name]) for t in tok(s)
                                   if t in {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl", "scm", "gie", "ei", "earl", "gaec", "association", "asso", "ste", "societe", "cie", "et"})
    print("French legal/company tokens seen:", dict(legal_fr.most_common()))
    short = collections.Counter(t for s in frq.business_address for t in tok(s) if len(t) <= 4 and not t.isdigit())
    print("France short address tokens (abbrev candidates):", dict(short.most_common(40)))
    print("  of these, handled by ADDR_ABBR/ADDR_DROP:", [t for t, _ in short.most_common(40) if t in ADDR_ABBR or t in ADDR_DROP or t in ADDR_ABBR.values()])
    print("\nFrance samples (S2/S3 raw -> normalized):")
    for _, r in frq.sample(12, random_state=1).iterrows():
        print(f"  {r.business_name!r} | {r.business_address!r}\n     -> {r['name']!r} | {r.addr!r}")
