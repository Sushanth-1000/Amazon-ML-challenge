"""E14 (built on E7): French-style AUGMENTATION. Train on US split A with normal text AND a French-style rewrite of the
same training pairs; evaluate on normal and French-style US split B. Compared with the normal-only model (base).
Original E7 description follows.
E7: French-style robustness test with REAL labels (train only; rules from general French conventions, nothing
learned from test). US EVAL records (split B) and their candidate S1 are rewritten in French style; the candidate pairs and
labels stay the same. Models are trained on US split A (normal text) for each config and scored on:
  (a) the normal US EVAL pairs, (b) the same pairs with French-style text  ->  robustness = (b) - (a), paired.
A config that loses less under the French-style rewrite is more likely to hold up on France.
Rewrites (applied with fixed probabilities, seeded): legal forms (LLC/Inc/Corp/Ltd/Co -> SARL/SAS/SASU/EURL/SA/Cie),
street types (Street/Avenue/Boulevard/Road/Drive/Lane/Place -> Rue/Avenue/Boulevard/Chemin/Allée/Impasse/Place, moved to
French order "Rue X"), 'bis/ter' house-number suffixes, accents on common letters, postcode + city order, English generic
business words -> French equivalents (Services->Services, Group->Groupe, Associates->Associés, Company->Compagnie,
Center->Centre, School->École, Club->Club, Sons->Fils, Friends->Amis, Health->Santé, Development->Développement).
Run from student_resource/: python e7_frenchstyle.py --data dataset --work ~/w4 --src ~/src5 --out ~/e7"""
import argparse
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset")
ap.add_argument("--work", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args()
sys.path.insert(0, args.src)
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz.process import cpdist  # noqa: E402

import er  # noqa: E402
import stage2 as S  # noqa: E402
from normalize import core_name, normalize_record  # noqa: E402

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
os.makedirs(args.out, exist_ok=True)
W = args.work
s1, r = er.load_split(args.data, "train", W)
truth, r2s = er.load_truth(args.data)
rkey = r.entity_id.map(r2s).fillna(r.entity_id).values
rb = er.id_bucket(rkey)
p = pd.read_parquet(f"{W}/loco_pairs_2_100.parquet")
y = p.y.values.astype(np.int8)
country = r.country.values[p.q.values]
fitm = (rb[p.q.values] < 51) & (country == "US")
evm = (rb[p.q.values] >= 51) & (country == "US")
half = rb[p.q.values] % 2
log(f"fit {int(fitm.sum()):,} | eval {int(evm.sum()):,}")

# ---------------- French-style rewrite (general conventions only)
rng = np.random.default_rng(7)
LEGAL = [(r"\b(llc|l\.l\.c\.)\b", ["SARL", "SAS", "EURL"]), (r"\b(inc|inc\.|incorporated)\b", ["SA", "SAS", "SASU"]),
         (r"\b(corp|corp\.|corporation)\b", ["SA", "SAS"]), (r"\b(ltd|ltd\.|limited)\b", ["SARL", "SA"]),
         (r"\b(co|co\.|company)\b", ["Cie", "Compagnie"])]
WORDS = {"services": "Services", "group": "Groupe", "associates": "Associés", "center": "Centre", "centre": "Centre",
         "school": "École", "sons": "Fils", "friends": "Amis", "health": "Santé", "development": "Développement",
         "management": "Gestion", "consulting": "Conseil", "solutions": "Solutions", "holdings": "Participations",
         "partners": "Partenaires", "international": "International", "medical": "Médical", "dental": "Dentaire"}
STREET = {"street": "Rue", "st": "Rue", "avenue": "Avenue", "ave": "Avenue", "boulevard": "Boulevard", "blvd": "Bd",
          "road": "Chemin", "rd": "Chemin", "drive": "Allée", "dr": "Allée", "lane": "Impasse", "ln": "Impasse",
          "place": "Place", "pl": "Place", "court": "Cour", "ct": "Cour", "way": "Voie"}
ACC = str.maketrans({"e": "é", "a": "à", "c": "ç", "o": "ô"})


def fr_name(n, seed):
    g = np.random.default_rng(seed)
    for pat, reps in LEGAL:
        n = re.sub(pat, lambda m: reps[g.integers(len(reps))], n, flags=re.I)
    toks = [WORDS.get(t.lower(), t) for t in n.split()]
    if g.random() < 0.15 and toks:  # occasional accent on one word (French text is accented)
        i = int(g.integers(len(toks))); toks[i] = toks[i][:1] + toks[i][1:].translate(ACC) if len(toks[i]) > 3 else toks[i]
    return " ".join(toks)


def fr_addr(a, seed):
    if not a.strip():
        return a
    g = np.random.default_rng(seed)
    parts = [x.strip() for x in a.split(",")]
    first = parts[0].split()
    if len(first) >= 3 and first[0].isdigit() and first[-1].lower().strip(".") in STREET:  # "12 Main Street" -> "12 Rue Main"
        num, typ, name = first[0], STREET[first[-1].lower().strip(".")], " ".join(first[1:-1])
        if g.random() < 0.08:
            num += g.choice([" bis", " ter"])
        parts[0] = f"{num} {typ} {name}"
    if len(parts) >= 3 and g.random() < 0.5:  # city/region order varies in French data
        parts[1], parts[2] = parts[2], parts[1]
    return ", ".join(parts)


ev_q = np.unique(p.q.values[evm | fitm]); ev_s = np.unique(p.s.values[evm | fitm])
r_fr, s1_fr = r.copy(), s1.copy()
for d, rows, is_r in ((r_fr, ev_q, True), (s1_fr, ev_s, False)):
    names = [fr_name(d.raw_name.values[i], i * 2 + is_r) for i in rows]
    addrs = [fr_addr(d.raw_addr.values[i], i * 3 + is_r) for i in rows]
    norm = [normalize_record(n, a, "France") for n, a in zip(names, addrs)]  # French cleaning path
    cols = [c for c in er.NORM_COLS if c in d.columns]
    upd = pd.DataFrame(norm, columns=er.NORM_COLS)[cols]
    for c in cols:
        d.loc[rows, c] = upd[c].values
    d.loc[rows, "raw_name"], d.loc[rows, "raw_addr"] = names, addrs
log(f"rewrote {len(ev_q):,} records and {len(ev_s):,} S1; example: {r.raw_name.values[ev_q[0]]} | {r.raw_addr.values[ev_q[0]]}"
    f"  ->  {r_fr.raw_name.values[ev_q[0]]} | {r_fr.raw_addr.values[ev_q[0]]}")


def stats(ss, rr, per_country):
    """Unsupervised statistics on (possibly rewritten) text, recomputed from scratch.
    per_country=False: v4 (split-wide generic words and IDF, raw counts). True: E1's 'inv' (US-only generic words and IDF,
    counts relative to the country's size). Only US rows are evaluated here, so 'per country' = US."""
    if per_country:
        ms, mr = ss.country.values == "US", rr.country.values == "US"
    else:
        ms, mr = np.ones(len(ss), bool), np.ones(len(rr), bool)
    df = Counter()
    for x in list(ss.name.values[ms]) + list(rr.name.values[mr]):
        df.update(set(x.split()))
    n = int(ms.sum() + mr.sum())
    gen = {w for w, k in df.items() if k / n > er.GENERIC_DF}
    idf = {"idf": {w: math.log(n / c) for w, c in df.items() if c > 1}, "default": math.log(n)}
    if per_country:
        ctx = {}
        for name, d, col in (("s1_name_freq", ss, "name"), ("r_name_freq", rr, "name"), ("s1_addr_freq", ss, "addr")):
            key = d.country + "|" + d[col]
            ctx[name] = (key.map(key.value_counts()).values / d.country.map(d.country.value_counts()).values * 1e6).astype(np.float32)
    else:
        ctx = er.split_context(ss, rr)
    cs = np.array([core_name(x, gen) for x in ss.name.values], object)
    cr = np.array([core_name(x, gen) for x in rr.name.values], object)
    return idf, ctx, cs, cr


rn, sn = r.name.values[p.q.values], s1.name.values[p.s.values]
rt, st = r["tail"].values[p.q.values], s1["tail"].values[p.s.values]
tset = cpdist(list(rn), list(sn), scorer=fuzz.token_set_ratio, workers=-1)
tables = lambda m: er._label_tables(rn[m], sn[m], rt[m], st[m], y[m], tset[m])


def feats(ss, rr, m, lo, loc, st_):
    idf, ctx, cs, cr = st_
    ss, rr = ss.copy(), rr.copy(); ss["name_core"], rr["name_core"] = cs, cr
    rows = np.flatnonzero(m)
    return er.features(p.iloc[rows].reset_index(drop=True), ss, rr, lo, idf, loc, ctx).set_index(rows)


def add_fmt(X):
    X = X.copy()
    X["addr_len_ratio"] = (np.minimum(X.r_addr_len, X.s_addr_len) / np.maximum(np.maximum(X.r_addr_len, X.s_addr_len), 1)).astype(np.float32)
    d = X.num_first_delta.values
    X["num_delta_class"] = np.where(np.isnan(d), -1, np.where(d == 0, 0, np.where(d <= 20, 1, 2))).astype(np.float32)
    return X.drop(columns=["r_addr_len", "s_addr_len", "r_name_len", "s_name_len", "num_first_delta", "num_min_delta"])


s1_pos = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
true_s = r.entity_id.map(r2s).map(s1_pos).fillna(-1).astype(np.int64).values
ev_rec = np.flatnonzero((r.country.values == "US") & (rb >= 51))
cl = pd.factorize(rkey[ev_rec])[0]
WB = np.random.default_rng(0).poisson(1.0, size=(args.boot, cl.max() + 1)).astype(np.float32)


def outcome(prob, t):
    run = S._fast_decider(p.loc[evm, ["q", "s"]], prob[evm], r)
    a = run({"t": {k: float(t) for k in S.SLICES}, "rescue": 0.0})
    asg = np.full(len(r), -2, np.int64); asg[a.q.values] = a.s.values
    got, tru = asg[ev_rec], true_s[ev_rec]
    tp = (got >= 0) & (got == tru)
    return tp, (got >= 0) & ~tp, (tru >= 0) & ~tp


def f05(tp, fp, fn):
    P, R = tp / np.maximum(tp + fp, 1e-9), tp / np.maximum(tp + fn, 1e-9)
    return 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12)


draws = lambda o: f05(*(WB @ np.bincount(cl, weights=x.astype(np.float32), minlength=cl.max() + 1) for x in o))
res, D = [], {}
st_n, st_f = stats(s1, r, False), stats(s1_fr, r_fr, False)
tab_n = [tables(fitm & (half != h)) for h in (0, 1)]
Xf_n = pd.concat([feats(s1, r, fitm & (half == h), *tab_n[h], st_n) for h in (0, 1)]).sort_index()
Xf_f = pd.concat([feats(s1_fr, r_fr, fitm & (half == h), *tab_n[h], st_f) for h in (0, 1)]).sort_index()
lo, loc = tables(fitm)
Xn, Xfr = feats(s1, r, evm, lo, loc, st_n), feats(s1_fr, r_fr, evm, lo, loc, st_f)
log("features ready (normal + French-style, train and eval)")
inner = er.id_bucket(r.entity_id.values[p.q.values[fitm]]) % 10 == 0
grid = np.round(np.arange(0.02, 0.99, 0.02), 2)
for name, (Xtr, ytr, hold) in {"base": (Xf_n, y[fitm], inner),
                                "augmented": (pd.concat([Xf_n, Xf_f], ignore_index=True), np.r_[y[fitm], y[fitm]], np.r_[inner, inner])}.items():
    m = er.train_lgb(Xtr, ytr, hold)
    pn = np.zeros(len(p)); pn[evm] = m.predict(Xn[Xtr.columns], num_threads=0)
    pf = np.zeros(len(p)); pf[evm] = m.predict(Xfr[Xtr.columns], num_threads=0)
    t = max(grid, key=lambda t: f05(*[x.sum() for x in outcome(pn, t)]))
    on, of = outcome(pn, t), outcome(pf, t)
    D[(name, "normal")], D[(name, "french")] = draws(on), draws(of)
    for kind, o in (("normal", on), ("french-style", of)):
        tp, fp, fn = (x.sum() for x in o)
        res.append(dict(config=name, text=kind, F05=f05(tp, fp, fn), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), t=t))
    log(f"{name}: normal {res[-2]['F05']:.4f} | french-style {res[-1]['F05']:.4f}")
CFG = {"base": None, "augmented": None}
pd.set_option("display.width", 200)
print("\n=== E7 French-style robustness (US split A -> US split B; pair-level F0.5) ===")
print(pd.DataFrame(res).round(4).to_string(index=False))
print("\n=== French-style score of each config vs base (paired, 95% CI) ===")
for name in list(CFG)[1:]:
    dd = D[(name, "french")] - D[("base", "french")]
    print(f"{name:12s}: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
for kind in ("normal", "french"):
    dd = D[("augmented", kind)] - D[("base", kind)]
    print(f"augmented - base on {kind:7s} text: {dd.mean():+.4f}  CI [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
log("E14 done")
