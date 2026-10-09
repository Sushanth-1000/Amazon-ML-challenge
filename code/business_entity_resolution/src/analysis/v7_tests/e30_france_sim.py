"""E30: label-free France post-processing, tested with labels on India playing France (US labelled, India unlabelled).
Per-S1 macro F0.5 (singletons included) on India EVAL S1; paired bootstrap vs the v6.2a recipe (st1, threshold from US).
Inputs = E19/E27 per-pair probabilities: reg (no self-training), st1 (self-trained), tonly_st1 (target-only).
  T7 stricter threshold chosen WITHOUT target labels: India threshold set so India's predicted matches per S1 equals
     US's at the US threshold (label-free rate matching); plus fixed offsets +0.04 / +0.08 (sensitivity only)
  T8 no-address records in the unlabelled country need p >= 0.9 / 0.95
  T9 combining versions: majority vote of the 3 record->S1 assignments; st1 AND reg; st1 AND tonly
Configs run in parallel (fork, 1 process per config)."""
import argparse
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--p_reg", required=True); ap.add_argument("--p_st1", required=True); ap.add_argument("--p_tonly", required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
import er  # noqa: E402
import stage2 as S  # noqa: E402

s1, r = er.load_split(a.data, "train", a.work)
truth, r2s = er.load_truth(a.data)
rb = er.id_bucket(r.entity_id.map(r2s).fillna(r.entity_id).values)
p = pd.read_parquet(f"{a.work}/loco_pairs_2_100.parquet", columns=["q", "s"])
country = r.country.values[p.q.values]
evm = rb[p.q.values] >= 51
sb = er.id_bucket(s1.entity_id.values)
ev_s1 = {c: np.flatnonzero((sb >= 51) & (s1.country.values == c)) for c in ("US", "India")}
sid, rid = s1.entity_id.values, r.entity_id.values
PR = {}
for k, f in (("reg", a.p_reg), ("st1", a.p_st1), ("tonly", a.p_tonly)):
    x = np.zeros(len(p), np.float32); x[evm] = np.load(f); PR[k] = x
noaddr_r = r.addr.values == ""
grid = np.round(np.arange(0.30, 0.991, 0.02), 2)


def assign(c, prob, t, na_min=None):
    m = evm & (country == c)
    asg = S._fast_decider(p.loc[m, ["q", "s"]], prob[m], r)({"t": {k: float(t) for k in S.SLICES}, "rescue": 0.0})
    if na_min is not None:
        pa = pd.Series(prob[m], index=pd.MultiIndex.from_arrays([p.q.values[m], p.s.values[m]]))
        pv = pa.reindex(pd.MultiIndex.from_arrays([asg.q.values, asg.s.values])).values
        asg = asg[~(noaddr_r[asg.q.values] & (pv < na_min))]
    return asg


def score(c, asg):
    pred = {}
    for q, s in zip(asg.q.values, asg.s.values):
        pred.setdefault(s, set()).add(rid[q])
    out = np.empty(len(ev_s1[c]))
    for k, s in enumerate(ev_s1[c]):
        P, G = pred.get(s, set()), truth.get(sid[s], set())
        if not P and not G: out[k] = 1.0; continue
        tp = len(P & G)
        out[k] = 0.0 if tp == 0 else 1.25 * (tp / len(P)) * (tp / len(G)) / (0.25 * (tp / len(P)) + tp / len(G))
    return out


def t_us(k):
    return max(grid, key=lambda t: score("US", assign("US", PR[k], t)).mean())


def rate(c, asg):
    return len(asg) / len(ev_s1[c])


def run(cfg):
    name, kind, arg = cfg
    if kind == "base":
        t = T[arg]; return name, score("India", assign("India", PR[arg], t)), f"t={t}"
    if kind == "ratematch":   # label-free: India threshold whose predicted matches/S1 equals US's at the US threshold
        t0 = T["st1"]; target = rate("US", assign("US", PR["st1"], t0))
        tt = min(grid[grid >= t0], key=lambda t: abs(rate("India", assign("India", PR["st1"], t)) - target))
        return name, score("India", assign("India", PR["st1"], tt)), f"t={tt} (US rate {target:.3f})"
    if kind == "offset":
        t = min(0.99, T["st1"] + arg); return name, score("India", assign("India", PR["st1"], t)), f"t={t:.2f}"
    if kind == "noaddr":
        return name, score("India", assign("India", PR["st1"], T["st1"], na_min=arg)), f"no-address p>={arg}"
    if kind == "vote":
        A = {k: assign("India", PR[k], T[k]) for k in arg}
        keys = [set(zip(v.q.values, v.s.values)) for v in A.values()]
        need = len(arg) if name.endswith("AND") or "AND" in name else 2
        cnt = {}
        for ks in keys:
            for x in ks: cnt[x] = cnt.get(x, 0) + 1
        kept = [x for x, n in cnt.items() if n >= need]
        asg = pd.DataFrame(kept, columns=["q", "s"]) if kept else pd.DataFrame({"q": [], "s": []}, dtype=int)
        asg = asg.sort_values("q").drop_duplicates("q")
        return name, score("India", asg), f"need {need} of {len(arg)}"


T = {k: t_us(k) for k in PR}
print(f"thresholds chosen on US (per-S1): {T}", flush=True)
cfgs = [("st1 (v6.2a recipe) BASE", "base", "st1"), ("reg (no self-training)", "base", "reg"),
        ("tonly (target-only)", "base", "tonly"),
        ("T7 rate-matched threshold (label-free)", "ratematch", None),
        ("T7 threshold +0.04 (sensitivity)", "offset", 0.04), ("T7 threshold +0.08 (sensitivity)", "offset", 0.08),
        ("T8 no-address p>=0.90", "noaddr", 0.90), ("T8 no-address p>=0.95", "noaddr", 0.95),
        ("T9 majority vote reg/st1/tonly", "vote", ("reg", "st1", "tonly")),
        ("T9 st1 AND reg", "vote", ("st1", "reg")), ("T9 st1 AND tonly", "vote", ("st1", "tonly"))]
with Pool(len(cfgs)) as pool:
    res = {n: (f, info) for n, f, info in pool.map(run, cfgs)}
base = res["st1 (v6.2a recipe) BASE"][0]
n = len(base); B = np.random.default_rng(0).integers(0, n, size=(1000, n))
print(f"\n=== E30 India (unlabelled, plays France) EVAL S1 {n:,}: per-S1 F0.5, paired vs st1 BASE (1000 bootstrap) ===")
for name, (f, info) in res.items():
    d = f - base; bm = d[B].mean(1)
    print(f"{name:42s} {f.mean():.4f} | {d.mean():+.5f} [{np.quantile(bm, .025):+.5f}, {np.quantile(bm, .975):+.5f}] | {info}",
          flush=True)
print("E30 DONE", flush=True)
