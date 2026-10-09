"""Decision rules + error analysis on the validation set produced by `er.py --stage train`.

Rebuilds the exact validation pairs, scores them with work/model.txt (cached in work/val_pred.parquet),
tunes each decoder on half A of the validation S1 entities and reports macro F0.5 on half B, writes the
winner into work/meta.json (used by `er.py --stage test`), then breaks the remaining errors down by slice.

Usage (from student_resource/):  python src/decode.py --work work
"""
import argparse
import json

import numpy as np
import pandas as pd

from er import decode, id_bucket, load_split, log, macro_f05, read_tsv, to_sets


def val_predictions(args):
    """Validation pairs with stage-1 probabilities, written by `er.py --stage train` (work/val_pred.parquet)."""
    s1, r = load_split(args.data, "train", args.work)
    r = r.iloc[np.load(f"{args.work}/train_keep.npy")].reset_index(drop=True)
    return s1, r, pd.read_parquet(f"{args.work}/val_pred.parquet")


def score(pva, prob, noaddr, dec, s1, r, truth, ids):
    return macro_f05(to_sets(decode(pva, prob, dec, noaddr), s1, r), truth, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--val-pct", type=int, default=1)
    args = ap.parse_args()

    s1, r, pva = val_predictions(args)
    # Pairs below 1% never win an assignment and barely move expected counts; dropping them makes
    # each decoder evaluation ~10x faster on the ~12M validation pairs.
    pva = pva[pva.prob.values >= 0.01].reset_index(drop=True)
    prob, noaddr = pva.prob.values, r.addr.values[pva.q.values] == ""
    gt = read_tsv(f"{args.data}/train/train_ground_truth.tsv")
    s_val = id_bucket(s1.entity_id.values) < args.val_pct
    val_ids = s1.entity_id.values[s_val]
    truth = {i: set(x for x in ms.split(",") if x) for i, ms in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    truth = {i: truth[i] for i in val_ids}
    half = id_bucket(np.array([i[::-1] for i in val_ids])) < 50  # independent of the val-selection hash
    ids_a, ids_b = val_ids[half], val_ids[~half]
    log(f"val entities {len(val_ids):,} (A {len(ids_a):,} / B {len(ids_b):,}), pairs {len(pva):,}")

    grids = {
        "threshold": [{"kind": "threshold", "t": t} for t in np.arange(0.2, 0.96, 0.025)],
        "margin": [{"kind": "margin", "t": t, "m": m} for t in np.arange(0.3, 0.91, 0.05)
                   for m in (0.0, 0.1, 0.2, 0.3, 0.5)],
        "expf": [{"kind": "expf", "floor": f} for f in (0.02, 0.05, 0.1, 0.2, 0.3)],
    }
    results = {}
    f = lambda d, ids: score(pva, prob, noaddr, d, s1, r, truth, ids)
    for kind, grid in grids.items():
        tuned = max(grid, key=lambda d: f(d, ids_a))
        key = "floor" if kind == "expf" else "t"  # then a separate, stricter cut for no-address records
        tuned = max([tuned] + [{**tuned, "t_na": float(x)} for x in np.arange(tuned[key], 1.0, 0.025)],
                    key=lambda d: f(d, ids_a))
        results[kind] = (tuned, f(tuned, ids_b), f(tuned, val_ids))
        log(f"{kind:9s} tuned on A: {tuned} -> F0.5 on B {results[kind][1]:.4f} | all val {results[kind][2]:.4f}")
    y = (s1.entity_id.values[pva.s.values] == r.entity_id.map(
        {m: s for s, ms in truth.items() for m in ms}).values[pva.q.values])
    ceil = macro_f05(to_sets(pva[y], s1, r), truth, val_ids)
    log(f"perfect matcher on these candidates (blocking ceiling): {ceil:.4f} | predict-nothing floor: "
        f"{np.mean([not truth[i] for i in val_ids]):.4f}")

    best_kind = max(results, key=lambda k: results[k][1])
    meta = json.load(open(f"{args.work}/meta.json"))
    meta["decoder"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in results[best_kind][0].items()}
    meta["decoder_val_f05_B"] = results[best_kind][1]
    json.dump(meta, open(f"{args.work}/meta.json", "w"))
    log(f"chosen decoder -> meta.json: {meta['decoder']}")

    # ---- error analysis with the chosen decoder, per S1 entity and per record
    a = decode(pva, prob, meta["decoder"], noaddr)
    pred = to_sets(a, s1, r)
    ent = pd.DataFrame({"id": val_ids, "country": s1.country.values[s_val]})
    ent["n_true"] = ent.id.map(lambda i: len(truth[i]))
    ent["f05"] = [macro_f05(pred, truth, [i]) for i in ent.id]
    ent["kind"] = np.select([ent.n_true == 0], ["singleton"], "has matches")
    print("\n=== macro F0.5 by slice (entity level) ===")
    for by in ("country", "kind"):
        print(ent.groupby(by).f05.agg(["mean", "size"]).round(4).to_string(), "\n")
    print("lost F0.5 points by slice (share of total loss):")
    ent["loss"] = 1 - ent.f05
    print((ent.groupby(["country", "kind"]).loss.sum() / ent.loss.sum()).round(3).to_string())

    rid2s = {m: s for s, ms in truth.items() for m in ms}
    rec = r.iloc[np.unique(pva.q.values)].copy()
    rec["true_s"] = rec.entity_id.map(rid2s)
    rec["pred_s"] = rec.index.map(pd.Series(s1.entity_id.values[a.s.values], index=a.q.values))
    in_val = rec.true_s.isin(set(val_ids)) | rec.pred_s.isin(set(val_ids))
    rec = rec[in_val]
    cand = pva.assign(sid=s1.entity_id.values[pva.s.values]).groupby("q").sid.agg(set)
    rec["blocked"] = [t in cand.get(q, ()) for q, t in zip(rec.index, rec.true_s)]
    rec["outcome"] = np.select(
        [rec.true_s.notna() & (rec.pred_s == rec.true_s),
         rec.pred_s.notna() & (rec.pred_s != rec.true_s) & rec.true_s.notna(),
         rec.pred_s.notna() & rec.true_s.isna(),
         rec.true_s.notna() & rec.pred_s.isna() & ~rec.blocked,
         rec.true_s.notna() & rec.pred_s.isna()],
        ["correct", "FP wrong entity", "FP decoy merged", "FN not in candidates", "FN model rejected"],
        "correct no-match")
    rec["source"] = rec.entity_id.str[:2]
    rec["script"] = np.where(rec.nonlatin, "native", "latin")
    rec["address"] = np.where(rec.addr == "", "empty", "present")
    print("\n=== record outcomes ===")
    print(rec.outcome.value_counts().to_string())
    for by in ("country", "source", "script", "address"):
        print(f"\n--- by {by} (row %) ---")
        print(pd.crosstab(rec[by], rec.outcome, normalize="index").round(4).to_string())
    sn = s1.set_index("entity_id")
    for kind in ("FP wrong entity", "FP decoy merged", "FN model rejected", "FN not in candidates"):
        ex = rec[rec.outcome == kind].head(8)
        print(f"\n--- examples: {kind} ---")
        for _, x in ex.iterrows():
            t = sn.loc[x.true_s] if isinstance(x.true_s, str) else None
            pr = sn.loc[x.pred_s] if isinstance(x.pred_s, str) else None
            print(f"  rec : {x['name']} | {x['addr']}")
            if t is not None: print(f"  true: {t['name']} | {t['addr']}")
            if pr is not None: print(f"  pred: {pr['name']} | {pr['addr']}")


if __name__ == "__main__":
    main()
