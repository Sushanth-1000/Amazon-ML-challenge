"""Cross-encoder (v4): fine-tuned BAAI/bge-reranker-v2-m3 (Apache-2.0) reading an S1 record and an S2/S3 record together.

v4 changes (from the v3 error analysis):
  * structured input: [NAME] ... [ADDR] ... [NUM] ... [UNIT] ... or [NO ADDRESS], so numbers / unit codes / missing
    addresses are explicit (decoys differ by one house number; no-address records were the main loss);
  * training groups = one record with its top candidates (split A only, out-of-fold p1); no-address groups x3;
  * synthetic groups generated from split-A TRAIN records with the known noise (positives: abbreviations, word order,
    dropped address, accents, legal-suffix swap; decoys: house number +-1..20, unit code change, added generic word);
  * two heads on the same logit: pointwise BCE + listwise "select" (softmax over the record's candidates + a learned
    "none" slot), matching "each record belongs to at most one S1";
  * all pairs with p1 >= MIN_P1 are scored (v3 skipped p1 >= 0.999 and let a confident decoy through); symmetric
    scoring (S1, record) + (record, S1) on the uncertain band;
  * resumable checkpoints (spot GPUs) and an input-length check.
Test records are never used for training.

  python src/ce.py export              # CPU: work/ce_groups.parquet (train groups), work/ce_score_{train,test}.parquet
  python src/ce.py train               # GPU
  python src/ce.py score --split train|test   # GPU -> work/{split}_ce.parquet (q, s, ce, ce_list[, ce_sym])
"""
import argparse
import functools
import json
import math
import os
import random
import re
import time

import numpy as np
import pandas as pd

MIN_P1 = 0.002
BAND_LO, BAND_HI = 0.005, 0.99
SYM_LO, SYM_HI = 0.02, 0.98
TOP_K = 8
SCORE_TOP = 3   # score each record's top-3 candidates (the top one always: fixes v3's skipped confident decoys)
GENERIC_ADD = ["services", "group", "center", "holdings", "partners", "enterprises", "solutions", "trading",
               "global", "international", "associates", "ventures", "industries", "consulting", "company"]
ABBR = {"limited": "ltd", "private": "pvt", "incorporated": "inc", "corporation": "corp", "company": "co",
        "services": "svcs", "street": "st", "road": "rd", "avenue": "ave", "and": "&"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def serialize(name, addr, nums, codes):
    s = f"[NAME] {name}"
    if addr:
        s += f" [ADDR] {addr}"
        if nums:
            s += f" [NUM] {nums}"
        if codes:
            s += f" [UNIT] {codes}"
    else:
        s += " [NO ADDRESS]"
    return s


# ---------------------------------------------------------------- synthetic variants (train records only)

def _positive(name, addr, rng):
    w = name.split()
    k = rng.integers(0, 5)
    if k == 0 and len(w) > 1:
        rng.shuffle(w)
    elif k == 1:
        w = [ABBR.get(x.lower(), x) for x in w]
    elif k == 2:
        addr = ""
    elif k == 3:
        w = [x + " LLC" if i == len(w) - 1 else x for i, x in enumerate(w)] if "llc" not in name.lower() else w
    else:
        w = [x.replace("a", "á", 1) if rng.random() < 0.3 else x for x in w]
    return " ".join(w), addr


def _decoy(name, addr, rng):
    k = rng.integers(0, 3)
    if k == 0 and re.search(r"\d+", addr):
        m = re.search(r"\d+", addr)
        v = int(m.group()) + int(rng.integers(1, 21)) * (1 if rng.random() < 0.5 else -1)
        return name, addr[:m.start()] + str(max(v, 1)) + addr[m.end():]
    if k == 1 and re.search(r"[A-Za-z]-?\d+", addr):
        m = re.search(r"([A-Za-z])-?(\d+)", addr)
        return name, addr[:m.start()] + m.group(1) + "-" + str(int(m.group(2)) + 1) + addr[m.end():]
    return name + " " + GENERIC_ADD[int(rng.integers(0, len(GENERIC_ADD)))].title(), addr


# ---------------------------------------------------------------- export (CPU)

def _text(df, i):
    return serialize(df.raw_name.values[i], df.raw_addr.values[i], df.nums.values[i], df.codes.values[i])


def export(args):
    from er import load_split  # the GPU box only needs torch + transformers
    rng = np.random.default_rng(0)
    s1, r = load_split(args.data, "train", args.work)
    d = pd.read_parquet(f"{args.work}/train_scored.parquet", columns=["q", "s", "p1", "y", "split"])
    # ---- training groups from split A (p1 is out-of-fold there): each record with its top-K prefilter survivors,
    # whatever their p1, so the listwise head sees real competitors
    a = d[d.split.values == "A"].sort_values(["q", "p1"], ascending=[True, False])
    a = a[a.groupby("q").cumcount().values < TOP_K]
    band_q = a[(a.p1 >= BAND_LO) & (a.p1 < BAND_HI)].q.unique()
    decoy_q = a[(a.y == 0) & (a.p1 >= 0.3)].q.unique()
    easy_q = a[(a.y == 1) & (a.p1 >= BAND_HI)].q.unique()
    easy_q = rng.choice(easy_q, size=min(len(easy_q), args.easy_groups), replace=False)
    qs = np.unique(np.concatenate([band_q, decoy_q, easy_q]))
    g = a[a.q.isin(set(qs))].copy()
    g["gid"] = g.q.astype(str)
    na = g[r.addr.values[g.q.values] == ""]
    g = pd.concat([g, na.assign(gid=na.gid + "_x2"), na.assign(gid=na.gid + "_x3")])  # no-address groups x3
    g["text_a"] = [_text(s1, i) for i in g.s.values]
    g["text_b"] = [_text(r, i) for i in g.q.values]
    if args.max_pairs and len(g) > args.max_pairs:  # cap training size (GPU time): keep whole groups, no-address copies included
        keep_g = pd.Series(g.gid.unique()).sample(frac=args.max_pairs / len(g), random_state=0)
        g = g[g.gid.isin(set(keep_g))]
    groups = [g[["gid", "text_a", "text_b", "y"]]]
    # ---- synthetic groups: variants of A's true records against the same candidate list
    pos = a[a.y.values == 1].sample(min(args.synthetic, int((a.y == 1).sum())), random_state=0)
    cand = a.groupby("q")
    rows = []
    for k, (q, s_true) in enumerate(zip(pos.q.values, pos.s.values)):
        name, addr = r.raw_name.values[q], r.raw_addr.values[q]
        is_pos = k % 2 == 0
        nn, na = (_positive if is_pos else _decoy)(name, addr, rng)
        tb = serialize(nn, na, " ".join(re.findall(r"\d+", na)), r.codes.values[q] if na == addr else "")
        c = cand.get_group(q)
        for s_, y_ in zip(c.s.values, c.y.values):
            rows.append((f"syn{k}", _text(s1, s_), tb, int(y_ == 1 and is_pos)))
    groups.append(pd.DataFrame(rows, columns=["gid", "text_a", "text_b", "y"]))
    out = pd.concat(groups, ignore_index=True)
    out.to_parquet(f"{args.work}/ce_groups.parquet")
    log(f"train groups {out.gid.nunique():,} ({len(out):,} pairs, positive share {out.y.mean():.3f}; "
        f"synthetic {len(rows):,} pairs)")
    export_scoring(args, ["train"])


def export_scoring(args, splits):
    """Pairs to score: held-out H + the stage-2 slice of B (train), and test."""
    from er import load_split
    s1, r = load_split(args.data, "train", args.work)
    for split in splits:
        s1x, rx = (s1, r) if split == "train" else load_split(args.data, "test", args.work)
        cols = ["q", "s", "p1"] + (["split"] if split == "train" else [])
        dx = pd.read_parquet(f"{args.work}/{split}_scored.parquet", columns=cols)
        dx = dx[dx.p1.values >= MIN_P1]
        dx = dx[dx.groupby("q").p1.rank(ascending=False, method="first").values <= SCORE_TOP]
        if split == "train":  # held-out H + the stage-2 slice of B (split A trained the cross-encoder)
            from er import STAGE2_B, id_bucket, load_truth
            _, r2s = load_truth(args.data)
            b = id_bucket(rx.entity_id.map(r2s).fillna(rx.entity_id).values)[dx.q.values]
            dx = dx[(dx.split.values == "H") | ((b >= STAGE2_B[0]) & (b < STAGE2_B[1]))]
        dx = dx[["q", "s", "p1"]].reset_index(drop=True)
        dx["text_a"] = [_text(s1x, i) for i in dx.s.values]
        dx["text_b"] = [_text(rx, i) for i in dx.q.values]
        dx.to_parquet(f"{args.work}/ce_score_{split}.parquet")
        log(f"{split}: {len(dx):,} pairs to score")


# ---------------------------------------------------------------- model helpers (GPU)

def _amp(torch, device):
    """bf16 autocast on GPU only; a no-op context on CPU (torch.autocast("cuda", enabled=False) still queries the GPU)."""
    import contextlib
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


@functools.lru_cache(None)
def _torch():
    import torch
    return torch


def _load(model_name, device, dtype=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": ["[NAME]", "[ADDR]", "[NUM]", "[UNIT]", "[NO ADDRESS]"]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=dtype)
    model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    return tok, model.to(device)


def _encode(tok, a, b, max_len, device):
    enc = tok(list(a), list(b), padding=True, truncation="longest_first", max_length=max_len, return_tensors="pt")
    return {k: v.to(device, non_blocking=True) for k, v in enc.items()}


def length_check(tok, df, max_len, n=20000):
    smp = df.sample(min(n, len(df)), random_state=0)
    lens = [len(x) for x in tok(list(smp.text_a), list(smp.text_b))["input_ids"]]
    share = float(np.mean(np.array(lens) > max_len))
    log(f"input length p50 {int(np.median(lens))}, p99 {int(np.quantile(lens, 0.99))}; share > {max_len}: {share:.3%}")
    return share


def train(args):
    torch = _torch()
    from transformers import get_linear_schedule_with_warmup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0)
    df = pd.read_parquet(f"{args.work}/ce_groups.parquet")
    tok, model = _load(args.model, device)
    if length_check(tok, df, args.max_len) > 0.02:
        args.max_len = 128
        log("more than 2% of inputs are truncated -> max_len 128")
    os.makedirs(args.out, exist_ok=True)
    json.dump({"max_len": args.max_len}, open(f"{args.out}/ce_config.json", "w"))  # scoring must use the same length
    gids = df.gid.unique()
    hold_g = set(pd.Series(gids).sample(frac=0.02, random_state=0))
    hold = df[df.gid.isin(hold_g)].reset_index(drop=True)
    trn = df[~df.gid.isin(hold_g)]
    groups = [g for _, g in trn.groupby("gid", sort=False)]
    none_logit = torch.nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.AdamW(list(model.parameters()) + [none_logit], lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(groups) / args.groups_per_batch) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    bce = torch.nn.BCEWithLogitsLoss()
    ck = f"{args.out}/ckpt.pt"
    step = 0
    if os.path.exists(ck):  # resume after a spot preemption
        state = torch.load(ck, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        none_logit.data = state["none"].to(device); step = state["step"]
        log(f"resumed from step {step:,}")
    model.train()
    log(f"train {len(groups):,} groups, {steps:,} steps, device {device}")
    rng = np.random.default_rng(0)
    order = np.concatenate([rng.permutation(len(groups)) for _ in range(args.epochs)])
    best_ap, t0 = -1.0, time.time()
    for i in range(step, steps):
        if args.max_steps and i >= args.max_steps:
            break
        batch = [groups[j] for j in order[i * args.groups_per_batch:(i + 1) * args.groups_per_batch]]
        flat = pd.concat(batch)
        enc = _encode(tok, flat.text_a.values, flat.text_b.values, args.max_len, device)
        with _amp(torch, device):
            logits = model(**enc).logits.float().squeeze(-1)
        y = torch.tensor(flat.y.values, dtype=torch.float32, device=device)
        loss = bce(logits, y)
        lw, o = [], 0
        for g in batch:  # listwise: softmax over the group's candidates + "none"
            n = len(g)
            z = torch.cat([logits[o:o + n], none_logit])
            tgt = int(np.argmax(g.y.values)) if g.y.values.max() == 1 else n
            lw.append(torch.nn.functional.cross_entropy(z.unsqueeze(0), torch.tensor([tgt], device=device)))
            o += n
        loss = loss + args.listwise_weight * torch.stack(lw).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step = i + 1
        if step % 100 == 0:
            log(f"  step {step:,}/{steps:,} loss {loss.item():.4f} ({step * len(flat) / (time.time() - t0):.0f} pairs/s)")
        if step % args.ckpt_every == 0:
            os.makedirs(args.out, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "none": none_logit.detach().cpu(), "step": step}, ck)
        if step % args.eval_every == 0 or step == steps or step == args.max_steps:
            ap = _evaluate(model, tok, hold, args, device)
            if ap > best_ap:
                best_ap = ap
                model.save_pretrained(args.out); tok.save_pretrained(args.out)
                torch.save(none_logit.detach().cpu(), f"{args.out}/none_logit.pt")
                log(f"  saved best (holdout AP {ap:.4f})")
    log(f"done; best holdout AP {best_ap:.4f}")


def _score_frame(model, tok, a, b, args, device):
    torch = _torch()
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), np.float32)
    with torch.inference_mode(), _amp(torch, device):
        for i in range(0, len(a), args.score_bs):
            idx = order[i:i + args.score_bs]
            out[idx] = model(**_encode(tok, a[idx], b[idx], args.max_len, device)).logits.float().squeeze(-1).cpu().numpy()
            if (i // args.score_bs) % 500 == 0:
                log(f"  scored {min(i + args.score_bs, len(a)):,}/{len(a):,}")
    return out


def _evaluate(model, tok, df, args, device):
    from sklearn.metrics import average_precision_score
    s = _score_frame(model, tok, df.text_a.values, df.text_b.values, args, device)
    model.train()
    return average_precision_score(df.y.values, s)


def score(args):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.out)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.out, torch_dtype=torch.bfloat16 if device == "cuda" else None).to(device)
    none = float(torch.load(f"{args.out}/none_logit.pt")) if os.path.exists(f"{args.out}/none_logit.pt") else 0.0
    if os.path.exists(f"{args.out}/ce_config.json"):
        args.max_len = json.load(open(f"{args.out}/ce_config.json"))["max_len"]
    d = pd.read_parquet(f"{args.work}/ce_score_{args.split}.parquet")
    log(f"scoring {len(d):,} {args.split} pairs")
    d["ce"] = _score_frame(model, tok, d.text_a.values, d.text_b.values, args, device)
    band = (d.p1.values >= SYM_LO) & (d.p1.values < SYM_HI)
    sym = _score_frame(model, tok, d.text_b.values[band], d.text_a.values[band], args, device)
    d["ce_sym"] = d.ce.values
    d.loc[band, "ce_sym"] = (d.ce.values[band] + sym) / 2
    z = np.exp(d.ce_sym.values - d.groupby("q").ce_sym.transform("max").values)  # listwise: P(this S1 | record)
    den = d.assign(z=z).groupby("q").z.transform("sum").values + np.exp(none - d.groupby("q").ce_sym.transform("max").values)
    d["ce_list"] = z / den
    t = args.tag  # "" for bge-reranker, "2" for the second cross-encoder (Qwen3-Reranker)
    out = d[["q", "s", "ce", "ce_sym", "ce_list"]].rename(columns={"ce": f"ce{t}", "ce_sym": f"ce{t}_sym",
                                                                    "ce_list": f"ce{t}_list"})
    out.to_parquet(f"{args.work}/{args.split}_ce{t}.parquet")
    log(f"wrote {args.work}/{args.split}_ce{t}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "export_test", "train", "score"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="work/ce_model")
    ap.add_argument("--easy-groups", type=int, default=60000)
    ap.add_argument("--synthetic", type=int, default=60000)
    ap.add_argument("--max-pairs", type=int, default=0, help="optional cap on training pairs (0 = no cap: accuracy first)")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--groups-per-batch", type=int, default=16)
    ap.add_argument("--score-bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--listwise-weight", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--tag", default="", help='"" = first cross-encoder (ce*), "2" = second (ce2*)')
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"export": export, "export_test": lambda a: export_scoring(a, ["test"]), "train": train, "score": score}[args.cmd](args)
