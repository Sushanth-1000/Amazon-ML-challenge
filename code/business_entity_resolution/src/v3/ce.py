"""Cross-encoder re-scoring of uncertain candidate pairs (fine-tuned BAAI/bge-reranker-v2-m3, Apache-2.0).

The stage-1 LightGBM compares normalized strings; the cross-encoder reads BOTH raw records together
(original script, no normalization), so it can use meaning ("Pvt Ltd" = "Private Limited", Hindi
"बॉम्बे" = "Bombay", same name but house number 155 vs 208 = decoy). It only scores the pairs whose
stage-1 probability is uncertain; its logit becomes a stage-2 feature.

  export : write the pairs to score / train on, with raw text     (CPU)
             train: out-of-fold stage-1 band [BAND_LO, BAND_HI) + top-ranked decoys + a sample of easy pairs
             val / test: every pair stage 2 will see with p1 in [MIN_P1, SKIP_HI)
  train  : fine-tune (pointwise BCE on the single relevance logit)  (GPU)
  score  : logits for work/ce_<split>_pairs.parquet -> work/<split>_ce.parquet  (GPU)

Usage (from student_resource/):
    python src/ce.py export --split train|val|test
    python src/ce.py train  --model BAAI/bge-reranker-v2-m3 --out work/ce_model
    python src/ce.py score  --split val|test --model work/ce_model
"""
import argparse
import functools
import math
import os
import time

import numpy as np
import pandas as pd

BAND_LO, BAND_HI = 0.005, 0.99   # training band on out-of-fold stage-1 probability
MIN_P1, SKIP_HI = 0.005, 0.999  # scored at val/test time (pairs above SKIP_HI are near-certain)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _text(name, addr):
    return f"{name} | {addr}" if addr else name


# ---------------------------------------------------------------- export (CPU)

def export(args):
    from er import load_split  # local import: the GPU box only needs torch + transformers for train/score
    split = "test" if args.split == "test" else "train"
    s1, r = load_split(args.data, split, args.work)
    if split == "train":
        r = r.iloc[np.load(f"{args.work}/train_keep.npy")].reset_index(drop=True)
    rng = np.random.default_rng(0)
    if args.split == "train":
        d = pd.read_parquet(f"{args.work}/train_oof.parquet")
        rank = d.groupby("q").p1.rank(ascending=False, method="first").values
        band = (d.p1.values >= BAND_LO) & (d.p1.values < BAND_HI)
        decoy = (d.y.values == 0) & (rank <= 3) & (d.p1.values >= 0.005)
        easy_pos = (d.y.values == 1) & (d.p1.values >= BAND_HI) & (rng.random(len(d)) < 0.2)
        easy_neg = (d.y.values == 0) & (rank <= 10) & (d.p1.values < BAND_LO) & (rng.random(len(d)) < 0.05)
        d = d[band | decoy | easy_pos | easy_neg]
        if len(d) > args.max_pairs:
            d = d.sample(args.max_pairs, random_state=0)
        d = d.rename(columns={"y": "label"})[["q", "s", "p1", "label", "fold"]]
    else:
        src = "val_pred" if args.split == "val" else "test_pred"
        d = pd.read_parquet(f"{args.work}/{src}.parquet")
        d = d[(d.prob.values >= MIN_P1) & (d.prob.values < SKIP_HI)]
        d = d.rename(columns={"prob": "p1"})[["q", "s", "p1"] + (["y"] if "y" in d.columns else [])]
    d = d.reset_index(drop=True)
    d["text_a"] = [_text(n, a) for n, a in zip(s1.raw_name.values[d.s.values], s1.raw_addr.values[d.s.values])]
    d["text_b"] = [_text(n, a) for n, a in zip(r.raw_name.values[d.q.values], r.raw_addr.values[d.q.values])]
    path = f"{args.work}/ce_{args.split}_pairs.parquet"
    d.to_parquet(path)
    pos = d.label.mean() if "label" in d.columns else float("nan")
    log(f"exported {len(d):,} pairs -> {path} (positive share {pos:.3f})")


# ---------------------------------------------------------------- model helpers (GPU)

def _batches(n, bs, order):
    for i in range(0, n, bs):
        yield order[i:i + bs]


def _encode(tok, a, b, max_len, device):
    enc = tok(list(a), list(b), padding=True, truncation="longest_first", max_length=max_len, return_tensors="pt")
    return {k: v.to(device, non_blocking=True) for k, v in enc.items()}


def _load(model_name, device, dtype=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=dtype)
    return tok, model.to(device)


@functools.lru_cache(None)
def _torch():
    import torch
    return torch


def evaluate(model, tok, df, args, device):
    torch = _torch()
    from sklearn.metrics import average_precision_score, roc_auc_score
    logits = score_frame(model, tok, df, args, device)
    y = df.label.values
    model.train()
    return roc_auc_score(y, logits), average_precision_score(y, logits)


def score_frame(model, tok, df, args, device):
    torch = _torch()
    model.eval()
    lens = (df.text_a.str.len() + df.text_b.str.len()).values
    order = np.argsort(lens)  # similar lengths per batch = little padding
    out = np.empty(len(df), np.float32)
    a, b = df.text_a.values, df.text_b.values
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for i, idx in enumerate(_batches(len(df), args.score_bs, order)):
            logits = model(**_encode(tok, a[idx], b[idx], args.max_len, device)).logits.float().squeeze(-1)
            out[idx] = logits.cpu().numpy()
            if i % 500 == 0:
                log(f"  scored {min((i + 1) * args.score_bs, len(df)):,}/{len(df):,}")
    return out


# ---------------------------------------------------------------- train (GPU)

def train(args):
    torch = _torch()
    from transformers import get_linear_schedule_with_warmup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d = pd.read_parquet(f"{args.work}/ce_train_pairs.parquet")
    # early-stopping / checkpoint-selection holdout from the TRAINING band (never the validation set)
    rec = pd.util.hash_pandas_object(d.q, index=False).values % 100
    hold, trn = d[rec < 2].reset_index(drop=True), d[rec >= 2].reset_index(drop=True)
    if len(hold) > args.eval_n:
        hold = hold.sample(args.eval_n, random_state=0).reset_index(drop=True)
    tok, model = _load(args.model, device)
    model.train()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(trn) / args.bs) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    log(f"train {len(trn):,} pairs, holdout {len(hold):,}, {steps:,} steps, device {device}")
    step, best_ap, t0 = 0, -1.0, time.time()
    a, b, y = trn.text_a.values, trn.text_b.values, trn.label.values.astype(np.float32)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        for idx in _batches(len(trn), args.bs, rng.permutation(len(trn))):
            enc = _encode(tok, a[idx], b[idx], args.max_len, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits.float().squeeze(-1)
            loss = lossf(logits, torch.from_numpy(y[idx]).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                rate = step * args.bs / (time.time() - t0)
                log(f"  step {step:,}/{steps:,} loss {loss.item():.4f} ({rate:.0f} pairs/s)")
            if step % args.eval_every == 0 or step == steps:
                auc, ap = evaluate(model, tok, hold, args, device)
                log(f"  eval step {step:,}: holdout AUC {auc:.4f} AP {ap:.4f}")
                if ap > best_ap:
                    best_ap = ap
                    model.save_pretrained(args.out); tok.save_pretrained(args.out)
                    log(f"  saved best -> {args.out}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    if step % args.eval_every and step != steps:  # stopped early: evaluate the last state too
        auc, ap = evaluate(model, tok, hold, args, device)
        log(f"  eval step {step:,}: holdout AUC {auc:.4f} AP {ap:.4f}")
        if ap > best_ap:
            best_ap = ap
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
            log(f"  saved best -> {args.out}")
    log(f"done; best holdout AP {best_ap:.4f}")


def score(args):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = _load(args.model, device, torch.bfloat16 if device == "cuda" else None)
    d = pd.read_parquet(f"{args.work}/ce_{args.split}_pairs.parquet")
    log(f"scoring {len(d):,} {args.split} pairs")
    d["ce"] = score_frame(model, tok, d, args, device)
    d[["q", "s", "ce"]].to_parquet(f"{args.work}/{args.split}_ce.parquet")
    if "y" in d.columns:
        from sklearn.metrics import average_precision_score, roc_auc_score
        log(f"{args.split}: AUC {roc_auc_score(d.y, d.ce):.4f} AP {average_precision_score(d.y, d.ce):.4f} "
            f"(stage-1 p1 on same pairs: AUC {roc_auc_score(d.y, d.p1):.4f} AP {average_precision_score(d.y, d.p1):.4f})")
    log(f"wrote {args.work}/{args.split}_ce.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "train", "score"])
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="work/ce_model")
    ap.add_argument("--max-pairs", type=int, default=1_000_000)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--score-bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--eval-n", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    {"export": export, "train": train, "score": score}[args.cmd](args)
