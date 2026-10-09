"""v6.1 GPU step 1 (H100): bge-m3 (MIT) embeddings + dense candidate channel.
  * embeds "raw_name | raw_addr" of every S1 and record of train and test (fp16, normalized, max 64 tokens: as E13/E15)
  * dense channel: each record's top-20 S1 of its own country by cosine -> <out>/{split}_dense.parquet (r_id, s1_id, cos),
    the format er.load_dense() reads (exactly E13's channel)
  * French-style copies (E14 recipe, same rewrite functions and seeds) of the US train records and S1 -> embeddings too,
    so the copies' similarity features are computed on their own text
Embeddings stay on this machine (<emb>/*.npy) for the later pair-cosine step. Usage:
  python v61_gpu_embed.py --work ~/w --emb ~/v61emb --out ~/v61out"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=20)
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frstyle import fr_addr, fr_name  # noqa: E402  (E14's rewrite, unchanged)

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
os.makedirs(args.emb, exist_ok=True); os.makedirs(args.out, exist_ok=True)
m = SentenceTransformer("BAAI/bge-m3", device="cuda"); m.half(); m.max_seq_length = 64
txt = lambda names, addrs: [f"{n} | {a}" for n, a in zip(names, addrs)]


def embed(texts, path):
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    e = m.encode(texts, batch_size=2048, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    e = e.astype(np.float16); np.save(path, e)
    return e


cols = ["entity_id", "country", "raw_name", "raw_addr"]
for split in ("train", "test"):
    s1 = pd.read_parquet(f"{args.work}/{split}_s1.parquet", columns=cols)
    r = pd.read_parquet(f"{args.work}/{split}_r.parquet", columns=cols)
    for d in (s1, r):
        d["raw_name"], d["raw_addr"] = d.raw_name.fillna(""), d.raw_addr.fillna("")
    es = embed(txt(s1.raw_name, s1.raw_addr), f"{args.emb}/{split}_s1.npy")
    log(f"{split} S1 embedded {len(s1):,}")
    er_ = embed(txt(r.raw_name, r.raw_addr), f"{args.emb}/{split}_r.npy")
    log(f"{split} records embedded {len(r):,}")
    parts = []
    for c in r.country.unique():
        si, qi = np.flatnonzero(s1.country.values == c), np.flatnonzero(r.country.values == c)
        if not len(si) or not len(qi):
            continue
        S = torch.from_numpy(np.ascontiguousarray(es[si])).cuda()
        for a in range(0, len(qi), 2048):                          # 2048 x 1.3M fp16 block = 5.4 GB
            Q = torch.from_numpy(np.ascontiguousarray(er_[qi[a:a + 2048]])).cuda()
            v, i = (Q @ S.T).topk(args.k, dim=1)                  # top-k in fp16 (as E13's fp16 embeddings)
            v = v.float()
            parts.append(pd.DataFrame({"r_id": np.repeat(r.entity_id.values[qi[a:a + 2048]], args.k),
                                       "s1_id": s1.entity_id.values[si[i.cpu().numpy().ravel()]],
                                       "cos": v.cpu().numpy().ravel().astype(np.float32)}))
        del S
        log(f"{split} {c}: dense top-{args.k} for {len(qi):,} records")
    pd.concat(parts, ignore_index=True).to_parquet(f"{args.out}/{split}_dense.parquet")
    log(f"{split}_dense.parquet written")
    if split == "train":  # French-style copies of the US train rows (E14: records seed i*2+1, S1 seed i*2)
        for name, d, is_r in (("r", r, True), ("s1", s1, False)):
            rows = np.flatnonzero(d.country.values == "US")
            names = [fr_name(d.raw_name.values[i], i * 2 + is_r) for i in rows]
            addrs = [fr_addr(d.raw_addr.values[i], i * 3 + is_r) for i in rows]
            np.save(f"{args.emb}/train_{name}_fr_rows.npy", rows)
            embed(txt(names, addrs), f"{args.emb}/train_{name}_fr.npy")
            log(f"train {name} French-style embedded {len(rows):,}")
log("GPU step 1 done")
