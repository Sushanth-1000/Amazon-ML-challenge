"""E15 step 1 (GPU): bge-m3 (MIT) embeddings of every train S1 and record ("name | address"), then the cosine of every
full-scale leave-one-country-out candidate pair (loco_pairs_2_100: 66,967,430 pairs, same order). No labels used.
Usage: python e15_embed.py <work>  -> <work>/e15_cos.npy (float16, one value per pair)"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

W = sys.argv[1]
t0 = time.time()
s1 = pd.read_parquet(f"{W}/train_s1.parquet", columns=["raw_name", "raw_addr"])
r = pd.read_parquet(f"{W}/train_r.parquet", columns=["raw_name", "raw_addr"])
p = pd.read_parquet(f"{W}/loco_pairs_2_100.parquet", columns=["q", "s"])
m = SentenceTransformer("BAAI/bge-m3", device="cuda"); m.half(); m.max_seq_length = 64
txt = lambda d: (d.raw_name.fillna("") + " | " + d.raw_addr.fillna("")).tolist()
emb = {}
for name, d in (("s1", s1), ("r", r)):
    e = m.encode(txt(d), batch_size=2048, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    emb[name] = torch.tensor(e.astype(np.float16), device="cuda")
    print(f"{name}: {len(d):,} embedded at {time.time() - t0:.0f}s", flush=True)
cos = np.empty(len(p), np.float16)
q, s = torch.tensor(p.q.values, device="cuda"), torch.tensor(p.s.values, device="cuda")
for a in range(0, len(p), 2_000_000):
    cos[a:a + 2_000_000] = (emb["r"][q[a:a + 2_000_000]] * emb["s1"][s[a:a + 2_000_000]]).sum(1).float().cpu().numpy()
np.save(f"{W}/e15_cos.npy", cos)
print(f"{len(p):,} pair cosines in {time.time() - t0:.0f}s; mean {float(cos.astype(np.float32).mean()):.3f}", flush=True)
