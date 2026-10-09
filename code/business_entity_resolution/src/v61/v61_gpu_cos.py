"""v6.1 GPU step 2 (H100): bge-m3 cosine of every candidate pair, from the embeddings of step 1 (as E15: fp16, stored
float16). Inputs: <work>/{train,test}_qs.npy (2 x N int32: record row, S1 row; order of {split}_pairs61.parquet) and
<work>/train_copyqs.npy (US training pairs whose French-style copies are used; embeddings of the rewritten text).
Usage: python v61_gpu_cos.py --emb ~/v61emb --work ~/v61cos --what train|test|copies"""
import argparse
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--what", choices=["train", "test", "copies"], required=True)
args = ap.parse_args()
t0 = time.time()
E = lambda name: torch.from_numpy(np.load(f"{args.emb}/{name}.npy")).cuda()
if args.what == "copies":
    qs = np.load(f"{args.work}/train_copyqs.npy").astype(np.int64)
    rrows, srows = np.load(f"{args.emb}/train_r_fr_rows.npy"), np.load(f"{args.emb}/train_s1_fr_rows.npy")
    q, s = np.searchsorted(rrows, qs[0]), np.searchsorted(srows, qs[1])
    assert (rrows[q] == qs[0]).all() and (srows[s] == qs[1]).all(), "copy pair outside the rewritten US rows"
    R, S, out = E("train_r_fr"), E("train_s1_fr"), f"{args.work}/train_cos_fr.npy"
else:
    qs = np.load(f"{args.work}/{args.what}_qs.npy").astype(np.int64)
    q, s = qs[0], qs[1]
    R, S, out = E(f"{args.what}_r"), E(f"{args.what}_s1"), f"{args.work}/{args.what}_cos.npy"
cos = np.empty(len(q), np.float16)
qt, st = torch.from_numpy(q).cuda(), torch.from_numpy(s).cuda()
for a in range(0, len(q), 2_000_000):
    cos[a:a + 2_000_000] = (R[qt[a:a + 2_000_000]] * S[st[a:a + 2_000_000]]).sum(1).float().cpu().numpy()
np.save(out, cos)
print(f"{args.what}: {len(q):,} cosines in {time.time() - t0:.0f}s, mean {float(cos.astype(np.float32).mean()):.3f} -> {out}", flush=True)
