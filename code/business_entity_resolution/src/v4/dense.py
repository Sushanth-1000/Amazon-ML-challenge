"""Dense (embedding) blocking channel with BAAI/bge-m3 (MIT license, multilingual).

TF-IDF channels miss true pairs that share almost no characters: a name written in another script,
a renamed business at the same place, a heavily rewritten address. bge-m3 embeds the raw record text
("name | address") into one multilingual vector space, so "बॉम्बे एस्टेट" lands near "Bombay Estate".
For each query record we keep the top-k most similar S1 entities of the same country; er.py adds them
to the candidate set as an extra channel (load_dense), with their cosine as feature `sc_dense`.

Usage (GPU, from student_resource/):
    python src/dense.py --split train --queries nonlatin     # -> work/train_dense.parquet
    python src/dense.py --split test  --queries nonlatin     # -> work/test_dense.parquet
Only the kept training queries (work/train_keep.npy) are embedded for the train split.
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd

NONLATIN = re.compile(r"[^\x00-ɏ]")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def embed(texts, tok, model, args, device):
    import torch
    order = np.argsort([len(t) for t in texts])
    out = np.empty((len(texts), model.config.hidden_size), np.float16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        for i in range(0, len(texts), args.bs):
            idx = order[i:i + args.bs]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(device)
            v = model(**enc).last_hidden_state[:, 0]  # bge-m3 dense vector = CLS token
            out[idx] = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy().astype(np.float16)
            if (i // args.bs) % 200 == 0:
                log(f"  embedded {min(i + args.bs, len(texts)):,}/{len(texts):,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--queries", choices=["nonlatin", "all"], default="nonlatin")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = f"{args.data}/{args.split}/{args.split}_source"
    s1 = read_tsv(d + "1.tsv")
    r = pd.concat([read_tsv(d + f"{i}.tsv") for i in (2, 3)], ignore_index=True)
    if args.split == "train" and os.path.exists(f"{args.work}/train_keep.npy"):
        r = r.iloc[np.load(f"{args.work}/train_keep.npy")].reset_index(drop=True)
    if args.queries == "nonlatin":
        r = r[r.business_name.str.contains(NONLATIN)].reset_index(drop=True)
    text = lambda df: (df.business_name + " | " + df.business_address).tolist()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16 if device == "cuda" else None).to(device).eval()
    parts = []
    for c in sorted(r.country.unique()):
        sc, rc = s1[s1.country == c].reset_index(drop=True), r[r.country == c].reset_index(drop=True)
        if not len(sc) or not len(rc):
            continue
        log(f"{c}: {len(rc):,} queries x {len(sc):,} S1")
        S = torch.from_numpy(embed(text(sc), tok, model, args, device)).to(device)
        Q = embed(text(rc), tok, model, args, device)
        k = min(args.k, len(sc))
        for a in range(0, len(rc), args.chunk):
            q = torch.from_numpy(Q[a:a + args.chunk]).to(device)
            sim = (q @ S.T).float() if device == "cuda" else (q.float() @ S.float().T)
            v, i = sim.topk(k, dim=1)
            v, i = v.cpu().numpy(), i.cpu().numpy()
            parts.append(pd.DataFrame({"r_id": np.repeat(rc.entity_id.values[a:a + args.chunk], k),
                                       "s1_id": sc.entity_id.values[i.ravel()], "cos": v.ravel().astype(np.float32)}))
        del S
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(args.work, exist_ok=True)
    path = f"{args.work}/{args.split}_dense.parquet"
    out.to_parquet(path)
    log(f"wrote {path}: {len(out):,} pairs")


if __name__ == "__main__":
    main()
