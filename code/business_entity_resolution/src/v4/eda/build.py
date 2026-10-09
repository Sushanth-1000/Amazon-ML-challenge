"""EDA working set (fits a laptop): full normalized train S1 + a 3% id-hash sample of S2/S3 records
(train and test), raw strings kept, plus test S1 in full. Written to work_eda/*.parquet."""
import os, sys, zlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from er import _normalize, read_tsv, log

D, W, PCT = "dataset", "work_eda", 3
os.makedirs(W, exist_ok=True)

def sampled(path):
    out = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=1_000_000):
        b = np.fromiter((zlib.crc32(i.encode()) % 100 for i in ch.entity_id), np.int16, len(ch))
        out.append(ch[b < PCT].assign(row=ch.index[b < PCT]))
    return pd.concat(out, ignore_index=True)

def norm_keep_raw(df):
    n = _normalize(df)
    return pd.concat([df.reset_index(drop=True), n.drop(columns=["entity_id", "country"]).reset_index(drop=True)], axis=1)

if __name__ == "__main__":
    for split in ("train", "test"):
        s1 = read_tsv(f"{D}/{split}/{split}_source1.tsv")
        s1["row"] = s1.index
        norm_keep_raw(s1).to_parquet(f"{W}/{split}_s1.parquet"); del s1
        q = pd.concat([sampled(f"{D}/{split}/{split}_source{i}.tsv") for i in (2, 3)], ignore_index=True)
        norm_keep_raw(q).to_parquet(f"{W}/{split}_q.parquet")
        log(split, "done", len(q), "sampled queries")
