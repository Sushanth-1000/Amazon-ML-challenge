"""Full-scale paired held-out test: v4 (A) vs v6.1 (B) on held-out half B, per-S1 F0.5, bootstrap 95% CI, by country.
Aborts if v6.1's pair indices do not line up with v4's record tables (labels must match the ground truth exactly)."""
import os, subprocess, sys, zipfile, shutil
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
IN, OUT = "/opt/ml/input/data", "/opt/ml/model"
W, A, B = "/tmp/work", "/tmp/runA_v4", "/tmp/runB_v61"
for d in (W, A, B): os.makedirs(d, exist_ok=True)
zipfile.ZipFile(f"{IN}/data/student_resource.zip").extractall("/tmp/sr")
DS = "/tmp/sr/student_resource/dataset"
for f in ("train_s1", "train_r"): os.symlink(f"{IN}/data/{f}.parquet", f"{W}/{f}.parquet")
os.symlink(f"{IN}/data/train_stage2.parquet", f"{A}/train_stage2.parquet"); shutil.copy(f"{IN}/data/meta_v4.json", f"{A}/meta.json")
os.symlink(f"{IN}/v61/train_stage2.parquet", f"{B}/train_stage2.parquet"); shutil.copy(f"{IN}/v61/meta.json", f"{B}/meta.json")
import er
s1, r = er.load_split(DS, "train", W); truth, _ = er.load_truth(DS)
sid, rid = s1.entity_id.values, r.entity_id.values
for name, run in (("v4", A), ("v6.1", B)):
    d = pd.read_parquet(f"{run}/train_stage2.parquet", columns=["q", "s", "y"])
    k = np.random.default_rng(1).choice(len(d), size=min(2_000_000, len(d)), replace=False)
    yt = np.array([rid[q] in truth.get(sid[s], ()) for q, s in zip(d.q.values[k], d.s.values[k])])
    agree = (yt == d.y.values[k].astype(bool)).mean()
    print(f"ALIGNMENT {name}: labels match ground truth on {agree:.6f} of {len(k):,} sampled pairs", flush=True)
    if agree < 0.9999:
        print("ABORT: pair indices do not line up with the record tables -> comparison would be invalid", flush=True); sys.exit(1)
out = subprocess.run([sys.executable, f"{here}/cmp_heldout.py", "--data", DS, "--work", W, "--a", A, "--b", B],
                     capture_output=True, text=True)
print(out.stdout, out.stderr[-3000:], flush=True)
open(f"{OUT}/cmp_v61_vs_v4.txt", "w").write(out.stdout + "\n" + out.stderr[-3000:])
if out.returncode: sys.exit(out.returncode)
print("CMP DONE", flush=True)
