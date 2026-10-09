"""v6.2b cross-encoder: E20's recipe (bge-reranker-v2-m3, 12,000 steps, 16 groups/batch) on v6.1's real US/India groups +
French pseudo-labelled groups. Model + checkpoints in /opt/ml/checkpoints/m (synced to S3; spot interruptions resume)."""
import os, shutil, subprocess, sys
W, CK = "/tmp/work", "/opt/ml/checkpoints/m"
os.makedirs(W, exist_ok=True); os.makedirs(CK, exist_ok=True)
shutil.copy("/opt/ml/input/data/train/ce_groups.parquet", W)
here = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f"{CK}/DONE"):
    subprocess.run([sys.executable, f"{here}/ce.py", "train", "--work", W, "--model", "BAAI/bge-reranker-v2-m3", "--out", CK,
                    "--groups-per-batch", "16", "--max-steps", "12000", "--eval-every", "12000", "--ckpt-every", "2000", "--ckpt-secs", "300"], check=True)
    open(f"{CK}/DONE", "w").write("trained")
for f in os.listdir(CK):                                  # model files (not the resume checkpoint) -> model artifact
    if f != "ckpt.pt":
        p = f"{CK}/{f}"
        (shutil.copytree if os.path.isdir(p) else shutil.copy)(p, f"/opt/ml/model/{f}")
print("V62B TRAIN DONE", flush=True)
