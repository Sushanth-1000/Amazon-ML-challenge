"""v6.2b scoring shard: score /opt/ml/input/data/pairs/ce_score_<split>.parquet with the model in channel 'model'."""
import glob, os, shutil, subprocess, sys
M, E = "/opt/ml/input/data/model", "/tmp/ev"
os.makedirs(E, exist_ok=True)
src = glob.glob("/opt/ml/input/data/pairs/ce_score_*.parquet")[0]
split = "train" if "ce_score_train" in src else "test"
shutil.copy(src, f"{E}/ce_score_{split}.parquet")
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, f"{here}/ce.py", "score", "--split", split, "--work", E, "--out", M, "--score-bs", "1024"], check=True)
shutil.copy(f"{E}/{split}_ce.parquet", f"/opt/ml/model/{split}_ce.parquet")
print("V62B SCORE DONE", flush=True)
