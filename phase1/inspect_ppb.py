import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download("proteinea/ppb_affinity", "filtered.csv", repo_type="dataset",
                       local_dir=os.path.expanduser("~/Projects/ppi-data/affinity/ppb"))
df = pd.read_csv(path)
print("file:", path, flush=True)
print("rows:", len(df), flush=True)
print("columns:", list(df.columns), flush=True)
print("--- first row (values truncated) ---", flush=True)
for c in df.columns:
    print(f"  {c} = {str(df[c].iloc[0])[:100]}", flush=True)
