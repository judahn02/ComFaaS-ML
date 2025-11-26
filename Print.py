import time, json, os, pandas as pd, random, numpy as np, torch

def log(msg: str): print(msg, flush=True)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class Timer:
    def __init__(self): self.t = time.time()
    def lap(self):
        now = time.time(); dt = now - self.t; self.t = now; return dt

def save_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True); df.to_csv(path, index=False)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

