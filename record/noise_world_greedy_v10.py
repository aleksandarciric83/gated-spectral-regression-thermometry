"""Noise-world greedy forward selection from gate 0 (always add the best gate) up to size 6. usage: noise_world_greedy_v10.py tau seed"""
import os, sys, time, json, mimetypes, itertools
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2, lt2f
MODELS = ["GPR", "Bayesian Ridge (poly3 PCs)"]
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n = len(temps)
S = np.load(HERE / "Ssyn_v10.npy"); R = np.load(HERE / "Rsyn_v10.npy")
def world(tau, seed):
    s = np.ones(n) if seed < 0 else np.random.default_rng(seed).choice([-1, 1], n)
    return S + tau * s[:, None, None] * R
def evaluate(Xw, g):
    X = lt2.snv_rows(Xw[:, :, list(g)].reshape(n, -1)); return lt2f.real_loto(X, temps, "z", MODELS)

tau, seed = float(sys.argv[1]), int(sys.argv[2]); OUT = HERE / "v10_out" / "noise" / f"greedy_tau{tau}_seed{seed}.jsonl"; OUT.parent.mkdir(exist_ok=True)
Xw = world(tau, seed); cache = {}
if OUT.exists():
    for l in open(OUT):
        try: d = json.loads(l); cache[tuple(d["gates"])] = float(np.sqrt(np.mean((np.array(d["pred"]["GPR"]) - temps) ** 2)))
        except Exception: pass
fh = open(OUT, "a")
def ev(g):
    if g in cache: return cache[g]
    p = evaluate(Xw, g); fh.write(json.dumps(dict(tau=tau, seed=seed, gates=list(g), pred={m: v.tolist() for m, v in p.items()})) + "\n"); fh.flush()
    cache[g] = float(np.sqrt(np.mean((p["GPR"] - temps) ** 2))); return cache[g]
cur = (0,); ev(cur); t0 = time.time()
while len(cur) < 6:
    sc = [(ev(tuple(sorted(cur + (x,)))), tuple(sorted(cur + (x,)))) for x in range(40) if x not in cur]; f, cur = min(sc)
    print("size", len(cur), cur, round(f, 3), round(time.time() - t0), flush=True)
print("done", flush=True)
