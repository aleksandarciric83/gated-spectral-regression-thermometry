"""v10 follow-up phase 3: greedy forward selection path (always add the best gate, even if the error rises), GPR, unit variance, real LOTO.
usage: subset_greedy_v10.py wi start-gates max_size. Output v10_out/subset/greedy_part{wi}.jsonl"""
import os, sys, time, json, mimetypes
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2, lt2f
MODELS = ["GPR", "Bayesian Ridge (poly3 PCs)"]
wi = int(sys.argv[1]); start = tuple(sorted(int(x) for x in sys.argv[2].split(","))); maxsize = int(sys.argv[3])
OUT = HERE / "v10_out" / "subset" / f"greedy_part{wi}.jsonl"
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); cube = np.stack([s["rate"] for s in sp])
cache = {}
for pat in ("subset_part*.jsonl", "search_part*.jsonl", "greedy_part*.jsonl"):
    for p in (HERE / "v10_out" / "subset").glob(pat):
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            cache[tuple(r["gates"])] = float(np.sqrt(np.mean((np.array(r["pred"]["GPR"]) - temps) ** 2)))
fh = open(OUT, "a")
def ev(g):
    if g in cache: return cache[g]
    X = lt2.snv_rows(cube[:, :, list(g)].reshape(len(temps), -1)); p = lt2f.real_loto(X, temps, "z", MODELS)
    fh.write(json.dumps(dict(gates=list(g), pred={m: v.tolist() for m, v in p.items()}, start=list(start))) + "\n"); fh.flush()
    cache[g] = float(np.sqrt(np.mean((p["GPR"] - temps) ** 2))); return cache[g]
cur = start; ev(cur); t0 = time.time()
while len(cur) < maxsize:
    cands = [tuple(sorted(cur + (x,))) for x in range(40) if x not in cur]
    sc = [(ev(c), c) for c in cands]; f, cur = min(sc)
    print(wi, "size", len(cur), cur, round(f, 3), round(time.time() - t0), flush=True)
print("done", wi, flush=True)
