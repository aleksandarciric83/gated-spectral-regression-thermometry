"""v10 follow-up phase 2: simulated-annealing search over ANY gate subset (size 1-12) for GPR, unit variance, real LOTO.
usage: subset_search_v10.py wi seed budget start-gates(comma list). Output v10_out/subset/search_part{wi}.jsonl"""
import os, sys, time, json, mimetypes, math, random
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2, lt2f
MODELS = ["GPR", "Bayesian Ridge (poly3 PCs)"]
wi, seed, budget = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); start = tuple(sorted(int(x) for x in sys.argv[4].split(",")))
OUT = HERE / "v10_out" / "subset" / f"search_part{wi}.jsonl"
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); cube = np.stack([s["rate"] for s in sp])
cache = {}
for p in (HERE / "v10_out" / "subset").glob("subset_part*.jsonl"):
    for l in open(p):
        try: r = json.loads(l)
        except Exception: continue
        cache[tuple(r["gates"])] = float(np.sqrt(np.mean((np.array(r["pred"]["GPR"]) - temps) ** 2)))
if OUT.exists():
    for l in open(OUT):
        try: r = json.loads(l)
        except Exception: continue
        cache[tuple(r["gates"])] = float(np.sqrt(np.mean((np.array(r["pred"]["GPR"]) - temps) ** 2)))
fh = open(OUT, "a"); rng = random.Random(seed); n_new = 0
def ev(g):
    global n_new
    if g in cache: return cache[g]
    X = lt2.snv_rows(cube[:, :, list(g)].reshape(len(temps), -1)); p = lt2f.real_loto(X, temps, "z", MODELS)
    fh.write(json.dumps(dict(gates=list(g), pred={m: v.tolist() for m, v in p.items()}, start=list(start), seed=seed)) + "\n"); fh.flush()
    cache[g] = float(np.sqrt(np.mean((p["GPR"] - temps) ** 2))); n_new += 1; return cache[g]
def neighbour(g):
    s = set(g); mv = rng.choice(["add", "remove", "swap", "swap"])
    if mv == "add" and len(s) < 12: s.add(rng.choice([x for x in range(40) if x not in s]))
    elif mv == "remove" and len(s) > 1: s.remove(rng.choice(sorted(s)))
    elif len(s) >= 1:
        s.remove(rng.choice(sorted(s))); s.add(rng.choice([x for x in range(40) if x not in s]))
    return tuple(sorted(s))
cur = start; fcur = ev(cur); best, fbest = cur, fcur; t0 = time.time(); it = 0
while n_new < budget and it < 20 * budget:
    it += 1; T = 0.15 * (1 - n_new / budget) + 0.01
    g = neighbour(cur)
    if g == cur: continue
    f = ev(g)
    if f < fcur or rng.random() < math.exp(-(f - fcur) / T): cur, fcur = g, f
    if fcur < fbest: best, fbest = cur, fcur
    if n_new % 10 == 0: print(wi, n_new, budget, round(time.time() - t0), "cur", cur, round(fcur, 3), "best", best, round(fbest, 3), flush=True)
print("done", wi, "best", best, fbest, flush=True)
