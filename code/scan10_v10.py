"""v10 scans. Worker wi of nw handles every nw-th job of the list:
   SNV-only scan of all 820 contiguous ranges for the 11 non-MLP models, and the MLP under unit variance (z) and SNV-only (snv) for ranges that start at gates 0..3.
Output v10_out/scan10_part{wi}.jsonl (resumable). Each record: scaling, grp, a, b, pred{model: [34 LOTO predictions]}, sec."""
import os, sys, time, json, mimetypes
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2, lt2f
MLP = "MLP (64,64)"; FASTM = [m for m in lt2.MODEL_NAMES if m != MLP]
R = [(a, b) for a in range(40) for b in range(a, 40)]; RM = [(a, b) for (a, b) in R if a <= 3]
JOBS = [("snv", "fast", a, b) for a, b in R] + [("z", "mlp", a, b) for a, b in RM] + [("snv", "mlp", a, b) for a, b in RM]
JOBS = [JOBS[(k * 7919) % len(JOBS)] for k in range(len(JOBS))]    # deterministic shuffle (7919 is prime, so this is a permutation) for load balance
wi, nw = int(sys.argv[1]), int(sys.argv[2]); OUT = HERE / "v10_out" / f"scan10_part{wi}.jsonl"
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); cube = np.stack([s["rate"] for s in sp])
done = set()
if OUT.exists():
    for l in open(OUT): d = json.loads(l); done.add((d["scaling"], d["grp"], d["a"], d["b"]))
jobs = [j for j in JOBS[wi::nw] if j not in done]; print(wi, len(jobs), "jobs", flush=True); t0 = time.time()
with open(OUT, "a") as fh:
    for k, (sc_, grp, a, b) in enumerate(jobs):
        t1 = time.time(); X = lt2.snv_rows(cube[:, :, list(range(a, b + 1))].reshape(len(temps), -1))
        p = lt2f.real_loto(X, temps, sc_, [MLP] if grp == "mlp" else FASTM)
        fh.write(json.dumps(dict(scaling=sc_, grp=grp, a=a, b=b, pred={m: v.tolist() for m, v in p.items()}, sec=time.time() - t1)) + "\n"); fh.flush()
        if k % 10 == 0: print(wi, k, len(jobs), round(time.time() - t0), flush=True)
print("done", wi, round(time.time() - t0), flush=True)
