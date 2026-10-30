"""v9 exhaustive scan with unit-variance channel scaling (mode "z"): every contiguous gate range [a, b] (0 <= a <= b <= 39, 820 ranges, no gate dropped in between), unit-variance channel scaling,
real LOTO (one measured spectrum per set point) for the 12 regressors. Worker wi of nw appends to v9_out/z_scan_part{wi}.jsonl (resumable)."""
import os, sys, time, json, mimetypes
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2
NG = 40; MLP = "MLP (64,64)"; FASTM = [m for m in lt2.MODEL_NAMES if m != MLP]
RANGES = [(a, b) for a in range(NG) for b in range(a, NG)]
_st = {}
def init():
    sp = lt2.load_spectra(); _st["temps"] = np.array([s["temperature_c"] for s in sp], float); _st["cube"] = np.stack([s["rate"] for s in sp])
def work(job):
    a, b, grp = job; cube, temps = _st["cube"], _st["temps"]; t0 = time.time()
    X = lt2.snv_rows(cube[:, :, list(range(a, b + 1))].reshape(len(temps), -1))
    p = lt2.real_loto(X, temps, "z", [MLP] if grp == "mlp" else FASTM)
    return dict(a=a, b=b, grp=grp, pred={m: v.tolist() for m, v in p.items()}, sec=time.time() - t0)
if __name__ == "__main__":
    wi, nw = int(sys.argv[1]), int(sys.argv[2]); OUT = HERE / "v9_out" / f"z_scan_part{wi}.jsonl"
    done = set()
    if OUT.exists():
        for l in open(OUT):
            d = json.loads(l); done.add((d["a"], d["b"], d["grp"]))
    alljobs = [(a, b, g) for g in ("fast",) for (a, b) in RANGES]
    jobs = [j for j in alljobs[wi::nw] if j not in done]
    print(wi, len(jobs), "jobs", flush=True); t0 = time.time(); init()
    with open(OUT, "a") as fh:
        for k, jb in enumerate(jobs):
            fh.write(json.dumps(work(jb)) + "\n"); fh.flush()
            if k % 10 == 0: print(wi, k, len(jobs), round(time.time() - t0), flush=True)
    print("done", wi, round(time.time() - t0), flush=True)
