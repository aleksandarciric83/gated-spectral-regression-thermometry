"""Noise-world scan (family of gate sets). usage: noise_world_scan_v10.py wi nw. Worlds: X = S_syn + tau * sign_i * R_syn.
Gate sets per world: (0,b) b=0..39 contiguous; (0,x) pairs x=2..39; (0,1,x) x=3..39. Output v10_out/noise/fam_part{wi}.jsonl (resumable)."""
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

WORLDS = [(2.0, 11), (2.0, 12), (2.0, 13), (1.0, 21), (1.0, 22), (1.0, 23), (0.5, 31), (0.5, 32), (0.5, 33), (0.25, 41), (0.25, 42), (0.25, 43), (0.0, -1)]
GS = [tuple(range(b + 1)) for b in range(40)] + [(0, x) for x in range(2, 40)] + [(0, 1, x) for x in range(3, 40)]
JOBS = [(t, sd, g) for (t, sd) in WORLDS for g in GS]; JOBS = [JOBS[(k * 7919) % len(JOBS)] for k in range(len(JOBS))]
if __name__ == "__main__":
    wi, nw = int(sys.argv[1]), int(sys.argv[2]); OUT = HERE / "v10_out" / "noise" / f"fam_part{wi}.jsonl"; OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        for l in open(OUT):
            try: d = json.loads(l); done.add((d["tau"], d["seed"], tuple(d["gates"])))
            except Exception: pass
    jobs = [j for j in JOBS[wi::nw] if j not in done]; print(wi, len(jobs), "jobs", flush=True); t0 = time.time(); cache = {}
    with open(OUT, "a") as fh:
        for k, (tau, sd, g) in enumerate(jobs):
            if (tau, sd) not in cache: cache = {(tau, sd): world(tau, sd)}
            p = evaluate(cache[(tau, sd)], g)
            fh.write(json.dumps(dict(tau=tau, seed=sd, gates=list(g), pred={m: v.tolist() for m, v in p.items()})) + "\n"); fh.flush()
            if k % 20 == 0: print(wi, k, len(jobs), round(time.time() - t0), flush=True)
    print("done", wi, flush=True)
