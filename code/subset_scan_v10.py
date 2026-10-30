"""v10 follow-up: GPR (and Bayesian ridge) with ANY subset of gates, unit variance. Real LOTO on the 34 measured spectra.
Phase 1: all subsets of size 1, 2 and 3 (40 + 780 + 9880). Worker wi of nw handles every nw-th job. Output v10_out/subset/subset_part{wi}.jsonl (resumable)."""
import os, sys, time, json, mimetypes, itertools
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2, lt2f
MODELS = ["GPR", "Bayesian Ridge (poly3 PCs)"]
def jobs_list():
    s12 = [c for k in (1, 2) for c in itertools.combinations(range(40), k)]
    t0 = [c for c in itertools.combinations(range(40), 3) if c[0] == 0]
    t1 = [c for c in itertools.combinations(range(40), 3) if c[0] != 0]
    out = []
    for grp in (s12, t0, t1): out += [grp[(k * 7919) % len(grp)] for k in range(len(grp))] if len(grp) % 7919 else grp
    return out
if __name__ == "__main__":
    wi, nw = int(sys.argv[1]), int(sys.argv[2]); OUT = HERE / "v10_out" / "subset" / f"subset_part{wi}.jsonl"; OUT.parent.mkdir(exist_ok=True)
    sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); cube = np.stack([s["rate"] for s in sp])
    done = set()
    if OUT.exists():
        for l in open(OUT): done.add(tuple(json.loads(l)["gates"]))
    J = jobs_list(); jobs = [j for j in J[wi::nw] if j not in done]; print(wi, len(jobs), "jobs", flush=True); t0 = time.time()
    with open(OUT, "a") as fh:
        for k, g in enumerate(jobs):
            t1 = time.time(); X = lt2.snv_rows(cube[:, :, list(g)].reshape(len(temps), -1))
            p = lt2f.real_loto(X, temps, "z", MODELS)
            fh.write(json.dumps(dict(gates=list(g), pred={m: v.tolist() for m, v in p.items()}, sec=time.time() - t1)) + "\n"); fh.flush()
            if k % 20 == 0: print(wi, k, len(jobs), round(time.time() - t0), flush=True)
    print("done", wi, round(time.time() - t0), flush=True)
