"""v9 nested validation, general version (leave-two-out formulation of the inner LOTO).
usage: python nested_gen_v9.py wi nw scaling max_a tag   -> v9_out/nested_{tag}_pairs_part{wi}.json
Family: all contiguous ranges [a, b] with 0 <= a <= max_a. A fit without set points i and j predicts both; the inner LOTO RMSE of a range in outer fold i is the RMSE of
the predictions of the j != i; the selected range is then taken from the real-LOTO scan (fit on all 33)."""
import os, sys, time, json, mimetypes
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2, lt2f
MODELS = json.loads(os.environ.get("NESTED_MODELS", '["Bayesian Ridge (poly3 PCs)", "GPR", "Kernel Ridge (Poly)", "Sensor Fusion (PC1-3 inv-var)", "3D Spline (PC1-3 TPS)"]'))
wi, nw, scaling, max_a, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); cube = np.stack([s["rate"] for s in sp]); n = len(temps)
FAM = [(a, b) for a in range(max_a + 1) for b in range(a, 40)]
Xs = {(a, b): lt2.snv_rows(cube[:, :, list(range(a, b + 1))].reshape(n, -1)) for a, b in FAM}
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)][wi::nw]
OUT = HERE / "v9_out" / f"nested_{tag}_pairs_part{wi}.json"; res = json.load(open(OUT)) if OUT.exists() else {}; t0 = time.time()
for k, (i, j) in enumerate(pairs):
    key = f"{i}-{j}"
    if key in res: continue
    tr = np.array([m for m in range(n) if m not in (i, j)]); r = {m: [] for m in MODELS}
    for ab in FAM:
        o = lt2f.fit_predict_all(Xs[ab][tr], temps[tr], Xs[ab][[i, j]], scaling, MODELS)
        for m in MODELS: r[m].append([float(o[m][0]), float(o[m][1])])
    res[key] = r
    if k % 5 == 0: json.dump(res, open(OUT, "w")); print(wi, k, len(pairs), round(time.time() - t0), flush=True)
json.dump(res, open(OUT, "w")); print("done", wi, round(time.time() - t0), flush=True)
