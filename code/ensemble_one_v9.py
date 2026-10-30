import os, sys, json, time, mimetypes
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2, lt2f
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n = len(temps); cube = np.stack([s["rate"] for s in sp])
sigma_t = np.array([float(np.median(s["rate"][:, -lt2.N_LAST:].std(axis=1))) for s in sp]); N_SIM = lt2.N_SIM
mode, a, b, model, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
Xm = cube[:, :, list(range(a, b + 1))].reshape(n, -1); rng = np.random.default_rng(lt2.RNG)
Xs = lt2.snv_rows(np.stack([Xm[i] + rng.normal(0.0, sigma_t[i], size=(N_SIM, Xm.shape[1])) for i in range(n)])); pred = np.full((n, N_SIM), np.nan)
for i in range(n):
    tr = np.array([j for j in range(n) if j != i]); pred[i] = lt2f.fit_predict_all(Xs[tr].mean(axis=1), temps[tr], Xs[i], mode, [model])[model]
np.save(out, pred); print("done", model, mode, a, b)
