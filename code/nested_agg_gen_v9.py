"""Aggregation of nested_gen_v9.py: python nested_agg_gen_v9.py scan_prefix tag max_a [scaling label] -> prints and writes v9_out/nested_{tag}_summary.csv / _extra.json"""
import sys, json, glob, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2
OUT = HERE / "v9_out"; prefix, tag, max_a = sys.argv[1], sys.argv[2], int(sys.argv[3])
MODELS = json.loads(sys.argv[4]) if len(sys.argv) > 4 else ["Bayesian Ridge (poly3 PCs)", "GPR", "Kernel Ridge (Poly)", "Sensor Fusion (PC1-3 inv-var)", "3D Spline (PC1-3 TPS)"]
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n = len(temps); wl = sp[0]["wavelength_nm"]; X50 = np.stack([s["rate"][:, 0] for s in sp])
FAM = [(a, b) for a in range(max_a + 1) for b in range(a, 40)]; fi = {ab: k for k, ab in enumerate(FAM)}
PRED = {m: {} for m in MODELS}
for f in sorted(glob.glob(str(OUT / f"{prefix}_part*.jsonl"))):
    for l in open(f):
        d = json.loads(l)
        if d.get("grp") == "fast" and d["a"] <= max_a:
            for m in MODELS: PRED[m][(d["a"], d["b"])] = np.array(d["pred"][m])
assert all(len(PRED[m]) == len(FAM) for m in MODELS), [len(PRED[m]) for m in MODELS]
PAIR = {m: np.full((len(FAM), n, n), np.nan) for m in MODELS}
for f in sorted(glob.glob(str(OUT / f"nested_{tag}_pairs_part*.json"))):
    for key, r in json.load(open(f)).items():
        i, j = map(int, key.split("-"))
        for m in MODELS:
            v = np.array(r[m]); PAIR[m][:, i, j] = v[:, 1]; PAIR[m][:, j, i] = v[:, 0]
assert not np.isnan(PAIR[MODELS[0]][:, ~np.eye(n, dtype=bool)]).any(), "missing pairs"
lo_, hi_ = wl < 675, wl > 680; lir = X50[:, lo_].sum(1) / np.maximum(X50[:, hi_].sum(1), 1.0); lp = np.empty(n)
for i in range(n):
    tr = np.array([j for j in range(n) if j != i]); od = np.argsort(lir[tr]); lp[i] = np.clip(PchipInterpolator(lir[tr][od], temps[tr][od], extrapolate=True)(lir[i]), temps[tr].min() - 5, temps[tr].max() + 5)
e_lir = lp - temps; rb = np.random.default_rng(1); idx = rb.integers(0, n, size=(5000, n)); lo = temps <= 300; rows = []; extra = {}
for m in MODELS:
    pick = np.empty(n, int); pred = np.empty(n)
    for i in range(n):
        oth = [j for j in range(n) if j != i]; inner = np.array([np.sqrt(np.mean((PAIR[m][k, i, oth] - temps[oth]) ** 2)) for k in range(len(FAM))]); pick[i] = int(np.argmin(inner)); pred[i] = PRED[m][FAM[pick[i]]][i]
    e = pred - temps; e2 = e ** 2; full = min(FAM, key=lambda ab: np.sqrt(np.mean((PRED[m][ab] - temps) ** 2))); non = float(np.sqrt(np.mean((PRED[m][full] - temps) ** 2))); fk = fi[full]
    d = np.sqrt((e[idx] ** 2).mean(1)) - np.sqrt((e_lir[idx] ** 2).mean(1)); ends = np.isin(temps, [25, 350])
    lab = lambda k: f"{FAM[k][0]}-{FAM[k][1]}"
    extra[m] = dict(nested=float(np.sqrt(e2.mean())), nested_25_300=float(np.sqrt(e2[lo].mean())), nested_310_350=float(np.sqrt(e2[~lo].mean())), nested_no_ends=float(np.sqrt(e2[~ends].mean())), full_range=f"{full[0]}-{full[1]}", full_rmse=non,
                    full_25_300=float(np.sqrt(np.mean(((PRED[m][full] - temps) ** 2)[lo]))), full_310_350=float(np.sqrt(np.mean(((PRED[m][full] - temps) ** 2)[~lo]))), same=int((pick == fk).sum()), distinct=int(len(set(pick))),
                    picks={lab(k): int((pick == k).sum()) for k in sorted(set(pick))}, diff_lir=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)), share_25_30_350=float(e2[np.isin(temps, [25, 30, 350])].sum() / e2.sum()))
    rows.append(dict(model=m, nested=extra[m]["nested"], full_range=extra[m]["full_range"], full=non, same_folds=extra[m]["same"], distinct=extra[m]["distinct"], diff_vs_LIR=extra[m]["diff_lir"], lo=extra[m]["lo"], hi=extra[m]["hi"]))
json.dump(extra, open(OUT / f"nested_{tag}_extra.json", "w"), indent=1); df = pd.DataFrame(rows); df.round(4).to_csv(OUT / f"nested_{tag}_summary.csv", index=False)
print("LIR real LOTO", np.sqrt(np.mean(e_lir ** 2))); print(df.round(3).to_string())
for m in MODELS: print(m, extra[m]["picks"])
