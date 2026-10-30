"""v10 evaluation of all models at ONE fixed contiguous gate range per model and ONE channel scaling (the same range and scaling in every LOTO fold).
usage: python eval_scaling_v10.py {z|pareto|snv}
 1 scan table from the real-LOTO scans (820 ranges; MLP: ranges that start at gates 0..3), best range per model (lowest overall RMSE, ties by smaller b)
 2 100-replicate detector-noise ensemble LOTO of every model at its range -> RMSE, bias, precision per set point
 3 range summary (overall, 25-300, 310-350, 25-100, 110-200, 210-300, interior), paired bootstrap over set points, plateau widths
Writes v10_out/{tag}_*.csv / .npz"""
import sys, json, glob, time, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2, lt2f
OUT = HERE / "v10_out"; V9 = HERE / "v9_out"; tag = sys.argv[1]; assert tag in ("z", "pareto", "snv")
RNG, N_SIM = lt2.RNG, lt2.N_SIM; MN = lt2.MODEL_NAMES; MLP = "MLP (64,64)"; t_start = time.time()
def log(*a): print(f"[{time.time()-t_start:6.0f}s]", *a, flush=True)
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n = len(temps); cube = np.stack([s["rate"] for s in sp])
sigma_t = np.array([float(np.median(s["rate"][:, -lt2.N_LAST:].std(axis=1))) for s in sp])
# ---- 1 scan
PRED = {}
def add(d):
    for m, v in d["pred"].items(): PRED.setdefault((d["a"], d["b"]), {})[m] = np.array(v)
if tag == "z":
    for f in sorted(glob.glob(str(V9 / "z_scan_part*.jsonl"))):
        for l in open(f): add(json.loads(l))
elif tag == "pareto":
    for f in sorted(glob.glob(str(V9 / "scan_ranges_part*.jsonl"))):
        for l in open(f):
            d = json.loads(l)
            if d["grp"] == "fast" or d["a"] <= 3: add(d)
for f in sorted(glob.glob(str(OUT / "scan10_part*.jsonl"))):
    for l in open(f):
        d = json.loads(l)
        if d["scaling"] == tag: add(d)
rows = []
for (a, b), pr in PRED.items():
    for m, p in pr.items():
        e = p - temps; rows.append(dict(a=a, b=b, gates=f"{a}-{b}", n_gates=b - a + 1, model=m, rmse=np.sqrt(np.mean(e ** 2)), rmse_25_300=np.sqrt(np.mean(e[temps <= 300] ** 2)), rmse_310_350=np.sqrt(np.mean(e[temps > 300] ** 2))))
scan = pd.DataFrame(rows); scan.to_csv(OUT / f"{tag}_scan_table.csv", index=False)
cnt = scan.groupby("model").size(); log("ranges per model", cnt.to_dict()); assert all(cnt[m] == 820 for m in MN if m != MLP) and cnt[MLP] == 154, cnt.to_dict()
best = scan.sort_values(["model", "rmse", "b"]).groupby("model").head(1).set_index("model").loc[MN]; best.to_csv(OUT / f"{tag}_best_range.csv"); log(best[["gates", "n_gates", "rmse", "rmse_25_300", "rmse_310_350"]].round(3).to_string())
pl = []
for m in MN:
    s = scan[scan.model == m]; b0 = s.rmse.min(); pl.append(dict(model=m, best=b0, n_within_10pct=int((s.rmse <= 1.10 * b0).sum()), n_within_25pct=int((s.rmse <= 1.25 * b0).sum()), n_ranges=len(s), median_all=float(s.rmse.median())))
pd.DataFrame(pl).round(4).to_csv(OUT / f"{tag}_plateau.csv", index=False)
# ---- 2 ensemble
def ensemble(gates, models):
    Xm = cube[:, :, list(gates)].reshape(n, -1); rng = np.random.default_rng(RNG)
    Xs = lt2.snv_rows(np.stack([Xm[i] + rng.normal(0.0, sigma_t[i], size=(N_SIM, Xm.shape[1])) for i in range(n)])); pred = {m: np.full((n, N_SIM), np.nan) for m in models}
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        for m, v in lt2f.fit_predict_all(Xs[tr].mean(axis=1), temps[tr], Xs[i], tag, models).items(): pred[m][i] = v
    return pred
P = {}; ranges = {m: (int(best.loc[m, "a"]), int(best.loc[m, "b"])) for m in MN}
for rg in sorted(set(ranges.values())):
    ms = [m for m in MN if ranges[m] == rg]; P.update(ensemble(range(rg[0], rg[1] + 1), ms)); log("ensemble", rg, ms)
np.savez_compressed(OUT / f"{tag}_ensemble.npz", temps=temps, **{f"m{i}": P[m] for i, m in enumerate(MN)}, names=np.array(MN))
cl = np.load(HERE / "v6_out" / "classical_ensemble.npz"); allp = [("LIR", cl["lir"]), ("tau640_MS", cl["tau640"]), ("tau688_MS", cl["tau688"])] + [(m, P[m]) for m in MN]
def per_t(pr):
    err = pr - temps[:, None]; return dict(rmse=np.sqrt(np.nanmean(err ** 2, axis=1)), bias=np.nanmean(err, axis=1), prec=np.nanstd(pr, axis=1, ddof=1))
pd.DataFrame({f"{name}|{k}": v for name, pr in allp for k, v in per_t(pr).items()}, index=temps).rename_axis("T_C").to_csv(OUT / f"{tag}_per_setpoint_stats.csv")
lo = temps <= 300; interior = (temps > 25) & (temps < 350); rows = []
for name, pr in allp:
    q = np.nanmean((pr - temps[:, None]) ** 2, axis=1); pt = per_t(pr)
    rows.append(dict(model=name, overall=np.sqrt(q.mean()), r25_300=np.sqrt(q[lo].mean()), r310_350=np.sqrt(q[~lo].mean()), interior_no_endpoints=np.sqrt(q[interior].mean()), r25_100=np.sqrt(q[temps <= 100].mean()),
                     r110_200=np.sqrt(q[(temps >= 110) & (temps <= 200)].mean()), r210_300=np.sqrt(q[(temps >= 210) & (temps <= 300)].mean()), share_350_of_sq_error=q[-1] / q.sum(), max_abs_bias=np.abs(pt["bias"]).max(), at_T=temps[np.abs(pt["bias"]).argmax()],
                     prec_min=pt["prec"][pt["prec"] > 0].min(), prec_max=pt["prec"].max(), mean_abs_bias=np.abs(pt["bias"]).mean()))
rs = pd.DataFrame(rows).set_index("model"); rs.round(5).to_csv(OUT / f"{tag}_range_summary.csv"); log(rs[["overall", "r25_300", "r310_350", "max_abs_bias", "prec_max"]].round(3).to_string())
# ---- 3 bootstrap over set points (ensemble squared errors)
e2 = {k: np.nanmean((pr - temps[:, None]) ** 2, axis=1) for k, pr in allp}; rb = np.random.default_rng(1); idx = rb.integers(0, n, size=(10000, n)); ranked = rs.loc[MN, "overall"].sort_values(); brow = []
def boot(a, b):
    d = np.sqrt(e2[a][idx].mean(1)) - np.sqrt(e2[b][idx].mean(1)); return dict(a=a, b=b, mean_diff=d.mean(), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), p_a_better=np.mean(d < 0))
for m in MN: brow.append(boot(m, "LIR"))
for m in MN:
    if m != "GPR": brow.append(boot("GPR", m))
pd.DataFrame(brow).round(4).to_csv(OUT / f"{tag}_bootstrap.csv", index=False)
json.dump({"ranked": ranked.to_dict(), "ranges": {m: list(v) for m, v in ranges.items()}}, open(OUT / f"{tag}_ranking.json", "w"), indent=1); log("done")
