
"""Analysis for the v6 manuscript: gates 0-10 (50 ns to 500.05 us), SNV + Pareto channel scaling, 12 regressors,
100 detector-noise replicates per set point, leave-one-temperature-out (LOTO). Writes tables and arrays to ./v6_out."""
import os, sys, time, json, warnings, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None   # sandbox denies the registry read
from pathlib import Path
import numpy as np, pandas as pd, openpyxl
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "Fw" / "paper"))
import lt2
import make_paper_final_nofig as mp          # fit_biexp / biexp of the original analysis script
OUT = HERE / "v6_out"; OUT.mkdir(exist_ok=True)
FW = Path(os.environ.get("GATED_DATA_DIR", HERE.parent / "data"))
G11 = list(range(11)); RNG = lt2.RNG; N_SIM = lt2.N_SIM
t_start = time.time()
def log(*a): print(f"[{time.time()-t_start:6.0f}s]", *a, flush=True)

spectra = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in spectra], float); n = len(temps)
wl = spectra[0]["wavelength_nm"]; delays_full = spectra[0]["delays_us"]
cube = np.stack([s["rate"] for s in spectra])                       # (34, 1024, 40)
sigma_t = np.array([float(np.median(s["rate"][:, -lt2.N_LAST:].std(axis=1))) for s in spectra])
old = np.load(HERE / "Fw" / "paper" / "figures" / "raw_preds.npz", allow_pickle=True)   # classical ensembles of the original script
def feat(gates): return lt2.snv_rows(cube[:, :, list(gates)].reshape(n, -1))

# 1 scaling comparison, real LOTO (one acquisition per set point)
res = {}
for label, gates in [("g31", range(31)), ("g11", range(11))]:
    for mode in ["z", "snv", "pareto"]:
        res[(label, mode)] = lt2.real_loto(feat(gates), temps, mode)
pd.DataFrame({f"{l}-{m}": {k: lt2.rmse(v, temps) for k, v in p.items()} for (l, m), p in res.items()}).round(4).to_csv(OUT / "scaling_comparison_real_loto.csv"); log("scaling comparison")

# 2 gate-count sweep with Pareto scaling
keep = ["GPR", "Bayesian Ridge (poly3 PCs)", "Kernel Ridge (Poly)", "Sensor Fusion (PC1-3 inv-var)"]
sweep = {ng: {m: lt2.rmse(v, temps) for m, v in lt2.real_loto(feat(range(ng)), temps, "pareto", keep).items()} for ng in [2, 3, 5, 8, 11, 15, 21, 31]}
pd.DataFrame(sweep).T.round(3).to_csv(OUT / "gate_count_sweep_pareto_real_loto.csv"); log("gate sweep")

# 3 detector-noise ensemble LOTO on the final cube
def ensemble_loto(gates, mode, seed=RNG):
    X_mean = cube[:, :, list(gates)].reshape(n, -1); rng = np.random.default_rng(seed)
    X_sim = np.stack([X_mean[i] + rng.normal(0.0, sigma_t[i], size=(N_SIM, X_mean.shape[1])) for i in range(n)])
    X_snv = lt2.snv_rows(X_sim); pred = {m: np.full((n, N_SIM), np.nan) for m in lt2.MODEL_NAMES}
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        for m, v in lt2.fit_predict_all(X_snv[tr].mean(axis=1), temps[tr], X_snv[i], mode).items(): pred[m][i] = v
    return pred
ens = {mode: ensemble_loto(G11, mode) for mode in ["pareto", "z", "snv"]}
pd.DataFrame({mode: {m: float(np.sqrt(np.nanmean((p - temps[:, None]) ** 2))) for m, p in ens[mode].items()} for mode in ens}).round(4).to_csv(OUT / "scaling_comparison_ensemble_11gates.csv")
P = ens["pareto"]; np.savez_compressed(OUT / "ml_ensemble_pareto_11gates.npz", temps=temps, **{f"m{i}": P[m] for i, m in enumerate(lt2.MODEL_NAMES)}, names=np.array(lt2.MODEL_NAMES)); log("ensemble")

# 4 classical baselines (Mott-Seitz lifetimes, PCHIP LIR) on the ensembles of the original script
kB = 8.617333e-5
def ms(T_c, tau0, Cc, dE): return tau0 / (1 + Cc * np.exp(-dE / (kB * (np.asarray(T_c, float) + 273.15))))
def ms_fit(Tt, tau):
    best = None
    for p0 in [(tau[0], 1e3, 0.3), (tau[0], 1e2, 0.2), (tau[0], 1e4, 0.5), (tau[0] * 1.1, 3e2, 0.15)]:
        try:
            popt, _ = curve_fit(ms, Tt, tau, p0=p0, bounds=([1, 0, 0.01], [1e4, 1e9, 3.0]), maxfev=20000); sse = np.sum((ms(Tt, *popt) - tau) ** 2)
            if best is None or sse < best[0]: best = (sse, popt)
        except Exception: pass
    return None if best is None else best[1]
def ms_loto_ens(tau_sim):
    pred = np.full(tau_sim.shape, np.nan)
    for i in range(n):
        tr = np.arange(n) != i; popt = ms_fit(temps[tr], np.nanmean(tau_sim[tr], axis=1))
        grid = np.linspace(temps[tr].min() - 5, temps[tr].max() + 5, 4000); g = ms(grid, *popt)
        pred[i] = grid[np.argmin(np.abs(g[None, :] - tau_sim[i][:, None]), axis=1)]
    return pred
pred640 = ms_loto_ens(old["tau640_sim"]); predpk = ms_loto_ens(old["taupk_sim"]); lir_pred = old["cl_2"]
pf640 = ms_fit(temps, old["tau_640"]); pfpk = ms_fit(temps, old["tau_pk"])
json.dump({"MS_640": list(map(float, pf640)), "MS_688": list(map(float, pfpk))}, open(OUT / "mott_seitz_fit_all_points.json", "w"))
np.savez_compressed(OUT / "classical_ensemble.npz", temps=temps, tau640=pred640, tau688=predpk, lir=lir_pred, tau_640=old["tau_640"], tau_pk=old["tau_pk"], lir_meas=old["lir"])
def per_t(pred):
    err = pred - temps[:, None]; return dict(rmse=np.sqrt(np.nanmean(err ** 2, axis=1)), bias=np.nanmean(err, axis=1), prec=np.nanstd(pred, axis=1, ddof=1))
allp = [("LIR", lir_pred), ("tau640_MS", pred640), ("tau688_MS", predpk)] + [(m, P[m]) for m in lt2.MODEL_NAMES]
pd.DataFrame({f"{a}|{b}": v for name, pr in allp for a in [name] for b, v in per_t(pr).items()}, index=temps).rename_axis("T_C").to_csv(OUT / "per_setpoint_stats.csv")
lo = temps <= 300; interior = (temps > 25) & (temps < 350); rows = []
for name, pr in allp:
    q = np.nanmean((pr - temps[:, None]) ** 2, axis=1)
    rows.append(dict(model=name, overall=np.sqrt(q.mean()), r25_300=np.sqrt(q[lo].mean()), r310_350=np.sqrt(q[~lo].mean()), interior_no_endpoints=np.sqrt(q[interior].mean()),
                     r25_100=np.sqrt(q[temps <= 100].mean()), r110_200=np.sqrt(q[(temps >= 110) & (temps <= 200)].mean()), r210_300=np.sqrt(q[(temps >= 210) & (temps <= 300)].mean()),
                     share_350_of_sq_error=q[-1] / q.sum()))
pd.DataFrame(rows).set_index("model").round(4).to_csv(OUT / "range_summary.csv")
rb = np.random.default_rng(1); idx = rb.integers(0, n, size=(5000, n)); e2 = {k: np.nanmean((pr - temps[:, None]) ** 2, axis=1) for k, pr in allp}
brow = []
for a, b in [("Bayesian Ridge (poly3 PCs)", "LIR"), ("GPR", "LIR"), ("Kernel Ridge (Poly)", "LIR"), ("Sensor Fusion (PC1-3 inv-var)", "LIR"), ("Bayesian Ridge (poly3 PCs)", "GPR"), ("Bayesian Ridge (poly3 PCs)", "Kernel Ridge (Poly)"), ("GPR", "Kernel Ridge (Poly)"), ("GPR", "Sensor Fusion (PC1-3 inv-var)")]:
    d = np.sqrt(e2[a][idx].mean(1)) - np.sqrt(e2[b][idx].mean(1)); brow.append(dict(a=a, b=b, mean_diff=d.mean(), ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), p_a_better=np.mean(d < 0)))
pd.DataFrame(brow).round(4).to_csv(OUT / "paired_bootstrap_over_setpoints.csv"); log("classical + stats")

# 5 PCA of the final cube, and PC-subset check (PC1 only, PC1-2, PC1-3), real LOTO
X0 = feat(G11); Xs, _ = lt2.channel_scale(X0, X0, "pareto"); pca_all = PCA(10, random_state=RNG).fit(Xs); PC_all = pca_all.transform(Xs)
for k in range(3):
    if np.corrcoef(PC_all[:, k], temps)[0, 1] < 0: PC_all[:, k] *= -1
np.savez(OUT / "pca_final.npz", temps=temps, PC=PC_all, evr=pca_all.explained_variance_ratio_)
def gpr_(): return GaussianProcessRegressor(kernel=C(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(1.0, (1e-5, 1e3)), n_restarts_optimizer=3, normalize_y=True, random_state=RNG)
sub = {}
for k in (1, 2, 3):
    pg, pb = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i]); Xtr, Xte = lt2.channel_scale(X0[tr], X0[i:i + 1], "pareto")
        pca = PCA(3, random_state=RNG).fit(Xtr); a, b = pca.transform(Xtr), pca.transform(Xte)
        fl = np.array([-1 if np.corrcoef(a[:, j], temps[tr])[0, 1] < 0 else 1 for j in range(3)]); a, b = (a * fl)[:, :k], (b * fl)[:, :k]
        sc = StandardScaler().fit(a); pg[i] = gpr_().fit(sc.transform(a), temps[tr]).predict(sc.transform(b))[0]
        pb[i] = make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge()).fit(sc.transform(a), temps[tr]).predict(sc.transform(b))[0]
    sub[k] = dict(GPR=lt2.rmse(pg, temps), BayesianRidge=lt2.rmse(pb, temps))
pd.DataFrame(sub).T.round(3).rename_axis("n_PCs").to_csv(OUT / "pc_subset_real_loto.csv"); log("pc subset", sub)

# 6 weighted sensor fusion on the final cube, floor-noise weights, real LOTO
def sf_general(PC_tr, T_tr, PC_te, sig=None, cov=None, grid_pts=4000):
    tu_all = np.asarray(T_tr, float); o = np.argsort(tu_all); tu = tu_all[o]; fuse_T = np.linspace(tu.min() - 5, tu.max() + 5, grid_pts)
    g = np.stack([PchipInterpolator(tu, PC_tr[o, d], extrapolate=True)(fuse_T) for d in range(3)], 0); diff = PC_te[:, :, None] - g[None, :, :]
    if cov is not None:
        Sg = np.empty((grid_pts, 3, 3))
        for a in range(3):
            for b in range(3): Sg[:, a, b] = PchipInterpolator(tu, cov[o, a, b], extrapolate=True)(fuse_T)
        cost = np.einsum("mag,gab,mbg->mg", diff, np.linalg.inv(Sg + 1e-12 * np.eye(3)[None]), diff)
    elif sig is not None:
        s = np.stack([np.maximum(PchipInterpolator(tu, sig[o, d], extrapolate=True)(fuse_T), 1e-9) for d in range(3)], 0); cost = ((diff ** 2) / (s[None] ** 2)).sum(1)
    else: cost = (diff ** 2).sum(1)
    return fuse_T[np.argmin(cost, axis=1)]
rng_w = np.random.default_rng(7); Xm = cube[:, :, G11].reshape(n, -1)
Xsim = lt2.snv_rows(np.stack([Xm[i] + rng_w.normal(0, sigma_t[i], size=(N_SIM, Xm.shape[1])) for i in range(n)])); wout = {k: np.full(n, np.nan) for k in ["equal", "diag_floor", "gls_floor"]}
for i in range(n):
    tr = np.array([j for j in range(n) if j != i]); Xtr_m = Xsim[tr].mean(1); Xtr, Xte = lt2.channel_scale(Xtr_m, lt2.snv_rows(Xm[i:i + 1]), "pareto")
    pca = PCA(3, random_state=RNG).fit(Xtr); a, b = pca.transform(Xtr), pca.transform(Xte); fl = np.array([-1 if np.corrcoef(a[:, j], temps[tr])[0, 1] < 0 else 1 for j in range(3)]); a, b = a * fl, b * fl
    sig = np.empty((len(tr), 3)); cov = np.empty((len(tr), 3, 3))
    for k, j in enumerate(tr):
        Z = pca.transform(lt2.channel_scale(Xtr_m, Xsim[j], "pareto")[1]) * fl; sig[k] = Z.std(0, ddof=1); cov[k] = np.cov(Z.T)
    wout["equal"][i] = sf_general(a, temps[tr], b)[0]; wout["diag_floor"][i] = sf_general(a, temps[tr], b, sig=sig)[0]; wout["gls_floor"][i] = sf_general(a, temps[tr], b, cov=cov)[0]
pd.DataFrame({k: {"overall": lt2.rmse(v, temps), "25-300": lt2.rmse(v, temps, temps <= 300), "310-350": lt2.rmse(v, temps, temps > 300)} for k, v in wout.items()}).T.round(3).to_csv(OUT / "sensor_fusion_weights_final_cube_real_loto.csv")
nc = []
for Tq in (150.0, 340.0):
    i = int(np.where(temps == Tq)[0][0]); Z = pca_all.transform(lt2.channel_scale(X0, Xsim[i], "pareto")[1])[:, :3]
    sgn = np.array([1 if np.corrcoef(PC_all[:, k], pca_all.transform(Xs)[:, k])[0, 1] > 0 else -1 for k in range(3)]); Z = Z * sgn; cc = np.corrcoef(Z.T)
    nc.append(dict(T_C=Tq, r12=cc[0, 1], r13=cc[0, 2], r23=cc[1, 2]))
pd.DataFrame(nc).round(3).to_csv(OUT / "floor_noise_pc_correlation_final_cube.csv"); log("weights")

# 7 repeatability (40 x 30 pulses, 100 single pulses at 350 C) and energy series, models of the 50 ns spectrum with Pareto scaling
def read_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True); rows = list(wb[sheet].iter_rows(values_only=True)); wb.close(); return rows
R30 = np.array(read_sheet(FW / "repeatability.xlsx", "350C_0-50us_635mJ")[1:1025], float)[:, 1:].T
R1 = np.array(read_sheet(FW / "repeatability.xlsx", "350C_0-50us_647mJ")[1:1025], float)[:, 1:].T
energy = {mj: np.array(read_sheet(FW / "power_dependence.xlsx", f"{mj}mJ"), float)[:, 1:].T for mj in (573, 594, 616, 639)}
X50 = cube[:, :, 0]; X50s = lt2.snv_rows(X50); mlo, mhi = wl < 675, wl > 680
Xtr50, _ = lt2.channel_scale(X50s, X50s[:1], "pareto"); pca50 = PCA(3, random_state=RNG).fit(Xtr50); PC0 = pca50.transform(Xtr50)
fl50 = np.array([-1 if np.corrcoef(PC0[:, k], temps)[0, 1] < 0 else 1 for k in range(3)])
def pcs50(X): return pca50.transform(lt2.channel_scale(X50s, lt2.snv_rows(np.atleast_2d(X)), "pareto")[1]) * fl50
PCtr50 = pcs50(X50); lir_of = lambda X: X[:, mlo].sum(1) / np.maximum(X[:, mhi].sum(1), 1.0)
lc = lir_of(X50); lmap = PchipInterpolator(lc[np.argsort(lc)], temps[np.argsort(lc)], extrapolate=True)
def predict50(X):
    PCte = pcs50(X); sc = StandardScaler().fit(PCtr50); a, b = sc.transform(PCtr50), sc.transform(PCte); ym = temps.mean()
    return {"GPR": gpr_().fit(a, temps).predict(b), "Kernel Ridge (Poly)": KernelRidge(kernel="poly", alpha=0.1, degree=3, coef0=1).fit(a, temps - ym).predict(b) + ym,
            "Bayesian Ridge (poly3 PCs)": make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge()).fit(a, temps).predict(b),
            "Sensor Fusion (equal)": sf_general(PCtr50, temps, PCte), "LIR": lmap(lir_of(X))}
p30, p1 = predict50(R30), predict50(R1); pc30 = pcs50(R30)
pd.DataFrame({"mean_30pulse": {k: v.mean() for k, v in p30.items()}, "sd_30pulse": {k: v.std(ddof=1) for k, v in p30.items()}, "mean_single": {k: v.mean() for k, v in p1.items()}, "sd_single": {k: v.std(ddof=1) for k, v in p1.items()}}).round(3).to_csv(OUT / "replicates_350C.csv")
pd.DataFrame(p30).round(4).to_csv(OUT / "replicates_350C_predictions_30pulse.csv", index_label="replicate")
Cm = np.corrcoef(pc30.T); json.dump({"r12": Cm[0, 1], "r13": Cm[0, 2], "r23": Cm[1, 2], "p_r12": float(sps.pearsonr(pc30[:, 0], pc30[:, 1])[1]),
    "sd_pc": list(map(float, pc30.std(0, ddof=1))), "lir_rel_sd_30pulse": float(lir_of(R30).std(ddof=1) / lir_of(R30).mean()), "lir_rel_sd_single": float(lir_of(R1).std(ddof=1) / lir_of(R1).mean()),
    "sum_rel_sd_30pulse": float(R30.sum(1).std(ddof=1) / R30.sum(1).mean()), "sum_rel_sd_single": float(R1.sum(1).std(ddof=1) / R1.sum(1).mean())}, open(OUT / "replicates_350C_stats.json", "w"), indent=1)
ep, rows = {}, []
for mj, Xe in energy.items():
    p = predict50(Xe); ep[mj] = p; rows.append(dict(E_mJ=mj, total=Xe.sum(1).mean(), LIR_meas=lir_of(Xe).mean(), **{k: v.mean() for k, v in p.items()}, **{k + "_sem": v.std(ddof=1) / np.sqrt(len(v)) for k, v in p.items()}))
edf = pd.DataFrame(rows).set_index("E_mJ"); edf.round(4).to_csv(OUT / "energy_series.csv")
json.dump({"exponent": float(np.polyfit(np.log(edf.index.values.astype(float)), np.log(edf["total"].values), 1)[0]), **{f"shift_{k}": float(edf.loc[639, k] - edf.loc[573, k]) for k in p30}}, open(OUT / "energy_series_stats.json", "w"), indent=1)
sim50 = np.stack([X50[i] + np.random.default_rng(11 + i).normal(0, sigma_t[i], size=(N_SIM, X50.shape[1])) for i in range(n)])
sig50 = np.empty((n, 3)); cov50 = np.empty((n, 3, 3))
for j in range(n): Z = pcs50(sim50[j]); sig50[j] = Z.std(0, ddof=1); cov50[j] = np.cov(Z.T)
pd.DataFrame({lab: {"mean": (v := sf_general(PCtr50, temps, pcs50(R30), **kw)).mean(), "sd": v.std(ddof=1)} for lab, kw in [("equal", {}), ("diag_floor", dict(sig=sig50)), ("gls_floor", dict(cov=cov50))]}).T.round(3).to_csv(OUT / "sensor_fusion_weights_replicates_350C.csv"); log("replicates + energy")

# 8 data for Fig. 3 (decay fits at 100 C) and Fig. 7 (GPR at 210 C on the final cube)
win640 = (wl >= 635) & (wl <= 645); winpk = (wl >= 685.2) & (wl <= 690.2); dm = delays_full <= 1500.5; i100 = int(np.where(temps == 100)[0][0]); dec = {}
for lab, win in [("640", win640), ("688", winpk)]:
    base = cube[i100][win].mean(0)[-lt2.N_LAST:].mean(); y = cube[i100][win].mean(0)[dm] - base; tau, popt = mp.fit_biexp(delays_full[dm], y)
    dec[lab] = dict(y=y, fit=mp.biexp(delays_full[dm], *popt), tau=tau)
np.savez(OUT / "fig3_decays_100C.npz", delays=delays_full[dm], y640=dec["640"]["y"], f640=dec["640"]["fit"], y688=dec["688"]["y"], f688=dec["688"]["fit"], tau640=dec["640"]["tau"], tau688=dec["688"]["tau"])
Xs0, _ = lt2.channel_scale(X0, X0, "pareto"); pcaA = PCA(3, random_state=RNG).fit(Xs0); PCa = pcaA.transform(Xs0); fa = np.array([-1 if np.corrcoef(PCa[:, k], temps)[0, 1] < 0 else 1 for k in range(3)]); PCa = PCa * fa
xi_all = PLSRegression(1, scale=False).fit(PCa, temps).transform(PCa).ravel(); sgn = 1 if np.corrcoef(xi_all, temps)[0, 1] > 0 else -1; xi_all *= sgn
g1 = gpr_().fit(xi_all.reshape(-1, 1), temps); gridA = np.linspace(xi_all.min() - 0.2, xi_all.max() + 0.2, 400); muA, sdA = g1.predict(gridA.reshape(-1, 1), return_std=True)
i_ex = int(np.argmin(np.abs(temps - 210))); tr = np.array([j for j in range(n) if j != i_ex]); rng7 = np.random.default_rng(RNG)
Xte = lt2.snv_rows(Xm[i_ex] + rng7.normal(0, sigma_t[i_ex], size=(N_SIM, Xm.shape[1]))); Xtr_s, Xte_s = lt2.channel_scale(X0[tr], Xte, "pareto")
pcaF = PCA(3, random_state=RNG).fit(Xtr_s); a, b = pcaF.transform(Xtr_s), pcaF.transform(Xte_s); ff = np.array([-1 if np.corrcoef(a[:, k], temps[tr])[0, 1] < 0 else 1 for k in range(3)]); a, b = a * ff, b * ff
scF = StandardScaler().fit(a); p_te = gpr_().fit(scF.transform(a), temps[tr]).predict(scF.transform(b))
pls_f = PLSRegression(1, scale=False).fit(a, temps[tr]); xi_tr = pls_f.transform(a).ravel(); xi_te = pls_f.transform(b).ravel(); s2 = 1 if np.corrcoef(xi_tr, temps[tr])[0, 1] > 0 else -1; xi_tr, xi_te = xi_tr * s2, xi_te * s2
g2 = gpr_().fit(xi_tr.reshape(-1, 1), temps[tr]); gridB = np.linspace(xi_tr.min() - 0.2, xi_tr.max() + 0.2, 400); muB, sdB = g2.predict(gridB.reshape(-1, 1), return_std=True)
np.savez(OUT / "fig7_gpr_210C.npz", xi_all=xi_all, temps=temps, gridA=gridA, muA=muA, sdA=sdA, xi_tr=xi_tr, T_tr=temps[tr], xi_te=xi_te, p_te=p_te, gridB=gridB, muB=muB, sdB=sdB, T_true=temps[i_ex])
log("done")
