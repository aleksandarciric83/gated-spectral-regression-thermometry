"""v10 origin of the advantage above 300 C, in RMSE / bias / precision terms. Needs v10_out/pca_gpr.npz (gpr_detail_v10.py) and v10_out/z_per_setpoint_stats.csv (eval_scaling_v10.py z).
Writes v10_out/why_*.csv"""
import os, sys, re, json, mimetypes; mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np, pandas as pd, openpyxl, scipy.stats as sps
from scipy.interpolate import PchipInterpolator
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2
V = HERE / "v10_out"; FW = Path(os.environ.get("GATED_DATA_DIR", HERE.parent / "data"))   # folder with repeatability.xlsx and power_dependence.xlsx
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); wl = sp[0]["wavelength_nm"]; n = len(temps); cube = np.stack([s["rate"] for s in sp]); X50 = cube[:, :, 0]; lo, hi = wl < 675, wl > 680
lir_of = lambda X: np.atleast_2d(X)[:, lo].sum(1) / np.maximum(np.atleast_2d(X)[:, hi].sum(1), 1.0)
lc = lir_of(X50); Sr = lambda y: np.gradient(np.log(y), temps) * 100          # relative sensitivity [% per C]
tot = lambda g: cube[:, :, g].sum(1)
sens = pd.DataFrame({"T_C": temps, "LIR_rel_sens_pct_per_C": Sr(lc), "I(gate1)/I(gate0)_rel_sens_pct_per_C": -Sr(tot(1) / tot(0)), "I(gate2)/I(gate0)_rel_sens_pct_per_C": -Sr(tot(2) / tot(0)),
                     "line_to_band_peak_ratio": [float(X50[i][(wl > 686) & (wl < 690)].mean() / X50[i][(wl > 640) & (wl < 660)].mean()) for i in range(n)]})
sens.round(4).to_csv(V / "why_sensitivity.csv", index=False)
# LIR real-LOTO bias expressed as an equivalent change of ln LIR, and the 30-pulse scatter of the LIR at 350 C
def read_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True); rows = list(wb[sheet].iter_rows(values_only=True)); wb.close(); return rows
R30 = np.array(read_sheet(FW / "repeatability.xlsx", "350C_0-50us_635mJ")[1:1025], float)[:, 1:].T; l30 = lir_of(R30); sd_lir_pct = 100 * l30.std(ddof=1) / l30.mean()
lp = np.empty(n)
for i in range(n):
    tr = np.array([j for j in range(n) if j != i]); od = np.argsort(lc[tr]); lp[i] = np.clip(PchipInterpolator(lc[tr][od], temps[tr][od], extrapolate=True)(lc[i]), temps[tr].min() - 5, temps[tr].max() + 5)
e = lp - temps; res_pct = np.abs(e) * np.abs(Sr(lc)); ok = temps > 25
lirb = pd.DataFrame(dict(T_C=temps, LIR_bias_C=e, LIR_rel_sens=Sr(lc), residual_lnLIR_pct=res_pct))
summ = dict(sd_lir_30pulse_pct=sd_lir_pct, resid_sd_below250=float(np.sqrt(np.mean(res_pct[(temps < 250) & ok] ** 2))), resid_sd_from250=float(np.sqrt(np.mean(res_pct[temps >= 250] ** 2))),
            bias_C_low=[float(np.abs(e[(temps > 25) & (temps <= 100)]).min()), float(np.abs(e[(temps > 25) & (temps <= 100)]).max())], bias_C_310_340=[float(np.abs(e[(temps >= 310) & (temps <= 340)]).min()), float(np.abs(e[(temps >= 310) & (temps <= 340)]).max())])
lirb.round(4).to_csv(V / "why_lir_residuals.csv", index=False)
# PCA loadings of the unit-variance GPR model split by gate
pc = np.load(V / "pca_gpr.npz", allow_pickle=True); L = pc["loadings"]; gates = list(pc["gates"]); nw_ = len(wl); rows = []
for k in range(3):
    for gi, g in enumerate(gates):
        blk = L[k, gi::len(gates)]; tot_sq = (L[k] ** 2).sum()
        rows.append(dict(PC=k + 1, gate=int(g), share_sq_loading=float((blk ** 2).sum() / tot_sq), share_in_560_600nm=float((blk[(wl >= 560) & (wl <= 600)] ** 2).sum() / tot_sq), share_in_675_700nm=float((blk[(wl >= 675) & (wl <= 700)] ** 2).sum() / tot_sq)))
pd.DataFrame(rows).round(4).to_csv(V / "why_loading_shares.csv", index=False)
cols = {"wavelength_nm": wl}
for k in range(3):
    for gi, g in enumerate(gates): cols[f"PC{k+1}_gate{int(g)}"] = L[k, gi::len(gates)]
pd.DataFrame(cols).to_csv(V / "why_loadings.csv", index=False)
# set-point energy deviation against LIR and GPR bias (unit variance, ensemble)
En = np.array([float(re.findall(r"(\d+)mJ", Path(s_["path"]).name)[-1]) for s_ in sp]); dE = En - np.polyval(np.polyfit(temps, En, 3), temps)
out = []
if (V / "z_per_setpoint_stats.csv").exists():
    ps = pd.read_csv(V / "z_per_setpoint_stats.csv").set_index("T_C")
    for name, col in [("LIR", "LIR|bias"), ("GPR", "GPR|bias")]:
        for sel, mk in [("all except 25 C", temps != 25), ("T >= 250 C", temps >= 250)]:
            r, p = sps.pearsonr(dE[mk], ps[col].values[mk]); out.append(dict(readout=name, subset=sel, n=int(mk.sum()), r=r, p=p))
    pd.DataFrame(out).round(4).to_csv(V / "why_bias_vs_energy.csv", index=False)
r_te, p_te = sps.pearsonr(temps, En); summ.update(energy_vs_T_r=float(r_te), energy_vs_T_p=float(p_te), energy_25C=float(En[0]), energy_350C=float(En[-1]), energy_min=float(En.min()), energy_max=float(En.max()))
json.dump(summ, open(V / "why_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1)); print(pd.DataFrame(rows).round(3).to_string())
