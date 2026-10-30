
"""v11 figures (unit-variance scaling, one fixed contiguous gate range per model). Figures 1-3 as in v9.
Figures of the v6 manuscript in the Wiley Advanced Optical Materials profile (scientific-figures skill):
85 mm / 178 mm widths, Arial 10 pt, Okabe-Ito colours, lowercase bold panel labels, vector PDF + 600 dpi PNG + plotting CSV."""
import sys, json, re, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter, MultipleLocator
from scipy.interpolate import PchipInterpolator
from matplotlib.colors import LogNorm
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lt2
from miroslav_figure_style import set_publication_style, OKABE_ITO, MM_TO_INCH
IN = HERE / "v6_out"; IN7 = HERE / "v10_out"; OUT = HERE / "figs_v13"; OUT.mkdir(exist_ok=True)
set_publication_style(); mpl.rcParams.update({"savefig.bbox": "tight", "savefig.dpi": 600, "axes.unicode_minus": True, "mathtext.default": "regular", "legend.frameon": False,
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "axes.formatter.use_mathtext": True})
BLUE, VERM, GREEN, ORANGE, SKY, PURPLE, BLACK = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000"
W1, W2 = 85 * MM_TO_INCH, 178 * MM_TO_INCH
LAB = {"MLR (3 PCs)": "MLR", "Poly Reg 3PCs (deg=3)": "Polynomial regression (degree 3)", "3D Spline (PC1-3 TPS)": "3D thin-plate spline", "Sensor Fusion (PC1-3 inv-var)": "Sensor fusion (PC1-3, equal weights)",
       "SVR (RBF)": "SVR (RBF)", "GPR": "GPR", "Kernel Ridge (RBF)": "Kernel ridge (RBF)", "Kernel Ridge (Poly)": "Kernel ridge (polynomial)", "PLS (n=5)": "PLS (5 components)",
       "Bayesian Ridge (poly3 PCs)": "Bayesian ridge (degree 3)", "ElasticNet (poly3 PCs)": "ElasticNet (degree 3)", "MLP (64,64)": "MLP (64, 64)", "Sensor Fusion (equal)": "Sensor fusion"}
def panel(ax, letter, dx=-0.19, dy=1.03):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
def save(fig, name, data: pd.DataFrame):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=600); data.to_csv(OUT / f"{name}_data.csv", index=False); plt.close(fig); print("saved", name)
def tT(ax, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis; a.set_major_locator(MultipleLocator(50)); a.set_minor_locator(MultipleLocator(25))
def cbT(cb):
    cb.set_ticks(np.arange(50, 351, 50)); cb.ax.yaxis.set_minor_locator(MultipleLocator(25))
fmt_minus = FuncFormatter(lambda v, p: ("%g" % v).replace("-", "\u2212"))

spectra = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in spectra], float); wl = spectra[0]["wavelength_nm"]; dly = spectra[0]["delays_us"]
cube = np.stack([s["rate"] for s in spectra]); n = len(temps)
cl = np.load(IN / "classical_ensemble_v11.npz"); ps = pd.read_csv(IN7 / "z_per_setpoint_stats.csv", index_col="T_C"); ps_raw = ps.copy()
for _k in ["LIR", "tau640_MS", "tau688_MS"]:
    _z = ps[f"{_k}|rmse"] == 0
    if _z.any(): ps.loc[_z, [f"{_k}|rmse", f"{_k}|bias", f"{_k}|prec"]] = np.nan; print("masked zero-error (clipped) point(s):", _k, list(ps.index[_z]))
rs = pd.read_csv(IN7 / "z_range_summary.csv", index_col="model"); RK = json.load(open(IN7 / "z_ranking.json")); RANGES = RK["ranges"]
ms_par = json.load(open(IN / "mott_seitz_fit_all_points_v11.json")); kB = 8.617333e-5
ms = lambda T, a, c, e: a / (1 + c * np.exp(-e / (kB * (np.asarray(T, float) + 273.15))))
cmapT = mpl.colormaps["viridis"]; normT = mpl.colors.Normalize(25, 350)

# ---- Fig 1: (a) spectra at the 50 ns gate for all set points, (b) gated emission matrix at 210 C (wavelength x delay, colour = intensity)
i210 = int(np.argmin(np.abs(temps - 210)))
fig, axs = plt.subplots(1, 2, figsize=(W2, 72 * MM_TO_INCH), layout="constrained"); ax = axs[0]
for i in range(n): ax.plot(wl, cube[i, :, 0], color=cmapT(normT(temps[i])), lw=0.8)
ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Intensity [counts s$^{-1}$]"); ax.set_xlim(wl.min(), wl.max()); ax.set_ylim(bottom=0); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.text(642, ax.get_ylim()[1] * 0.50, r"4$\mathit{f}$5$\mathit{d}\rightarrow$4$\mathit{f}$", ha="center", fontsize=9); ax.text(688, ax.get_ylim()[1] * 1.0, "$^{5}$D$_{0}$\u2192$^{7}$F$_{J}$".replace("F$_{J}$", "F$_{0}$"), ha="left", va="top", fontsize=9)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=normT, cmap=cmapT), ax=ax, pad=0.02); cb.set_label("Temperature [\u00b0C]"); cb.outline.set_linewidth(0.6); cbT(cb); panel(ax, "a", -0.22)
ax = axs[1]; Zb = np.clip(cube[i210].T, 1e2, None); gcent = 25.0 + 50.0 * np.arange(len(dly))
imb = ax.pcolormesh(wl, gcent, Zb, shading="nearest", cmap="viridis", norm=LogNorm(vmin=1e2, vmax=1e5), rasterized=True)
ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Gate delay [\u00b5s]"); ax.set_xlim(wl.min(), wl.max()); ax.set_ylim(0, 2000)
cb = fig.colorbar(imb, ax=ax, pad=0.02); cb.set_label("Intensity [counts s$^{-1}$]"); cb.outline.set_linewidth(0.6); panel(ax, "b", -0.22)
save(fig, "Fig1", pd.concat([pd.DataFrame({"wavelength_nm": wl, **{f"I_{int(t)}C_gate0_counts_per_s": cube[i, :, 0] for i, t in enumerate(temps)}}), pd.DataFrame({f"I_210C_delay{d:.2f}us": cube[i210, :, j] for j, d in enumerate(dly)})], axis=1))

# ---- Fig 2: kinetics at 210 C: (a) spectra at all 40 delays, (b) the same spectra normalised to their maximum
win640 = (wl >= 635) & (wl <= 645); winpk = (wl >= 685.2) & (wl <= 690.2); cmapD = mpl.colormaps["viridis"]; normD = mpl.colors.Normalize(dly.min(), dly.max())
fig, axs = plt.subplots(1, 2, figsize=(W2, 70 * MM_TO_INCH), layout="constrained")
ax = axs[0]
for j, d in enumerate(dly): ax.plot(wl, cube[i210, :, j], color=cmapD(normD(d)), lw=0.7)
ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Intensity [counts s$^{-1}$]"); ax.set_xlim(wl.min(), wl.max()); ax.set_ylim(bottom=0); ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); panel(ax, "a", -0.22)
ax = axs[1]
from scipy.signal import medfilt, savgol_filter
SM = np.array([savgol_filter(medfilt(cube[i210, :, j], 5), 21, 3, mode="interp") for j in range(len(dly))]); nmax = SM.max(axis=1)   # display only: the analysis uses the unsmoothed data
for j in range(len(dly) - 1, -1, -1): ax.plot(wl, SM[j] / nmax[j], color=cmapD(normD(dly[j])), lw=0.6, alpha=0.9)
ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Normalised intensity"); ax.set_xlim(wl.min(), wl.max()); ax.set_ylim(0, 1.15); panel(ax, "b", -0.22)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=normD, cmap=cmapD), ax=axs, pad=0.02); cb.set_label("Gate delay [\u00b5s]"); cb.outline.set_linewidth(0.6)
save(fig, "Fig2", pd.concat([pd.DataFrame({"wavelength_nm": wl, **{f"I_delay{d:.2f}us": cube[i210, :, j] for j, d in enumerate(dly)}}), pd.DataFrame({f"Inorm_smoothed_delay{d:.2f}us": SM[j] / nmax[j] for j, d in enumerate(dly)})], axis=1))

# ---- Fig 3: conventional thermometry
dec = np.load(IN / "fig3_decays_100C_v11.npz"); tau640, tau688, lirm = cl["tau_640"], cl["tau_pk"], cl["lir_meas"]
fig = plt.figure(figsize=(W2, 175 * MM_TO_INCH), layout="constrained"); gs = fig.add_gridspec(3, 2); ax = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in zip("abcdef", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])}
Tg = np.linspace(25, 350, 300)
ax["a"].plot(temps, tau640, "o", color=BLUE, ms=3.5, label="640 nm"); ax["a"].plot(temps, tau688, "s", color=VERM, ms=3.5, label="688 nm")
ax["a"].plot(Tg, ms(Tg, *ms_par["MS_640"]), "-", color=BLUE, lw=1.0, alpha=0.8); ax["a"].plot(Tg, ms(Tg, *ms_par["MS_688"]), "-", color=VERM, lw=1.0, alpha=0.8); ax["a"].plot([], [], "-", color="0.35", lw=1.0, label="Mott-Seitz fits")
ax["a"].set_ylim(bottom=0); ax["a"].set_ylabel("Average lifetime [\u00b5s]"); ax["a"].legend(fontsize=8.5, loc="upper right")
tt = dec["delays"]; ax["b"].semilogy(tt, np.where(dec["y640"] > 0, dec["y640"], np.nan), "o", color=BLUE, ms=3, label="640 nm"); ax["b"].semilogy(tt, np.clip(dec["f640"], 1e-2, None), "-", color=BLUE, lw=1.2)
ax["b"].semilogy(tt, np.where(dec["y688"] > 0, dec["y688"], np.nan), "s", color=VERM, ms=3, label="688 nm"); ax["b"].semilogy(tt, np.clip(dec["f688"], 1e-2, None), "-", color=VERM, lw=1.2)
ax["b"].set_xlabel("Gate delay [\u00b5s]"); ax["b"].set_ylabel("Intensity [counts s$^{-1}$]"); ax["b"].set_xlim(0, 1950); ax["b"].legend(fontsize=8.5, loc="upper right")
ax["c"].plot(Tg, PchipInterpolator(temps, lirm)(Tg), "-", color=GREEN, lw=1.0); ax["c"].plot(temps, lirm, "^", color=GREEN, ms=3.5); ax["c"].set_ylabel("LIR"); ax["c"].set_ylim(bottom=0)
obs = [("tau640_MS", "640 nm lifetime", BLUE, "o"), ("tau688_MS", "688 nm lifetime", VERM, "s"), ("LIR", "LIR", GREEN, "^")]
pos = lambda v: np.where(np.asarray(v) > 0, v, np.nan)
for key, lab, col, mk in obs:
    ov = rs.loc[key, "overall"]
    ax["d"].plot(temps, pos(ps[f"{key}|rmse"]), marker=mk, color=col, ms=3.5, lw=0.9, label=f"{lab} ({ov:.2f} \u00b0C)")
    ax["e"].plot(temps, pos(np.abs(ps[f"{key}|bias"])), marker=mk, color=col, ms=3.5, lw=0.9)
    ax["f"].plot(temps, pos(ps[f"{key}|prec"]), marker=mk, color=col, ms=3.5, lw=0.9)
ax["d"].set_yscale("log"); ax["d"].set_ylim(1e-2, 1e3); ax["d"].set_ylabel("RMSE [\u00b0C]"); ax["d"].legend(fontsize=8, loc="upper right", ncol=1)
ax["e"].set_yscale("log"); ax["e"].set_ylim(1e-3, 1e2); ax["e"].set_ylabel("|Bias| [\u00b0C]")
ax["f"].set_yscale("log"); ax["f"].set_ylim(1e-3, 10); ax["f"].set_ylabel("Precision [\u00b0C]")
for k in "acdef": ax[k].set_xlabel("Temperature [\u00b0C]"); tT(ax[k])
ax["c"].set_xlabel("Temperature [\u00b0C]")
for k, l in zip("abcdef", "abcdef"): panel(ax[k], l, -0.21)
rows = [pd.DataFrame({"T_C": temps, "tau_640_us": tau640, "tau_688_us": tau688, "LIR": lirm})]
for key, lab, *_ in obs: rows.append(pd.DataFrame({f"{key}_rmse": ps[f"{key}|rmse"].values, f"{key}_abs_bias": np.abs(ps[f"{key}|bias"].values), f"{key}_precision": ps[f"{key}|prec"].values}))
save(fig, "Fig3", pd.concat(rows, axis=1))

from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
def rtxt(m):
    a, b = RANGES[m]; return ("gate %d" % a) if a == b else ("gates %d-%d" % (a, b))
# ---- Fig 4: PCA of the GPR model (gates 0-1, SNV + unit variance), 34 mean spectra
pg = np.load(IN7 / "pca_gpr.npz", allow_pickle=True); PC = pg["PC"]; g_ = [int(x) for x in pg["gates"]]
X0 = lt2.snv_rows(cube[:, :, g_].reshape(n, -1)); Xs_, _ = lt2.channel_scale(X0, X0, "z"); evr10 = PCA(10, random_state=lt2.RNG).fit(Xs_).explained_variance_ratio_
fig, axs = plt.subplots(1, 2, figsize=(W2, 70 * MM_TO_INCH), layout="constrained")
for k, (col, mk) in enumerate([(BLUE, "o"), (VERM, "s"), (GREEN, "^")]): axs[0].plot(temps, PC[:, k], marker=mk, color=col, ms=3.5, lw=0.9, label=f"PC{k+1} ({100*evr10[k]:.1f}%)")
axs[0].axhline(0, color="k", lw=0.6); tT(axs[0]); axs[0].set_xlabel("Temperature [\u00b0C]"); axs[0].set_ylabel("PC score"); axs[0].legend(fontsize=8.5, loc="upper left"); panel(axs[0], "a", -0.2)
x = np.arange(1, 11); axs[1].bar(x, 100 * evr10, color=SKY, width=0.7, label="Individual"); axs[1].plot(x, 100 * np.cumsum(evr10), "o-", color=VERM, ms=3.5, lw=0.9, label="Cumulative")
axs[1].set_xlabel("Principal component"); axs[1].set_ylabel("Explained variance [%]"); axs[1].set_xticks(x); axs[1].set_ylim(0, 105); axs[1].legend(fontsize=8.5, loc="center right"); panel(axs[1], "b", -0.2)
save(fig, "Fig4", pd.concat([pd.DataFrame({"T_C": temps, "PC1": PC[:, 0], "PC2": PC[:, 1], "PC3": PC[:, 2]}), pd.DataFrame({"component": x, "explained_variance_pct": 100 * evr10, "cumulative_pct": 100 * np.cumsum(evr10)})], axis=1))

# ---- Fig 5: overall RMSE of the twelve models (a) and RMSE landscape of the four leading models over all ranges [a, b] (b-e)
ml = rs.loc[lt2.MODEL_NAMES, "overall"].sort_values(); fig = plt.figure(figsize=(W2, 140 * MM_TO_INCH), layout="constrained"); gs5 = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.62])
ax = fig.add_subplot(gs5[0, :]); y = np.arange(len(ml)); ax.barh(y, ml.values, color=SKY, height=0.7); ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in ml.index]); ax.invert_yaxis()
for yi, (m, v) in enumerate(zip(ml.index, ml.values)): ax.text(v + 0.08, yi, f"{v:.2f}  ({rtxt(m)})", va="center", fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.axvline(rs.loc["LIR", "overall"], color=GREEN, ls="--", lw=1.2, label=f"LIR ({rs.loc['LIR','overall']:.2f} \u00b0C)"); ax.axvline(rs.loc["tau688_MS", "overall"], color=VERM, ls=":", lw=1.4, label=f"688 nm lifetime ({rs.loc['tau688_MS','overall']:.2f} \u00b0C)")
ax.set_xlabel("Overall LOTO RMSE [\u00b0C]"); ax.set_xlim(0, ml.max() * 1.30); ax.legend(loc="upper right", fontsize=8.5); panel(ax, "a", -0.17)
scanT = pd.read_csv(IN7 / "z_scan_table.csv"); heat_rows = []; axh = []; im = None; AMAX = 12
SHORT = {"Bayesian Ridge (poly3 PCs)": "Bayesian ridge", "GPR": "GPR", "Kernel Ridge (Poly)": "Kernel ridge (poly)", "Sensor Fusion (PC1-3 inv-var)": "Sensor fusion", "3D Spline (PC1-3 TPS)": "Thin-plate spline", "MLP (64,64)": "MLP"}
for k, m in enumerate(ml.index[:4]):
    a_ = fig.add_subplot(gs5[1, k]); axh.append(a_); s_ = scanT[scanT.model == m]; Z = np.full((AMAX + 1, 40), np.nan); s12 = s_[s_.a <= AMAX]
    Z[s12.a.values, s12.b.values] = s12.rmse.values
    im = a_.pcolormesh(np.arange(41) - 0.5, np.arange(AMAX + 2) - 0.5, Z, cmap="viridis", norm=LogNorm(vmin=0.3, vmax=12), rasterized=True)
    bb = s_.loc[s_.rmse.idxmin()]; a_.plot(bb.b, bb.a, marker="*", color="white", mec="k", mew=0.6, ms=9)
    a_.set_xlim(-0.5, 39.5); a_.set_ylim(AMAX + 0.5, -0.5); a_.set_xlabel("Last gate b"); a_.set_title(SHORT.get(m, LAB[m]), fontsize=8.5); a_.set_yticks([0, 4, 8, 12])
    if k == 0: a_.set_ylabel("First gate a")
    else: a_.tick_params(labelleft=False)
    panel(a_, "bcde"[k], -0.28 if k == 0 else -0.12, 1.12); heat_rows.append(pd.DataFrame({"model": m, "a": s_.a.values, "b": s_.b.values, "loto_rmse_measured_C": s_.rmse.values}))
fig.colorbar(im, ax=axh, shrink=0.9, pad=0.02, label="LOTO RMSE, measured spectra [\u00b0C]")
save(fig, "Fig5", pd.concat([pd.DataFrame({"model": [LAB[m] for m in ml.index], "overall_loto_rmse_C": ml.values, "gates": [rtxt(m) for m in ml.index]}), pd.concat(heat_rows, ignore_index=True).add_prefix("heat_")], axis=1))

# ---- Fig 6: three best models and the LIR vs temperature (logarithmic axes), GPR emphasised
top = list(ml.index[:3]); sty = {top[0]: (BLUE, "o"), top[1]: (VERM, "s"), top[2]: (GREEN, "^")}
fig, axs = plt.subplots(3, 1, figsize=(W2 * 0.72, 150 * MM_TO_INCH), sharex=True, layout="constrained"); rows = [pd.DataFrame({"T_C": temps})]
for m in top:
    col, mk = sty[m]; lw = 1.6 if m == "GPR" else 0.9
    for a, key in zip(axs, ["rmse", "bias", "prec"]): a.plot(temps, pos(np.abs(ps[f"{m}|{key}"])), marker=mk, color=col, ms=3.5, lw=lw, label=f"{LAB[m]} ({rtxt(m)})")
    rows.append(pd.DataFrame({f"{LAB[m]}_{('abs_bias' if k == 'bias' else k)}": (np.abs(ps[f"{m}|{k}"].values) if k == "bias" else ps[f"{m}|{k}"].values) for k in ["rmse", "bias", "prec"]}))
for a, key in zip(axs, ["rmse", "bias", "prec"]): a.plot(temps, pos(np.abs(ps[f"LIR|{key}"])), "-", color="0.35", lw=1.0, ls="--", label="LIR")
rows.append(pd.DataFrame({f"LIR_{('abs_bias' if k == 'bias' else k)}": (np.abs(ps[f"LIR|{k}"].values) if k == "bias" else ps[f"LIR|{k}"].values) for k in ["rmse", "bias", "prec"]}))
axs[0].set_yscale("log"); axs[0].set_ylim(1e-2, 2e1); axs[0].set_ylabel("RMSE [\u00b0C]")
axs[1].set_yscale("log"); axs[1].set_ylim(1e-3, 2e1); axs[1].set_ylabel("|Bias| [\u00b0C]")
axs[2].set_yscale("log"); axs[2].set_ylim(1e-3, 1); axs[2].set_ylabel("Precision [\u00b0C]"); axs[2].set_xlabel("Temperature [\u00b0C]"); tT(axs[2]); axs[2].legend(fontsize=8, loc="lower left", ncol=2, columnspacing=1.0, handlelength=1.6)
for a, l in zip(axs, "abc"): panel(a, l, -0.14)
save(fig, "Fig6", pd.concat(rows, axis=1))

# ---- Fig 7: GPR explanation (a, b: posterior along the latent coordinate and the 210 C fold)
g = np.load(IN7 / "fig7_gpr_210C.npz"); cal = pd.read_csv(IN7 / "gpr_calibration.csv"); T_true = float(g["T_true"])
fig = plt.figure(figsize=(W2, 72 * MM_TO_INCH), layout="constrained"); gs7 = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0]); axs = [fig.add_subplot(gs7[0, k]) for k in range(2)]
ax = axs[0]; ax.fill_between(g["gridA"], g["muA"] - 2 * g["sdA"], g["muA"] + 2 * g["sdA"], color=SKY, alpha=0.45, lw=0, label="Posterior \u00b1 2 s.d."); ax.plot(g["gridA"], g["muA"], color=BLUE, lw=1.4, label="Posterior mean")
ax.plot(g["xi_all"], g["temps"], "o", color=VERM, ms=3.5, label="Calibration points"); ax.set_xlabel("Latent coordinate from PC1-PC3"); ax.set_ylabel("Temperature [\u00b0C]"); ax.set_ylim(0, 380); tT(ax, "y"); ax.legend(fontsize=8, loc="upper left"); panel(ax, "a", -0.24)
ax = axs[1]; ax.fill_between(g["gridB"], g["muB"] - 2 * g["sdB"], g["muB"] + 2 * g["sdB"], color=SKY, alpha=0.4, lw=0); ax.plot(g["gridB"], g["muB"], color=BLUE, lw=1.2)
ax.plot(g["xi_tr"], g["T_tr"], "o", color=VERM, ms=4.5, label="Training set (LOTO)"); ax.scatter(g["xi_te"], g["p_te"], s=70, color=PURPLE, alpha=0.35, label="100 replicates", zorder=4, lw=0)
xc, Th = float(np.mean(g["xi_te"])), float(np.mean(g["p_te"])); ax.plot(xc, T_true, "D", color="k", ms=4.5, zorder=6, label=f"True {T_true:.0f} \u00b0C"); ax.plot(xc, Th, "D", color=GREEN, ms=3, zorder=7, label=f"Mean prediction {Th:.2f} \u00b0C")
sel = (g["muB"] > T_true - 14) & (g["muB"] < T_true + 14); ax.set_xlim(g["gridB"][sel].min(), g["gridB"][sel].max()); ax.set_ylim(T_true - 14, T_true + 14)
ax.set_xlabel("Latent coordinate from PC1-PC3"); ax.tick_params(labelleft=True); ax.legend(fontsize=7.5, loc="upper left"); panel(ax, "b", -0.2)
b_, p_, r_ = Th - T_true, float(np.std(g["p_te"], ddof=1)), float(np.sqrt(np.mean((g["p_te"] - T_true) ** 2)))
ax.text(0.97, 0.04, f"|bias| = {abs(b_):.2f} \u00b0C\nprecision = {p_:.2f} \u00b0C\nRMSE = {r_:.2f} \u00b0C".replace("-", "\u2212"), transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
save(fig, "Fig7", pd.concat([pd.DataFrame({"xi_calibration": g["xi_all"], "T_calibration_C": g["temps"]}), pd.DataFrame({"xi_grid_a": g["gridA"], "post_mean_a": g["muA"], "post_sd_a": g["sdA"]}), pd.DataFrame({"xi_train_loto": g["xi_tr"], "T_train_loto_C": g["T_tr"]}),
                              pd.DataFrame({"xi_test": g["xi_te"], "pred_3D_GPR_C": g["p_te"]}), pd.DataFrame({"xi_grid_b": g["gridB"], "post_mean_b": g["muB"], "post_sd_b": g["sdB"]})], axis=1))

# ---- Fig 8: origin of the advantage above 300 C
sen = pd.read_csv(IN7 / "why_sensitivity.csv"); ld = pd.read_csv(IN7 / "why_loadings.csv"); gl_ = [int(x) for x in pg["gates"]]
fig = plt.figure(figsize=(W2, 105 * MM_TO_INCH), layout="constrained"); gs8 = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25]); axa = fig.add_subplot(gs8[:, 0]); axb = fig.add_subplot(gs8[0, 1]); axc = fig.add_subplot(gs8[1, 1], sharex=axb, sharey=axb)
rows8 = []
axa.plot(sen["T_C"], pos(np.abs(sen["LIR_rel_sens_pct_per_C"])), marker="o", ms=3.2, lw=0.9, color="0.35"); rows8.append(pd.DataFrame({"panel": "a", "series": "LIR (50 ns)", "x": sen["T_C"], "y": sen["LIR_rel_sens_pct_per_C"]}))
axa.set_yscale("log"); axa.set_ylim(0.1, 2); axa.set_xlabel("Temperature [\u00b0C]"); axa.set_ylabel("Relative sensitivity of the LIR [%/\u00b0C]"); tT(axa); axa.set_yticks([0.1, 0.2, 0.5, 1, 2]); axa.yaxis.set_major_formatter(FuncFormatter(lambda v, p: "%g" % v)); axa.yaxis.set_minor_formatter(NullFormatter()); panel(axa, "a", -0.24)
for ax_, gi, nm, lt in [(axb, gl_[0], "gate 0 (delay 50 ns)", "b"), (axc, gl_[1], "gate 1 (delay 50 \u00b5s)", "c")]:
    ax_.axvspan(wl[0], 675, color="0.85", lw=0, alpha=0.6); ax_.axvspan(680, wl[-1], color=ORANGE, lw=0, alpha=0.15)
    for k, c in ((1, BLUE), (2, VERM), (3, GREEN)): ax_.plot(ld["wavelength_nm"], ld[f"PC{k}_gate{gi}"], color=c, lw=0.9, label=f"PC{k}"); rows8.append(pd.DataFrame({"panel": lt, "series": f"PC{k} {nm}", "x": ld["wavelength_nm"], "y": ld[f"PC{k}_gate{gi}"]}))
    ax_.axhline(0, color="k", lw=0.5); ax_.set_ylabel("Loading"); ax_.text(0.02, 0.92, nm, transform=ax_.transAxes, fontsize=8, va="top"); panel(ax_, lt, -0.14)
axb.legend(fontsize=8, loc="upper right", ncol=3, columnspacing=0.9, handlelength=1.4); axb.tick_params(labelbottom=False); axc.set_xlabel("Wavelength [nm]"); axc.set_xlim(wl[0], wl[-1])
axc.text(0.02, 0.06, "LIR numerator (<675 nm)", transform=axc.transAxes, fontsize=7.5, color="0.3"); axc.text(0.98, 0.06, "denominator (>680 nm)", transform=axc.transAxes, fontsize=7.5, color="0.3", ha="right")
save(fig, "Fig8", pd.concat(rows8, ignore_index=True))

# ---- Fig 9: pulse energy and repeatability (LIR and the 50 ns models of the multivariate analysis, trained on the 34 mean spectra)
rep = pd.read_csv(HERE / "repeatability_350C_unit_variance_50ns_v11.csv"); en = pd.read_csv(HERE / "energy_series_unit_variance_50ns_v11.csv", index_col="E_mJ")
methods = [("LIR", "LIR", "0.4"), ("GPR", "GPR", BLUE), ("Kernel Ridge (Poly)", "Kernel\nridge", ORANGE), ("Bayesian Ridge (poly3 PCs)", "Bayesian\nridge", VERM), ("Sensor Fusion (equal)", "Sensor\nfusion", GREEN)]
fig = plt.figure(figsize=(W2, 72 * MM_TO_INCH), layout="constrained"); gs9 = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0]); axs = [fig.add_subplot(gs9[0, k]) for k in range(2)]
ax = axs[0]; rng_j = np.random.default_rng(3); ytop = 0
for k, (key, lab, col) in enumerate(methods):
    v = rep[key].values; ax.scatter(k + rng_j.uniform(-0.18, 0.18, len(v)), v, s=9, color=col, alpha=0.6, lw=0); ax.hlines(v.mean(), k - 0.3, k + 0.3, color="k", lw=1.6)
ax.axhline(350, color="k", ls=":", lw=0.8); ax.text(-0.45, 350.5, "set point", fontsize=8, va="bottom"); ax.set_xticks(range(len(methods))); ax.set_xticklabels([m[1] for m in methods], fontsize=7); ax.set_ylabel("Predicted temperature [\u00b0C]"); panel(ax, "a", -0.2)
ax = axs[1]; E0 = en.index.values.astype(float)
for key, lab, col in methods: ax.errorbar(E0, en[key].values - en.loc[573, key], yerr=en[key + "_sem"].values, marker="o", ms=4, lw=1.1, capsize=2, color=col, label=lab.replace("\n", " "))
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Mean pulse energy [mJ]"); ax.set_ylabel("Shift of predicted temperature [\u00b0C]"); ax.set_ylim(-6, 12); ax.legend(fontsize=7, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.4); panel(ax, "b", -0.24)
save(fig, "Fig9", pd.concat([rep.add_prefix("pred_30pulse_"), en.reset_index()], axis=1))

# ---- Fig 10: scaling comparison (SNV only, Pareto, unit variance), every model at its own range under each scaling
# ---- Fig 10: scaling comparison (SNV only, Pareto, unit variance), every model at its own range under each scaling (single panel)
S_ = {t: pd.read_csv(IN7 / f"{t}_range_summary.csv", index_col="model") for t in ("snv", "pareto", "z")}; SB = {t: pd.read_csv(IN7 / f"{t}_best_range.csv", index_col="model") for t in ("snv", "pareto", "z")}
order = list(ml.index); fig, axa = plt.subplots(figsize=(W2, 88 * MM_TO_INCH), layout="constrained")
yy = np.arange(len(order)); cols = {"snv": SKY, "pareto": ORANGE, "z": BLUE}; labs = {"snv": "SNV only", "pareto": "Pareto", "z": "Unit variance"}; rows9 = []
for k, t in enumerate(("snv", "pareto", "z")):
    v = S_[t].loc[order, "overall"].values; axa.scatter(v, yy + (k - 1) * 0.2, s=26, color=cols[t], label=labs[t], zorder=3, marker=["s", "D", "o"][k]); rows9.append(pd.DataFrame({"series": labs[t], "model": [LAB[m] for m in order], "overall_rmse_C": v, "range": [f"{int(SB[t].loc[m,'a'])}-{int(SB[t].loc[m,'b'])}" for m in order]}))
axa.grid(axis="y", color="0.9", lw=0.6, zorder=0)
axa.axvline(S_["z"].loc["LIR", "overall"], color=GREEN, ls="--", lw=1.0, label="LIR"); axa.set_yticks(yy); axa.set_yticklabels([LAB[m] for m in order]); axa.invert_yaxis(); axa.set_xscale("log"); axa.set_xlim(0.3, 12); axa.set_xlabel("Overall LOTO RMSE [\u00b0C]"); axa.legend(fontsize=8, loc="upper right")
save(fig, "Fig10", pd.concat(rows9, ignore_index=True)); print("all figures done")
