"""v10 GPR in detail (unit-variance scaling, gates 0-1 unless argv gives another range).
Writes v10_out/gpr_internals.json, gpr_calibration.csv, fig7_gpr_210C.npz, pca_gpr.npz"""
import sys, json, time, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2
OUT = HERE / "v10_out"; SC = "z"; RNG, N_SIM = lt2.RNG, lt2.N_SIM
A0, B0 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1); GATES = list(range(A0, B0 + 1))
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n = len(temps); cube = np.stack([s["rate"] for s in sp]); wl = sp[0]["wavelength_nm"]
sigma_t = np.array([float(np.median(s["rate"][:, -lt2.N_LAST:].std(axis=1))) for s in sp])
Xm = cube[:, :, GATES].reshape(n, -1); X0 = lt2.snv_rows(Xm)
def gpr_(): return GaussianProcessRegressor(kernel=C(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(1.0, (1e-5, 1e3)), n_restarts_optimizer=3, normalize_y=True, random_state=RNG)
def br_(): return make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge())
def prep(Xtr, Xte, T_tr):
    a, b = lt2.channel_scale(Xtr, Xte, SC); pca = PCA(3, random_state=RNG).fit(a); A, B = pca.transform(a), pca.transform(b)
    fl = np.array([-1 if np.corrcoef(A[:, k], T_tr)[0, 1] < 0 else 1 for k in range(3)]); A, B = A * fl, B * fl; sc = StandardScaler().fit(A); return A, B, sc.transform(A), sc.transform(B), pca, fl
# ---- all-data PCA (explained variance, loadings split by gate), GPR hyperparameters
A, _, As, _, pca, fl = prep(X0, X0, temps); L = pca.components_ * fl[:, None]
np.savez(OUT / "pca_gpr.npz", temps=temps, PC=A, evr=pca.explained_variance_ratio_, loadings=L, gates=np.array(GATES), wl=wl)
g = gpr_().fit(As, temps); k = g.kernel_; ystd = temps.std(); ym = temps.mean()
amp, ls, noise = k.k1.k1.constant_value, k.k1.k2.length_scale, k.k2.noise_level
info = dict(gates=GATES, scaling=SC, evr=[float(x) for x in pca.explained_variance_ratio_], amplitude_sd_C=float(np.sqrt(amp) * ystd), length_scale_standardised_PC_units=float(ls), noise_sd_C=float(np.sqrt(noise) * ystd), lml=float(g.log_marginal_likelihood_value_), n_train=n)
mu_a, sd_a = g.predict(As, return_std=True); info["posterior_sd_at_calibration_points_C_min_max"] = [float(sd_a.min()), float(sd_a.max())]
# ---- LOTO calibration of the predictive standard deviation (measured spectra, one per set point), GPR and Bayesian ridge
rows = []
for i in range(n):
    tr = np.array([j for j in range(n) if j != i]); A_, B_, As_, Bs_, _, _ = prep(X0[tr], X0[i:i + 1], temps[tr])
    gm = gpr_().fit(As_, temps[tr]); m1, s1 = gm.predict(Bs_, return_std=True); bm = br_().fit(As_, temps[tr]); m2, s2 = bm.predict(Bs_, return_std=True)
    rows.append(dict(T_C=temps[i], gpr_pred=m1[0], gpr_sd=s1[0], gpr_err=m1[0] - temps[i], br_pred=m2[0], br_sd=s2[0], br_err=m2[0] - temps[i]))
cal = pd.DataFrame(rows); cal.to_csv(OUT / "gpr_calibration.csv", index=False)
for nm in ("gpr", "br"):
    z = cal[f"{nm}_err"] / cal[f"{nm}_sd"]; info[f"{nm}_calibration"] = dict(rmse=float(np.sqrt((cal[f"{nm}_err"] ** 2).mean())), mean_sd=float(cal[f"{nm}_sd"].mean()), within_1sd=float((z.abs() <= 1).mean()), within_2sd=float((z.abs() <= 2).mean()), rms_z=float(np.sqrt((z ** 2).mean())))
# ---- number of PCs: LOTO RMSE on the measured spectra with the first k scores (GPR and Bayesian ridge, same range and scaling)
sub = {}
for nm, mk in (("GPR", gpr_), ("BR", br_)):
    for kpc in (1, 2, 3):
        pr = np.empty(n)
        for i in range(n):
            tr = np.array([j for j in range(n) if j != i]); A_, B_, As_, Bs_, _, _ = prep(X0[tr], X0[i:i + 1], temps[tr]); sc2 = StandardScaler().fit(A_[:, :kpc]); pr[i] = mk().fit(sc2.transform(A_[:, :kpc]), temps[tr]).predict(sc2.transform(B_[:, :kpc]))[0]
        sub[f"{nm}|{kpc}"] = float(np.sqrt(np.mean((pr - temps) ** 2)))
info["pc_subset_rmse"] = sub
# ---- 210 C fold: replicates, latent coordinate, 1D posterior along latent coordinate (as in the explanation figure)
def gp_post(xi, T):
    gg = gpr_().fit(xi.reshape(-1, 1), T); grid = np.linspace(xi.min() - 0.2, xi.max() + 0.2, 400); mu, sd = gg.predict(grid.reshape(-1, 1), return_std=True); return grid, mu, sd
PCa = A; pls = PLSRegression(1, scale=False).fit(PCa, temps); xi_all = pls.transform(PCa).ravel(); sg = 1 if np.corrcoef(xi_all, temps)[0, 1] > 0 else -1; xi_all *= sg
i_ex = int(np.argmin(np.abs(temps - 210))); tr = np.array([j for j in range(n) if j != i_ex]); rng7 = np.random.default_rng(RNG)
Xte = lt2.snv_rows(Xm[i_ex] + rng7.normal(0, sigma_t[i_ex], size=(N_SIM, Xm.shape[1]))); a_, b_, as_, bs_, _, ff = prep(X0[tr], Xte, temps[tr])
mdl = gpr_().fit(as_, temps[tr]); p_te, sd_te = mdl.predict(bs_, return_std=True)
pls_f = PLSRegression(1, scale=False).fit(a_, temps[tr]); xi_tr = pls_f.transform(a_).ravel(); xi_te = pls_f.transform(b_).ravel(); s2 = 1 if np.corrcoef(xi_tr, temps[tr])[0, 1] > 0 else -1; xi_tr, xi_te = xi_tr * s2, xi_te * s2
gA, mA, sA = gp_post(xi_all, temps); gB, mB, sB = gp_post(xi_tr, temps[tr])
np.savez(OUT / "fig7_gpr_210C.npz", xi_all=xi_all, temps=temps, xi_tr=xi_tr, T_tr=temps[tr], xi_te=xi_te, p_te=p_te, sd_te=sd_te, T_true=temps[i_ex], gates=np.array(GATES), gridA=gA, muA=mA, sdA=sA, gridB=gB, muB=mB, sdB=sB)
info["fold210"] = dict(bias=float(p_te.mean() - temps[i_ex]), precision=float(p_te.std(ddof=1)), predictive_sd=float(sd_te.mean()))
json.dump(info, open(OUT / "gpr_internals.json", "w"), indent=1); print(json.dumps(info, indent=1))
