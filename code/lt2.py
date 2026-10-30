
from __future__ import annotations
import os, re, warnings, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator, RBFInterpolator, UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
warnings.filterwarnings("ignore")
WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))
from asc_io import parse_asc

SPECTRA_DIR = Path(os.environ.get("GATED_SPECTRA_DIR", WS.parent / "data" / "spectra"))   # raw .asc files from the Zenodo record
T_MIN, T_MAX = 25.0, 350.0
N_SIM, RNG, N_LAST, FUSION_GRID_PTS = 100, 42, 5, 4000
MODEL_NAMES = ["MLR (3 PCs)", "Poly Reg 3PCs (deg=3)", "3D Spline (PC1-3 TPS)", "Sensor Fusion (PC1-3 inv-var)", "SVR (RBF)", "GPR",
               "Kernel Ridge (RBF)", "Kernel Ridge (Poly)", "PLS (n=5)", "Bayesian Ridge (poly3 PCs)", "ElasticNet (poly3 PCs)", "MLP (64,64)"]

def load_spectra():
    by_temp = {}
    for path in sorted(SPECTRA_DIR.glob("*.asc")):
        if path.name == "Hg.asc": continue
        m = re.match(r"^(\d+)C", path.name)
        if not m: continue
        by_temp.setdefault(int(m.group(1)), []).append(path)
    spectra = []
    for t in sorted(by_temp):
        paths = by_temp[t]; reg = [p for p in paths if " reg" in p.name.lower()]
        spectra.append(parse_asc(reg[0] if reg else paths[0]))
    return [s for s in spectra if T_MIN <= s["temperature_c"] <= T_MAX]

def align_pc_signs(PC_tr, PC_te, T_tr):
    PC_tr, PC_te = PC_tr.copy(), PC_te.copy()
    for k in range(PC_tr.shape[1]):
        if np.corrcoef(PC_tr[:, k], T_tr)[0, 1] < 0:
            PC_tr[:, k] *= -1; PC_te[:, k] *= -1
    return PC_tr, PC_te

def sensor_fusion_eq(PC_tr, T_tr, PC_te, n_pcs=3, grid_pts=FUSION_GRID_PTS, sig_floor=1e-6):
    PC_tr = np.asarray(PC_tr, float)[:, :n_pcs]; PC_te = np.asarray(PC_te, float)[:, :n_pcs]; T_tr = np.asarray(T_tr, float)
    temps_u = np.unique(T_tr); fuse_T = np.linspace(temps_u.min() - 5.0, temps_u.max() + 5.0, grid_pts)
    g_grids, sig_grids = [], []
    for i in range(n_pcs):
        mean_i = np.array([PC_tr[T_tr == t, i].mean() for t in temps_u])
        sig_i = np.maximum(np.array([PC_tr[T_tr == t, i].std() for t in temps_u]), sig_floor)
        ord_i = np.argsort(temps_u)
        g = PchipInterpolator(temps_u[ord_i], mean_i[ord_i], extrapolate=True)
        s = PchipInterpolator(temps_u[ord_i], sig_i[ord_i], extrapolate=True)
        g_grids.append(g(fuse_T)); sig_grids.append(np.maximum(s(fuse_T), sig_floor))
    g_arr = np.stack(g_grids, 0); inv_sig2 = 1.0 / np.stack(sig_grids, 0) ** 2
    out = np.empty(PC_te.shape[0])
    for s0 in range(0, PC_te.shape[0], 2000):
        blk = PC_te[s0:s0 + 2000]; cost = np.zeros((blk.shape[0], fuse_T.size))
        for i in range(n_pcs):
            r = blk[:, i:i + 1] - g_arr[i][None, :]; cost += r * r * inv_sig2[i][None, :]
        out[s0:s0 + 2000] = fuse_T[np.argmin(cost, axis=1)]
    return out

def snv_rows(X):
    mu = X.mean(-1, keepdims=True); sd = X.std(-1, keepdims=True); sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd

def channel_scale(Xtr, Xte, mode):
    """Channel scaling fitted on training matrix Xtr (n_train x n_feat). mode: z | snv | pareto."""
    mu = Xtr.mean(0)
    if mode == "z": s = Xtr.std(0, ddof=0); s = np.where(s < 1e-12, 1.0, s)
    elif mode == "pareto": s = np.sqrt(Xtr.std(0, ddof=0)); s = np.where(s < 1e-12, 1.0, s)
    elif mode == "snv": s = np.ones(Xtr.shape[1])
    else: raise ValueError(mode)
    return (Xtr - mu) / s, (Xte - mu) / s

def fit_predict_all(Xtr_m, T_tr, Xte, mode, models=None, return_pc=False):
    """Xtr_m: SNV'd training matrix (one row per training T), Xte: SNV'd test rows. Returns dict model -> predictions."""
    models = models or MODEL_NAMES
    Xtr_s, Xte_s = channel_scale(Xtr_m, Xte, mode)
    pca = PCA(n_components=3, random_state=RNG)
    PC_tr = pca.fit_transform(Xtr_s); PC_te = pca.transform(Xte_s)
    PC_tr, PC_te = align_pc_signs(PC_tr, PC_te, T_tr)
    pc_sc = StandardScaler(); PC_tr_s = pc_sc.fit_transform(PC_tr); PC_te_s = pc_sc.transform(PC_te)
    out = {}
    def run(name, f):
        if name in models:
            try: out[name] = np.asarray(f(), float).ravel()
            except Exception: out[name] = np.full(len(Xte), np.nan)
    run("MLR (3 PCs)", lambda: LinearRegression().fit(PC_tr_s, T_tr).predict(PC_te_s))
    run("Poly Reg 3PCs (deg=3)", lambda: make_pipeline(PolynomialFeatures(3, include_bias=False), Ridge(alpha=1.0)).fit(PC_tr_s, T_tr).predict(PC_te_s))
    run("3D Spline (PC1-3 TPS)", lambda: RBFInterpolator(PC_tr, T_tr, kernel="thin_plate_spline", smoothing=0.1)(PC_te))
    run("Sensor Fusion (PC1-3 inv-var)", lambda: sensor_fusion_eq(PC_tr, T_tr, PC_te))
    run("SVR (RBF)", lambda: SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale").fit(PC_tr_s, T_tr).predict(PC_te_s))
    def gpr():
        ker = C(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(1.0, (1e-5, 1e3))
        return GaussianProcessRegressor(kernel=ker, n_restarts_optimizer=3, normalize_y=True, random_state=RNG).fit(PC_tr_s, T_tr).predict(PC_te_s)
    run("GPR", gpr)
    ym = T_tr.mean()
    run("Kernel Ridge (RBF)", lambda: KernelRidge(kernel="rbf", alpha=0.1).fit(PC_tr_s, T_tr - ym).predict(PC_te_s) + ym)
    run("Kernel Ridge (Poly)", lambda: KernelRidge(kernel="poly", alpha=0.1, degree=3, coef0=1).fit(PC_tr_s, T_tr - ym).predict(PC_te_s) + ym)
    run("PLS (n=5)", lambda: PLSRegression(n_components=max(min(5, len(T_tr) - 1, Xtr_s.shape[1]), 1), scale=False).fit(Xtr_s, T_tr).predict(Xte_s))
    run("Bayesian Ridge (poly3 PCs)", lambda: make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge()).fit(PC_tr_s, T_tr).predict(PC_te_s))
    run("ElasticNet (poly3 PCs)", lambda: make_pipeline(PolynomialFeatures(3, include_bias=False), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=RNG)).fit(PC_tr_s, T_tr).predict(PC_te_s))
    run("MLP (64,64)", lambda: MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", solver="lbfgs", max_iter=2000, random_state=RNG).fit(PC_tr_s, T_tr).predict(PC_te_s))
    if return_pc: return out, PC_tr, PC_te, pca
    return out

def real_loto(Xsnv, temps, mode, models=None):
    n = len(temps); pred = {m: np.full(n, np.nan) for m in (models or MODEL_NAMES)}
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        o = fit_predict_all(Xsnv[tr], temps[tr], Xsnv[i:i + 1], mode, models)
        for m, v in o.items(): pred[m][i] = v[0]
    return pred

def rmse(pred, temps, mask=None):
    e = pred - temps
    if mask is not None: e = e[mask]
    return float(np.sqrt(np.nanmean(e ** 2)))
