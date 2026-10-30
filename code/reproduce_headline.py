"""Headline check: real leave-one-temperature-out (LOTO) errors from the raw spectra.

Every set point is held out in turn; scaling, PCA and regression are refitted on the other 33 set points.
Printed: GPR and Bayesian ridge (gates 0-1, SNV, scaling to unit variance, 3 principal components) and the
luminescence intensity ratio (LIR) of the first gate (sum below 675 nm over sum above 680 nm, PCHIP calibration).
Values from this script (single measured spectrum per set point): GPR 0.396, Bayesian ridge 0.402, LIR 1.944 degC.
The manuscript quotes 0.40, 0.41 and 1.95 degC from 100 simulated detector-noise replicates per set point.

usage: python reproduce_headline.py            (raw spectra in ../data/spectra or GATED_SPECTRA_DIR)
"""
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lt2

spectra = lt2.load_spectra()
temps = np.array([s["temperature_c"] for s in spectra], float)
wl = spectra[0]["wavelength_nm"]
cube = np.stack([s["rate"] for s in spectra])          # (set points, channels, gates)
n = len(temps)
print(f"{n} set points, {cube.shape[1]} channels, {cube.shape[2]} gates, {temps.min():.0f}-{temps.max():.0f} degC")

X = lt2.snv_rows(cube[:, :, 0:2].reshape(n, -1))        # gates 0-1, channel-major, gate-minor
models = ["GPR", "Bayesian Ridge (poly3 PCs)"]
pred = lt2.real_loto(X, temps, "z", models)
for m in models:
    print(f"{m:28s} LOTO RMSE {lt2.rmse(pred[m], temps):.3f} degC (gates 0-1, unit variance)")

# classical intensity ratio at the first gate (gate 0, 50 ns delay)
g0 = cube[:, :, 0]
lir = g0[:, wl < 675].sum(1) / np.maximum(g0[:, wl > 680].sum(1), 1.0)
pl = np.full(n, np.nan)
for i in range(n):
    tr = np.arange(n) != i
    o = np.argsort(lir[tr])
    pl[i] = PchipInterpolator(lir[tr][o], temps[tr][o], extrapolate=True)(lir[i])
lo, hi = temps.min(), temps.max()
print(f"{'LIR (gate 0)':28s} LOTO RMSE {lt2.rmse(np.clip(pl, lo, hi), temps):.3f} degC (prediction limited to {lo:.0f}-{hi:.0f} degC)")
print(f"{'LIR (gate 0), unclipped':28s} LOTO RMSE {lt2.rmse(pl, temps):.3f} degC")
