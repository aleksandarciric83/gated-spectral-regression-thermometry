# Gated spectral regression for luminescence thermometry of &gamma;-Al<sub>2</sub>O<sub>3</sub>:Sm<sup>2+</sup>

Analysis code for the manuscript "Beyond Intensity Ratios: Gated Spectral Regression for Luminescence Thermometry of &gamma;-Al<sub>2</sub>O<sub>3</sub>:Sm<sup>2+</sup>" (A. Ćirić, S. Stojadinović, M. Kemere, V. Pankratov, A. Sarakovskis, M. D. Dramićanin).

Gated ICCD spectra of an Sm<sup>2+</sup>-doped &gamma;-Al<sub>2</sub>O<sub>3</sub> coating (plasma electrolytic oxidation on aluminium) are treated as one wavelength-delay data set. After SNV normalisation, channel scaling and PCA (3 components), twelve regression models are compared with the luminescence intensity ratio (LIR) and with lifetimes by leave-one-temperature-out (LOTO) validation over 34 set points from 25 to 350 &deg;C.

* Data: Zenodo, [doi:10.5281/zenodo.23817355](https://doi.org/10.5281/zenodo.23817355) (licence CC BY 4.0)
* Code: this repository (licence GPL-3.0)

## Quick start

```
pip install -r requirements.txt
python code/fetch_data.py            # downloads the Zenodo record into data/ and code/
python code/reproduce_headline.py    # about 10 s
python code/make_figures_v13.py      # about 30 s, writes code/figs_v13/ (Fig1-Fig10, png, pdf, _data.csv)
```

`fetch_data.py` uses only the Python standard library. It puts the raw spectra in `data/spectra/`, the two Excel files in `data/`, and the derived tables (`v6_out/`, `v10_out/` and two csv files) in `code/`. Alternatively, download the files from the record by hand and unpack them in the same places. The spectra folder can be elsewhere if the environment variable `GATED_SPECTRA_DIR` points to it.

Expected output of `reproduce_headline.py` (real LOTO on the measured spectra, one spectrum per set point):

```
GPR                          LOTO RMSE 0.396 degC (gates 0-1, unit variance)
Bayesian Ridge (poly3 PCs)   LOTO RMSE 0.402 degC (gates 0-1, unit variance)
LIR (gate 0)                 LOTO RMSE 1.944 degC (prediction limited to 25-350 degC)
```

The manuscript quotes 0.40, 0.41 and 1.95 &deg;C, which come from 100 simulated detector-noise replicates per set point (the noise is small against the LOTO error, so both give the same values within 0.01 &deg;C).

## Method in brief

1. Spectra of 1024 channels at 40 gates (gate g opens at 0.05 + 50g &micro;s, width 50 &micro;s) per set point are read as counts per second (`asc_io.py`, `lt2.load_spectra`).
2. Each regression model uses its own contiguous range of gates. The channels of the chosen gates are stacked into one vector, normalised by the standard normal variate (SNV) and scaled per channel (`lt2.channel_scale`: `z` unit variance, `pareto`, `snv` no scaling).
3. PCA to three components, signs aligned to the temperature, then one of twelve regressors (`lt2.fit_predict_all`): MLR, polynomial regression, thin-plate spline, sensor fusion (equal weights), SVR, GPR, kernel ridge (RBF, polynomial), PLS, Bayesian ridge, elastic net, MLP.
4. Leave-one-temperature-out: scaling, PCA and regression are refitted without the held-out set point (`lt2.real_loto`). The same gate range and scaling are used in every fold.

## Scripts

| Script | Purpose |
| --- | --- |
| `lt2.py`, `lt2f.py`, `asc_io.py` | Core pipeline (reading, SNV, scaling, PCA, 12 regressors, LOTO); `lt2f` replaces the PCA step by an exact Gram-matrix decomposition (same scores, faster) |
| `reproduce_headline.py` | LOTO errors of GPR, Bayesian ridge and the LIR |
| `scan_ranges_v9.py`, `scan_ranges_z_v9.py`, `scan10_v10.py` | Exhaustive scans of all 820 contiguous gate ranges (Pareto, unit variance, SNV only; MLP for ranges starting at gates 0 to 3); one worker per process |
| `eval_scaling_v10.py {z,pareto,snv}` | Aggregates the scans, best range per model, per-set-point RMSE, bias and precision, bootstrap, noise ensemble |
| `gpr_detail_v10.py`, `why_v10.py` | GPR internals and PCA loadings (gates 0-1), sensitivity of the LIR and origin of the advantage above 300 &deg;C |
| `nested_gen_v9.py`, `nested_agg_gen_v9.py`, `ensemble_one_v9.py` | Nested last-gate selection (leave-two-out inner loop, a diagnostic) and the noise ensemble of one model |
| `subset_scan_v10.py`, `subset_search_v10.py`, `subset_greedy_v10.py` | GPR with arbitrary gate subsets (exhaustive up to three gates, annealing, greedy) |
| `gate_gain_v11.py`, `pc_count_v11.py`, `sm3_mask_test_v11.py` | Gate 0 against gates 0-1, number of principal components, masking of the Sm<sup>3+</sup>-like lines |
| `make_figures_v13.py`, `miroslav_figure_style.py` | Figures 1 to 10 of the manuscript and their data tables |

Run the scan scripts as `python scan_ranges_z_v9.py <worker> <n_workers>` in `n_workers` separate processes. They append one json line per range to `v9_out/` or `v10_out/` and can be resumed. The full scan of 820 ranges for the 12 models took about 1.5 h on 8 cores.

## What is and is not reproducible from this repository

* Reproduced and checked on the deposited data: the headline errors above, and all ten figure tables. `make_figures_v13.py` returns figure data identical to those used in the manuscript.
* The scan outputs (jsonl files, several tens of MB) are not deposited. The scan scripts regenerate them.
* The classical readouts (lifetimes from bi-exponential fits, Mott-Seitz calibration, noise ensembles) were computed in an earlier script and in an interactive session. Their results are deposited as `v6_out/classical_ensemble_v11.npz`, `mott_seitz_fit_all_points_v11.json` and `fig3_decays_100C_v11.npz`. The folder `record/` holds the earlier script and the functions of that session for reference. They do not run on their own, because they depend on files and variables that were not kept. The LIR error is recomputed in `reproduce_headline.py`.
* `record/` also holds the noise-world scripts (they need two synthetic arrays that are not deposited) and the despiking test (it needs a despiked cube that is not deposited).
* The reported errors are optimistic. Gate range and scaling were selected on the same 34 set points that serve for validation, and all set points come from one heating run.

## Software

Tested with Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, pandas 2.3.3, Matplotlib 3.11.2 and openpyxl 3.1.5 (only for the Excel files). Other versions may work and may give slightly different numbers. The figure style uses Arial when it is installed.

## AI assistance

The Python code was created with the assistance of Claude Sonnet 5.5 (Anthropic).

## Licence and citation

Code: GNU General Public License v3.0 (`LICENSE`). Data: Creative Commons Attribution 4.0 (Zenodo record). Please cite the data record [doi:10.5281/zenodo.23817355](https://doi.org/10.5281/zenodo.23817355) and the manuscript. `CITATION.cff` has the metadata.
