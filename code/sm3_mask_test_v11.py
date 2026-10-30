
import sys, json, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2
SC="z"; RNG=lt2.RNG
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n=len(temps); cube=np.stack([s["rate"] for s in sp]); wl=sp[0]["wavelength_nm"]
def gpr_(): return GaussianProcessRegressor(kernel=C(1.0,(1e-2,1e4))*RBF(1.0,(1e-2,1e3))+WhiteKernel(1.0,(1e-5,1e3)), n_restarts_optimizer=3, normalize_y=True, random_state=RNG)
def br_(): return make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge())
def loto(X0, mk, k=3):
    pr=np.empty(n)
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i]); a,b=lt2.channel_scale(X0[tr],X0[i:i+1],SC); pca=PCA(3,random_state=RNG).fit(a); A=pca.transform(a); B=pca.transform(b)
        fl=np.array([-1 if np.corrcoef(A[:,q],temps[tr])[0,1]<0 else 1 for q in range(3)]); A=(A*fl)[:,:k]; B=(B*fl)[:,:k]; sc=StandardScaler().fit(A)
        pr[i]=mk().fit(sc.transform(A),temps[tr]).predict(sc.transform(B))[0]
    e=pr-temps; return float(np.sqrt(np.mean(e**2))), float(np.sqrt(np.mean(e[temps>300]**2)))
wins={"none":[], "Sm3+ lines (558-570, 594-606, 639-651, 706-722 nm)":[(558,570),(594,606),(639,651),(706,722)]}
out={}
for name,w in wins.items():
    mask=np.zeros(len(wl),bool)
    for a,b in w: mask|=(wl>=a)&(wl<=b)
    keep=~mask
    for gname,gates in (("gate0",[0]),("gates01",[0,1])):
        X=cube[:,keep,:][:,:,gates].reshape(n,-1); X0=lt2.snv_rows(X)
        g=loto(X0,gpr_); b=loto(X0,br_)
        out[f"{name}|{gname}"]=dict(n_masked_channels=int(mask.sum()), gpr_rmse=g[0], gpr_rmse_310_350=g[1], br_rmse=b[0], br_rmse_310_350=b[1])
        print(name, gname, out[f"{name}|{gname}"], flush=True)
json.dump(out, open(HERE/"v10_out"/"sm3_mask_test_v11.json","w"), indent=1)
