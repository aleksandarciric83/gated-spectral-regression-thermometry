
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
RNG=lt2.RNG; sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n=len(temps)
cubes={"original":np.stack([s["rate"] for s in sp]), "despiked":np.load(HERE/"cube_despiked_v11.npy")}
def gpr_(): return GaussianProcessRegressor(kernel=C(1.0,(1e-2,1e4))*RBF(1.0,(1e-2,1e3))+WhiteKernel(1.0,(1e-5,1e3)), n_restarts_optimizer=3, normalize_y=True, random_state=RNG)
def br_(): return make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge())
def loto(X0, mk):
    pr=np.empty(n)
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i]); a,b=lt2.channel_scale(X0[tr],X0[i:i+1],"z"); pca=PCA(3,random_state=RNG).fit(a); A=pca.transform(a); B=pca.transform(b)
        fl=np.array([-1 if np.corrcoef(A[:,q],temps[tr])[0,1]<0 else 1 for q in range(3)]); A*=fl; B*=fl; sc=StandardScaler().fit(A)
        pr[i]=mk().fit(sc.transform(A),temps[tr]).predict(sc.transform(B))[0]
    return float(np.sqrt(np.mean((pr-temps)**2)))
out={}
for cname,cu in cubes.items():
    for gname,gates in (("0",[0]),("0-1",[0,1]),("0-2",[0,1,2]),("0-3",[0,1,2,3]),("0-9",list(range(10))),("0-39",list(range(40)))):
        X0=lt2.snv_rows(cu[:,:,gates].reshape(n,-1)); out[f"{cname}|{gname}"]=dict(gpr=loto(X0,gpr_), br=loto(X0,br_)); print(cname,gname,out[f"{cname}|{gname}"],flush=True)
json.dump(out,open(HERE/"v10_out"/"despike_test_v11.json","w"),indent=1)
