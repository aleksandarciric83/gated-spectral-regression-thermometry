
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
sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n=len(temps); cube=np.stack([s["rate"] for s in sp])
def gpr_(): return GaussianProcessRegressor(kernel=C(1.0,(1e-2,1e4))*RBF(1.0,(1e-2,1e3))+WhiteKernel(1.0,(1e-5,1e3)), n_restarts_optimizer=3, normalize_y=True, random_state=RNG)
def br_(): return make_pipeline(PolynomialFeatures(3, include_bias=False), BayesianRidge())
def prep(Xtr, Xte, T_tr):
    a,b = lt2.channel_scale(Xtr, Xte, SC); pca=PCA(3, random_state=RNG).fit(a); A,B = pca.transform(a), pca.transform(b)
    fl = np.array([-1 if np.corrcoef(A[:,k],T_tr)[0,1]<0 else 1 for k in range(3)]); return A*fl, B*fl, pca
out={}
for name, gates in (("gate0",[0]),("gates01",[0,1])):
    Xm = cube[:,:,gates].reshape(n,-1); X0 = lt2.snv_rows(Xm)
    A,_,pca = prep(X0,X0,temps); evr=[float(x) for x in pca.explained_variance_ratio_]
    sub={}
    for nm, mk in (("GPR",gpr_),("BR",br_)):
        for k in (1,2,3):
            pr=np.empty(n)
            for i in range(n):
                tr=np.array([j for j in range(n) if j!=i]); A_,B_,_=prep(X0[tr],X0[i:i+1],temps[tr]); sc=StandardScaler().fit(A_[:,:k])
                pr[i]=mk().fit(sc.transform(A_[:,:k]),temps[tr]).predict(sc.transform(B_[:,:k]))[0]
            sub[f"{nm}|{k}"]=float(np.sqrt(np.mean((pr-temps)**2)))
    out[name]=dict(evr_pct=[100*x for x in evr], rmse=sub)
json.dump(out, open(HERE/"v10_out"/"pc_count_gate0_vs_gates01_v11.json","w"), indent=1); print(json.dumps(out, indent=1))
