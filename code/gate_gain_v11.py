
import sys, json, mimetypes
mimetypes.MimeTypes.read_windows_registry = lambda self, strict=True: None
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import lt2
RNG=lt2.RNG; sp = lt2.load_spectra(); temps = np.array([s["temperature_c"] for s in sp], float); n=len(temps); cube=np.stack([s["rate"] for s in sp])
def gpr_(): return GaussianProcessRegressor(kernel=C(1.0,(1e-2,1e4))*RBF(1.0,(1e-2,1e3))+WhiteKernel(1.0,(1e-5,1e3)), n_restarts_optimizer=3, normalize_y=True, random_state=RNG)
def preds(gates, ks):
    X0 = lt2.snv_rows(cube[:,:,gates].reshape(n,-1)); out={k:np.empty(n) for k in ks}
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i]); a,b=lt2.channel_scale(X0[tr],X0[i:i+1],"z"); pca=PCA(3,random_state=RNG).fit(a); A=pca.transform(a); B=pca.transform(b)
        fl=np.array([-1 if np.corrcoef(A[:,q],temps[tr])[0,1]<0 else 1 for q in range(3)]); A*=fl; B*=fl
        for k in ks:
            sc=StandardScaler().fit(A[:,:k]); out[k][i]=gpr_().fit(sc.transform(A[:,:k]),temps[tr]).predict(sc.transform(B[:,:k]))[0]
    return out
p0=preds([0],[1,2,3]); p01=preds([0,1],[2,3])
np.savez(HERE/"v10_out"/"gpr_loto_gate0_vs_gates01_v11.npz", temps=temps, g0_k1=p0[1], g0_k2=p0[2], g0_k3=p0[3], g01_k2=p01[2], g01_k3=p01[3])
print("saved")
