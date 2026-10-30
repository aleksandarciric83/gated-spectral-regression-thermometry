"""Functions defined interactively in the analysis kernel during the v11 session (lifetime re-implementation, sensor-fusion weighting test). Extracted for the record; they depend on the kernel variables cube, temps, wl, delays_full, sigma_t and on the module Fw/paper/make_paper_final_nofig.py (fit_biexp, biexp, predict_spline_T)."""
def tau_sim(band,dmax,seed,nsim=100):
    dm=delays_full<=dmax; t=delays_full[dm]; rng=np.random.default_rng(seed); out=np.full((n,nsim),np.nan)
    for i in range(n):
        y0=rate_win[band][i][dm]-base_[band][i]; sd=sigma_t[i]/np.sqrt(npix[band])
        for k in range(nsim):
            try: out[i,k]=mp.fit_biexp(t,y0+rng.normal(0,sd,size=y0.size))[0]
            except Exception: pass
    return out


def tau_real(band,dmax):
    dm=delays_full<=dmax; t=delays_full[dm]; out=np.full(n,np.nan)
    for i in range(n):
        try: out[i]=mp.fit_biexp(t,rate_win[band][i][dm]-base_[band][i])[0]
        except Exception: pass
    return out


def ms_fit(Tt,tau):
    best=None
    for p0 in [(tau[0],1e3,0.3),(tau[0],1e2,0.2),(tau[0],1e4,0.5),(tau[0]*1.1,3e2,0.15)]:
        try:
            popt,_=curve_fit(ms,Tt,tau,p0=p0,bounds=([1,0,0.01],[1e4,1e9,3.0]),maxfev=20000); sse=np.sum((ms(Tt,*popt)-tau)**2)
            if best is None or sse<best[0]: best=(sse,popt)
        except Exception: pass
    return None if best is None else best[1]


def ms_loto_ens(tau_sim):
    pred=np.full(tau_sim.shape,np.nan)
    for i in range(n):
        tr=np.arange(n)!=i; popt=ms_fit(temps[tr],np.nanmean(tau_sim[tr],axis=1))
        grid=np.linspace(temps[tr].min()-5,temps[tr].max()+5,4000); g=ms(grid,*popt)
        pred[i]=grid[np.argmin(np.abs(g[None,:]-tau_sim[i][:,None]),axis=1)]
    return pred


def rm_ens(pred): 
    q=np.nanmean((pred-temps[:,None])**2,axis=1); lo=temps<=300; return float(np.sqrt(q.mean())),float(np.sqrt(q[lo].mean())),float(np.sqrt(q[~lo].mean()))


def sf_pred(PC_tr,T_tr,PC_te,sig=None,grid_pts=4000):
    k=PC_tr.shape[1]; order=np.argsort(T_tr); Tu=T_tr[order]; fuse_T=np.linspace(Tu.min()-5,Tu.max()+5,grid_pts)
    g=np.stack([PchipInterpolator(Tu,PC_tr[order,i],extrapolate=True)(fuse_T) for i in range(k)],0)
    w=np.ones(k) if sig is None else 1.0/np.asarray(sig)**2
    out=np.empty(PC_te.shape[0])
    for j in range(PC_te.shape[0]):
        cost=sum(w[i]*(PC_te[j,i]-g[i])**2 for i in range(k)); out[j]=fuse_T[np.argmin(cost)]
    return out


def local_resid_sd(PC_tr,T_tr,w=7,deg=3):
    """per-component sd of leave-one-out residuals of a local cubic fit of the score vs T (within the training set)."""
    order=np.argsort(T_tr); T=T_tr[order]; P=PC_tr[order]; m=len(T); res=np.zeros_like(P)
    for i in range(m):
        idx=[j for j in np.argsort(np.abs(T-T[i])) if j!=i][:w-1]; idx=np.array(sorted(idx)); x=(T[idx]-T[i])/100.
        V=np.vander(x,deg+1,increasing=True); coef=np.linalg.lstsq(V,P[idx],rcond=None)[0]; res[i]=P[i]-coef[0]
    return np.sqrt((res**2).mean(0))


def loto_sf(gates,mode="z",weighting="equal",n_pcs=3):
    X=lt2.snv_rows(cube[:,:,list(gates)].reshape(n,-1)); pred=np.zeros(n); sigs=[]
    for i in range(n):
        tr=np.array([j for j in range(n) if j!=i]); Xtr_s,Xte_s=lt2.channel_scale(X[tr],X[i:i+1],mode)
        PCtr,PCte=gram_pca(Xtr_s,Xte_s,3); PCtr,PCte=lt2.align_pc_signs(PCtr,PCte,temps[tr])
        sig=None
        if weighting=="resid": sig=np.maximum(local_resid_sd(PCtr,temps[tr]),1e-9); sigs.append(sig)
        pred[i]=sf_pred(PCtr[:,:n_pcs],temps[tr],PCte[:,:n_pcs],None if sig is None else sig[:n_pcs])[0]
    return pred,(np.mean(sigs,0) if sigs else None)


# source of loto_gpr not retrievable: could not get source code


# source of gram_pca not retrievable: could not get source code


def loto_real(tr_):
    p=ms_loto_ens(tr_[:,None]); e=p[:,0]-temps; return float(np.sqrt(np.nanmean(e**2)))


# source of loto_k not retrievable: could not get source code


# source of loto_kw not retrievable: could not get source code


# source of loto_X not retrievable: could not get source code
