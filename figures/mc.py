import numpy as np, emcee, sys, os
from pipeline import *
zb,vb,kb,Cb=load_bao(); zhd,zhel,mb,Csn,n=load_sn()
L=Like(zhd,zhel,mb,Csn,zb,vb,kb,Cb, use_zhel=False)   # Fay convention
PR=dict(om=(0.05,0.60), w0=(-3.0,0.5), wa=(-4.0,3.0), eps=(-0.4,0.4))
def lnp_factory(we):
    def lnp(p):
        om,w0,wa=p[0],p[1],p[2]; eps=p[3] if we else 0.0
        if not PR['om'][0]<om<PR['om'][1]: return -np.inf
        if not PR['w0'][0]<w0<PR['w0'][1]: return -np.inf
        if not PR['wa'][0]<wa<PR['wa'][1]: return -np.inf
        if we and not PR['eps'][0]<eps<PR['eps'][1]: return -np.inf
        try: v=-0.5*L.chi2(om,w0,wa,eps)
        except Exception: return -np.inf
        return v if np.isfinite(v) else -np.inf
    return lnp
tag=sys.argv[1]; nsteps=int(sys.argv[2]); we = (tag=="eps")
nd=4 if we else 3; nw=32
bk=emcee.backends.HDFBackend(f"bk_{tag}.h5")
sam=emcee.EnsembleSampler(nw,nd,lnp_factory(we),backend=bk)
if bk.iteration==0:
    ctr=np.array([0.305,-0.89,-0.20,0.0])[:nd]
    p0=ctr+1e-3*np.random.default_rng(1).normal(size=(nw,nd))
    sam.run_mcmc(p0,nsteps,progress=False)
else:
    sam.run_mcmc(None,nsteps,progress=False)
print(f"{tag}: iterations={bk.iteration} acc={np.mean(sam.acceptance_fraction):.3f}")
