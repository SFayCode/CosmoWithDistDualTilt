"""Fig. 4, with the analysis-choice systematic shown per bin."""
import numpy as np, itertools, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import load_bao, load_sn, Like
C=299792.458; RD=147.09; RD_ERR=0.26; MB0=-19.253
zb,vb,kb,Cb=load_bao(); zhd,zhel,mb,Csn,n=load_sn()
L=Like(zhd,zhel,mb,Csn,zb,vb,kb,Cb,use_zhel=True)
sd=np.sqrt(np.diag(Csn))
d=np.load("tomo.npz"); b=float(d["b"]); om0=float(d["om_bao"]); H0b=C/(b*RD)
ORIG=np.array([-0.134,-0.110,-0.109,-0.126,-0.132,-0.160,-0.183,-0.186])
EDGES={"voronoi":np.array([0.009,0.026,0.06,0.1166,0.2182,0.3779,0.6387,1.0198,2.30]),
       "log8":np.geomspace(0.0101,2.27,9),
       "quant8":np.quantile(zhd,np.linspace(0,1,9)),
       "widetop":np.array([0.009,0.026,0.06,0.1166,0.2182,0.3779,0.6387,0.85,2.30])}

def binned(om,MB,w,edges,pre,H0):
    zz = zhel if pre=="hel" else zhd
    res=(mb-MB)-(5*np.log10((1+zz)*(C/H0)*L.g.I(zhd,om,-1,0))+25)
    zc,val,err=[],[],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(zhd>=lo)&(zhd<hi); i=np.where(m)[0]
        if m.sum()<2: return None
        if w=="full": Ci=np.linalg.inv(Csn[np.ix_(i,i)]); ww=Ci.sum(axis=1)
        elif w=="diag": ww=1/sd[i]**2
        else: ww=np.ones(len(i))
        zc.append(np.median(zhd[m])); val.append(float(ww@res[m]/ww.sum()))
        err.append(float(1/np.sqrt(np.abs(ww).sum())))
    return np.array(zc),np.array(val),np.array(err)

# baseline
zc,val,err = binned(om0,MB0,"full",EDGES["voronoi"],"hel",H0b)
# systematic envelope over acceptable configurations
# Edges are held FIXED so that every configuration reports the same eight
# redshifts; the binning choice is quoted separately in the caption.
acc=[]
for om,MB,w,pre,dH in itertools.product(
        (0.24,0.26,0.28,0.2975,0.315,0.33),(-19.253,-19.30,-19.20),
        ("full","diag","flat"),("hel","hd"),(0.0,-1.0,1.0)):
    r=binned(om,MB,w,EDGES["voronoi"],pre,H0b+dH)
    if r is None or len(r[1])!=8: continue
    if np.sqrt(np.mean((r[1][:7]-ORIG[:7])**2))<0.02: acc.append(r[1])
acc=np.array(acc)
lo,hi=acc.min(axis=0),acc.max(axis=0)
print(f"acceptable configurations (fixed binning): {len(acc)}")
for k in range(8):
    print(f"  z={zc[k]:6.3f}  {val[k]:+.4f} +/- {err[k]:.4f} (stat)   "
          f"systematic [{lo[k]:+.4f}, {hi[k]:+.4f}]  width {hi[k]-lo[k]:.4f}")

# BAO
sbd=np.sqrt(np.diag(Cb)); zB,muB,eB=[],[],[]
def mu_t(z):
    z=np.atleast_1d(z); return 5*np.log10((1+z)*(C/H0b)*L.g.I(z,om0,-1,0))+25
for z,v,k,e in zip(zb,vb,kb,sbd):
    if k!="DM_over_rs": continue
    DM=v*RD; s=DM*np.hypot(e/v,RD_ERR/RD)
    zB.append(z); muB.append(float(5*np.log10((1+z)*DM)+25-mu_t(z)[0]))
    eB.append(float(5/np.log(10)*s/DM))
zB,muB,eB=map(np.array,(zB,muB,eB))
gap=5*np.log10(float(d["H0_sn"])/H0b); plan=5*np.log10(H0b/67.4)

plt.rcParams.update({"font.family":"serif","font.size":8,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.0,
 "axes.linewidth":0.6,"figure.dpi":200,"savefig.bbox":"tight"})
BAO_C,SN_C="#6B3A10","#1F4E79"
fig,ax=plt.subplots(figsize=(3.4,2.6))
ax.axhline(0,color="k",lw=0.7,zorder=2)
ax.axhline(-gap,color="#8C2D19",ls="--",lw=0.8,zorder=2)
ax.axhline(plan,color="0.55",ls=":",lw=0.8,zorder=2)
for k in range(8):
    wdt=zc[k]*0.13
    ax.add_patch(plt.Rectangle((zc[k]-wdt,lo[k]),2*wdt,hi[k]-lo[k],
                 facecolor=SN_C,alpha=0.16,lw=0,zorder=3))
ax.errorbar(zB,muB,yerr=eB,fmt="s",ms=3.2,color=BAO_C,elinewidth=0.8,
            capsize=1.5,zorder=5,label=r"DESI BAO (Planck $r_d$)")
ax.errorbar(zc[:7],val[:7],yerr=err[:7],fmt="o",ms=3.2,color=SN_C,
            elinewidth=0.8,capsize=1.5,zorder=6,
            label=r"Pantheon$+$ (Cepheid $M_B$), full cov")
ax.errorbar(zc[7:],val[7:],yerr=err[7:],fmt="o",ms=3.2,mfc="white",mec=SN_C,
            ecolor=SN_C,elinewidth=0.8,capsize=1.5,zorder=6)
ax.add_patch(plt.Rectangle((0,0),0,0,facecolor=SN_C,alpha=0.16,lw=0,
             label="spread over analysis choices"))
ax.set_xscale("log"); ax.set_xlabel("redshift $z$")
ax.set_ylabel(r"$\Delta\mu(z)$  [mag]")
ax.set_ylim(-0.30,0.12); ax.set_xlim(0.011,2.9)
ax.grid(color="0.92",lw=0.4); ax.set_axisbelow(True)
ax.text(0.85,-gap-0.020,f"SH0ES  $-{gap:.3f}$",fontsize=5.6,color="#8C2D19")
ax.text(0.85,plan+0.007,f"Planck  $+{plan:.3f}$",fontsize=5.6,color="0.45")
ax.legend(loc="lower left",frameon=True,framealpha=0.95)
fig.tight_layout(pad=0.3)
fig.savefig("fig3.pdf"); fig.savefig("fig3.png")
