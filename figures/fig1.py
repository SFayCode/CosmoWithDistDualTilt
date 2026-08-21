import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
a=np.load("flat_noeps.npy")[:,1:3]; b=np.load("flat_eps.npy")[:,1:3]
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,
 "axes.linewidth":0.6,"figure.dpi":200,"savefig.bbox":"tight"})
fig,ax=plt.subplots(figsize=(3.4,3.0))
gx=np.linspace(-1.35,-0.45,260); gy=np.linspace(-2.6,1.5,260)
X,Y=np.meshgrid(gx,gy)
for ch,col,lab in ((a,"#1F4E79",r"$w_0w_a$CDM (no $\varepsilon$)"),
                   (b,"#8C2D19",r"$w_0w_a$CDM $+\,\varepsilon$")):
    k=gaussian_kde(ch.T); Z=k(np.vstack([X.ravel(),Y.ravel()])).reshape(X.shape)
    s=np.sort(Z.ravel())[::-1]; c=np.cumsum(s)/s.sum()
    l68,l95=s[np.searchsorted(c,0.68)],s[np.searchsorted(c,0.95)]
    ax.contourf(X,Y,Z,levels=[l95,l68,Z.max()],colors=[col,col],alpha=0.22)
    ax.contour(X,Y,Z,levels=[l95,l68],colors=col,linewidths=0.8)
    ax.plot([],[],color=col,lw=1.2,label=lab)
ax.plot(-1,0,marker="*",ms=9,color="k",zorder=5)
ax.axhline(0,color="0.55",ls=":",lw=0.6); ax.axvline(-1,color="0.55",ls=":",lw=0.6)
ax.set_xlabel(r"$w_0$"); ax.set_ylabel(r"$w_a$")
ax.set_xlim(-1.3,-0.5); ax.set_ylim(-2.4,1.4)
ax.text(-1.28,-1.75,"$\\chi^2_{\\rm min}$:\n"
        "$\\Lambda$CDM 1400.33\n$\\Lambda$CDM$+\\varepsilon$ 1397.44\n"
        "$w_0w_a$ 1395.59\n$w_0w_a{+}\\varepsilon$ 1395.45",fontsize=5.8,
        va="bottom",bbox=dict(fc="white",ec="0.7",lw=0.4,pad=2))
ax.legend(loc="upper right",frameon=True,framealpha=0.95)
fig.tight_layout(pad=0.3); fig.savefig("fig1.pdf"); fig.savefig("fig1.png")
print("ok")
