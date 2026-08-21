"""
Independent reimplementation of the paper's likelihood from public data.
Third code, to sit alongside Fay (Mathematica) and Artola (Python/Nautilus).

Data:
  DESI DR2 BAO   13 pts + 13x13 cov   (CobayaSampler/bao_data)
  Pantheon+      1701 SNe + STAT+SYS  (PantheonPlusSH0ES/DataRelease)
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

C_KMS = 299792.458

# ---------------------------------------------------------------- BAO
def load_bao(mean="data/desi_dr2_mean.txt", cov="data/desi_dr2_cov.txt"):
    z, val, kind = [], [], []
    for line in open(mean):
        if line.startswith("#") or not line.strip():
            continue
        a, b, c = line.split()
        z.append(float(a)); val.append(float(b)); kind.append(c)
    return np.array(z), np.array(val), np.array(kind), np.loadtxt(cov)


# ---------------------------------------------------------------- SNe
def load_sn(dat="data/PantheonPlusSH0ES.dat",
            cov="data/PantheonPlusSH0ES_STATSYS.cov", zmin=0.01):
    import pandas as pd
    df = pd.read_csv(dat, sep=r"\s+")
    raw = np.loadtxt(cov, skiprows=1)
    n = int(open(cov).readline())
    C = raw.reshape(n, n)
    keep = (df["zHD"].values > zmin) & (df["IS_CALIBRATOR"].values == 0)
    idx = np.where(keep)[0]
    return (df["zHD"].values[idx], df["zHEL"].values[idx],
            df["m_b_corr"].values[idx], C[np.ix_(idx, idx)], int(keep.sum()))


# ------------------------------------------------------- background
def E_of_z(z, om, w0, wa):
    ode = 1.0 - om
    f = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(om * (1 + z) ** 3 + ode * f)


class Grid:
    """One quadrature of 1/E on a fixed grid; interpolate I(z) from it."""
    def __init__(self, zmax=2.6, n=4000):
        self.zg = np.linspace(0.0, zmax, n)

    def build(self, om, w0, wa):
        inv = 1.0 / E_of_z(self.zg, om, w0, wa)
        return np.concatenate([[0.0], cumulative_trapezoid(inv, self.zg)])

    def I(self, z, om, w0, wa, Ig=None):
        if Ig is None:
            Ig = self.build(om, w0, wa)
        return np.interp(z, self.zg, Ig)


# ------------------------------------------------------ likelihoods
class Like:
    def __init__(self, zhd, zhel, mb, Csn, zb, vb, kb, Cbao, use_zhel=True):
        self.zhd, self.zhel, self.mb = zhd, zhel, mb
        self.use_zhel = use_zhel
        self.csn = cho_factor(Csn, lower=True)
        self.Lsn = np.linalg.cholesky(Csn)
        self.zb, self.vb, self.kb = zb, vb, kb
        self.cbao = cho_factor(Cbao, lower=True)
        self.g = Grid()
        self.one = np.ones_like(mb)
        self.Ci1 = cho_solve(self.csn, self.one)
        self.oCo = float(self.one @ self.Ci1)
        # whitened ones: u1 = L^-1 1  -> 1^T C^-1 r = u1 . (L^-1 r)
        from scipy.linalg import solve_triangular as _st
        self._st = _st
        self.u1 = _st(self.Lsn, self.one, lower=True)
        self.Cid_bao = cho_solve(self.cbao, self.vb)

    def _solve_sn(self, x):
        return cho_solve(self.csn, x)

    def _solve_bao(self, x):
        return cho_solve(self.cbao, x)

    def chi2_sn(self, om, w0, wa, eps, Ig=None):
        I = self.g.I(self.zhd, om, w0, wa, Ig)
        pre = self.zhel if self.use_zhel else self.zhd
        model = 5 * np.log10((1 + pre) * I) + 5 * eps * np.log10(1 + self.zhd)
        r = self.mb - model
        u = self._st(self.Lsn, r, lower=True, check_finite=False)
        return float(u @ u - (self.u1 @ u) ** 2 / self.oCo)

    def bao_template(self, om, w0, wa, Ig=None):
        """prediction vector with b = c/(H0 rd) factored out"""
        if Ig is None:
            Ig = self.g.build(om, w0, wa)
        Iz = self.g.I(self.zb, om, w0, wa, Ig)
        Ez = E_of_z(self.zb, om, w0, wa)
        t = np.empty_like(self.vb)
        for i, (z, k) in enumerate(zip(self.zb, self.kb)):
            I = float(Iz[i]); E = float(Ez[i])
            if k == "DM_over_rs":
                t[i] = I
            elif k == "DH_over_rs":
                t[i] = 1.0 / E
            elif k == "DV_over_rs":
                t[i] = (z * I * I / E) ** (1.0 / 3.0)
            else:
                raise ValueError(k)
        return t

    def chi2_bao(self, om, w0, wa, return_b=False, Ig=None):
        t = self.bao_template(om, w0, wa, Ig)
        Cit = self._solve_bao(t)
        b = (t @ self.Cid_bao) / (t @ Cit)
        r = self.vb - b * t
        chi2 = r @ self._solve_bao(r)
        if return_b:
            sb = 1.0 / np.sqrt(t @ Cit)
            return chi2, b, sb
        return chi2

    def chi2(self, om, w0, wa, eps):
        Ig = self.g.build(om, w0, wa)
        return self.chi2_sn(om, w0, wa, eps, Ig) + self.chi2_bao(om, w0, wa, Ig=Ig)


# ------------------------------------------------------------ fits
def fit(like, model):
    """model in {'lcdm','w0wa','lcdm_eps','w0wa_eps'}"""
    if model == "lcdm":
        f = lambda p: like.chi2(p[0], -1, 0, 0);            x0 = [0.30]
    elif model == "lcdm_eps":
        f = lambda p: like.chi2(p[0], -1, 0, p[1]);         x0 = [0.30, 0.0]
    elif model == "w0wa":
        f = lambda p: like.chi2(p[0], p[1], p[2], 0);       x0 = [0.30, -0.9, -0.2]
    elif model == "w0wa_eps":
        f = lambda p: like.chi2(p[0], p[1], p[2], p[3]);    x0 = [0.30, -0.9, -0.2, 0.0]
    best = None
    for seed in range(4):
        rng = np.random.default_rng(seed)
        s = np.array(x0) + (0 if seed == 0 else rng.normal(0, 0.03, len(x0)))
        r = minimize(f, s, method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-8, maxiter=40000,
                                  maxfev=40000))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x
