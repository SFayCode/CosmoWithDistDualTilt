"""Method II: absolutely-calibrated H0 from BAO (Planck rd) and SN (SH0ES M_B)."""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from pipeline import *

C = 299792.458
RD_PLANCK, RD_ERR = 147.09, 0.26      # Planck 2018
MB_SHOES, MB_ERR  = -19.253, 0.027    # Riess+2022

zb,vb,kb,Cb = load_bao(); zhd,zhel,mb,Csn,n = load_sn()
L = Like(zhd,zhel,mb,Csn,zb,vb,kb,Cb, use_zhel=True)

# ---- BAO: fit Om, read b = c/(H0 rd) and its error -------------------
r = minimize_scalar(lambda om: L.chi2_bao(om,-1,0), bounds=(0.15,0.45),
                    method="bounded", options=dict(xatol=1e-8))
om_bao = r.x
chi2b, b, sb = L.chi2_bao(om_bao,-1,0, return_b=True)
H0_bao = C/(b*RD_PLANCK)
# error: b (statistical) + rd
sH0_bao = H0_bao*np.hypot(sb/b, RD_ERR/RD_PLANCK)
sH0_bao_stat = H0_bao*(sb/b)
print(f"BAO : Om={om_bao:.4f}  b=c/(H0 rd)={b:.4f}+/-{sb:.4f}  chi2={chi2b:.2f}/12")
print(f"      H0_BAO = {H0_bao:.2f} +/- {sH0_bao:.2f} (with rd err)"
      f" / +/-{sH0_bao_stat:.2f} (stat only)      [paper 69.0 +/- 0.2]")

# ---- SN: fit Om and the offset, convert to H0 ------------------------
one = np.ones_like(mb)
def sn_offset(om):
    I  = L.g.I(zhd, om,-1,0)
    r  = mb - 5*np.log10((1+zhel)*I)
    Cir, Ci1 = L._solve_sn(r), L._solve_sn(one)
    off = (one@Cir)/(one@Ci1)
    chi2 = r@Cir - (one@Cir)**2/(one@Ci1)
    return off, 1/np.sqrt(one@Ci1), chi2
rs = minimize_scalar(lambda om: sn_offset(om)[2], bounds=(0.15,0.45),
                     method="bounded", options=dict(xatol=1e-8))
om_sn = rs.x
off, soff, chi2s = sn_offset(om_sn)
# off = M_B + 5log10(c/H0) + 25   ->  H0
H0_sn = C*10**((MB_SHOES + 25 - off)/5)
sH0_sn = H0_sn*np.log(10)/5*np.hypot(soff, MB_ERR)
sH0_sn_stat = H0_sn*np.log(10)/5*soff
print(f"SN  : Om={om_sn:.4f}  offset={off:.4f}+/-{soff:.4f}  chi2/dof={chi2s/(n-2):.3f}")
print(f"      H0_SN  = {H0_sn:.2f} +/- {sH0_sn:.2f} (with M_B err)"
      f" / +/-{sH0_sn_stat:.2f} (stat only)      [paper 73.2 +/- 0.9]")

# ---- the tension ------------------------------------------------------
d  = H0_sn-H0_bao; s = np.hypot(sH0_bao, sH0_sn)
dm = 5*np.log10(H0_sn/H0_bao)
sdm= 5/np.log(10)*np.hypot(sH0_sn/H0_sn, sH0_bao/H0_bao)
print(f"\nH0 difference : {d:.2f} +/- {s:.2f}  -> {d/s:.2f} sigma   [paper 4.4]")
print(f"magnitude gap : {dm:.4f} +/- {sdm:.4f} mag -> {dm/sdm:.2f} sigma [paper 0.13+/-0.03, 4.6]")
np.savez("tomo.npz", om_bao=om_bao,b=b,sb=sb,H0_bao=H0_bao,sH0_bao=sH0_bao,
         om_sn=om_sn,off=off,soff=soff,H0_sn=H0_sn,sH0_sn=sH0_sn)
