#!/usr/bin/env python3
"""
Single source of truth for Fig. 2 and Tables II-V of

    "A few-percent distance-duality tilt can absorb the DESI
     evolving-dark-energy signal", Fay, Artola & Perivolaropoulos.

Every number in the figure and the tables is derived here from the raw
chi^2 ladder, so the two cannot drift apart.

PROVENANCE
  chi2 ladder, posterior masses, eps values and uncertainties, and the
  Omega_k column all come from the Mathematica pipeline (Sec. A of the
  paper); the chi2 entries are reproduced by the Python pipelines to the
  precision quoted there.  Everything below this block is computed from
  them.
"""
import json
import numpy as np
from scipy.stats import norm, chi2

# ----------------------------------------------------------------------
# RAW INPUTS
# ----------------------------------------------------------------------
C_KMS = 299792.458

SAMPLES = ["Pantheon+", "DES-Dovekie", "Union3 (1.5)", "Union3.1 (1.8)"]

# N_SN and z_max (manuscript Table I)
NSN   = {"Pantheon+": 1580, "DES-Dovekie": 1820,
         "Union3 (1.5)": 22, "Union3.1 (1.8)": 22}
ZMAX  = {"Pantheon+": 2.26, "DES-Dovekie": 1.14,
         "Union3 (1.5)": 2.26, "Union3.1 (1.8)": 2.26}
N_BAO = 13

# chi2 minima: (LCDM, w0waCDM, LCDM+eps, w0waCDM+eps)
CHI2 = {
    "Pantheon+"      : (1400.33,  1395.59,  1397.44,  1395.45),
    "DES-Dovekie"    : (1645.30,  1638.84,  1641.84,  1638.21),
    "Union3 (1.5)"   : (  38.8139,  28.7833,  34.1837,  28.4318),
    "Union3.1 (1.8)" : (  41.0926,  34.6766,  38.9251,  33.6682),
}

# best-fit eps in LCDM+eps (Mathematica pipeline); posterior medians agree
EPS = {
    "Pantheon+"      : -0.0283482,
    "DES-Dovekie"    : -0.0275075,
    "Union3 (1.5)"   : -0.0482738,
    "Union3.1 (1.8)" : -0.0314062,
}
# eps profile-likelihood uncertainties (Table III).
EPS_ERR = {
    "Pantheon+": 0.017, "DES-Dovekie": 0.015,
    "Union3 (1.5)": 0.022, "Union3.1 (1.8)": 0.021,
}
# Delta chi2_DE with Omega_k free (Table II).
DCHI2_OMK = {
    "Pantheon+": 3.01, "DES-Dovekie": 4.52,
    "Union3 (1.5)": 8.40, "Union3.1 (1.8)": 5.17,
}
# posterior mass enclosing LCDM in the (w0,wa) plane
POST_MASS = {
    "Pantheon+"      : (0.904418, 0.631039),
    "DES-Dovekie"    : (0.970925, 0.916864),
    "Union3 (1.5)"   : (0.989863, 0.963688),
    "Union3.1 (1.8)" : (0.911301, 0.870943),
}

# inverse ladder; Om is a plug-in at the BAO best fit (Sec. VII B)
B_STAR, B_STAR_ERR = 29.5246, 0.081782
H0_CC, H0_CC_ERR = 68.7, 4.0

# Absolutely-calibrated H0 (Sec. VIII, Eqs. 18-19).  Unrounded, as returned
# by tomography.py; rounding these to 69.0/0.2 and 73.2/0.9 moves the
# tension from 4.45 to 4.56 sigma, which is why the paper quotes the extra digits.
H0_BAO, H0_BAO_ERR = 69.03, 0.23
H0_SN,  H0_SN_ERR  = 73.25, 0.92

EPS_REF = -0.032          # weighted mean, used in Sec. VII


# ----------------------------------------------------------------------
# CONVERSIONS
# ----------------------------------------------------------------------
def sigma_from_dchi2(d, dof=2):
    """Paper convention: 1D-equivalent two-sided sigma from a dof-dim region."""
    return norm.isf(chi2.sf(d, dof) / 2)


def sigma_from_mass(frac):
    """Posterior-mass convention: sqrt(Quantile[ChiSquare[1], frac])."""
    return np.sqrt(chi2.ppf(frac, 1))


def mass_from_sigma(s):
    return chi2.cdf(s ** 2, 1)


# ----------------------------------------------------------------------
# DERIVED QUANTITIES
# ----------------------------------------------------------------------
def build():
    R = {}
    for s in SAMPLES:
        c_l, c_w, c_le, c_we = CHI2[s]
        d_de   = c_l - c_w
        d_deps = c_le - c_we
        N = NSN[s] + N_BAO
        R[s] = dict(
            chi2_lcdm=c_l, chi2_w0wa=c_w, chi2_lcdm_eps=c_le, chi2_w0wa_eps=c_we,
            dchi2_DE=d_de,
            dchi2_DE_eps=d_deps,
            dchi2_omk=DCHI2_OMK[s],
            absorbed=1 - d_deps / d_de,
            sig_DE=sigma_from_dchi2(d_de),
            sig_DE_eps=sigma_from_dchi2(d_deps),
            sig_omk=sigma_from_dchi2(DCHI2_OMK[s]),
            eps=EPS[s], eps_err=EPS_ERR[s],
            eps_pull=abs(EPS[s]) / EPS_ERR[s],
            N_bic=N,
            # BIC: k = 1 (LCDM), 3 (w0wa), 2 (LCDM+eps).  Positive favours LCDM.
            dbic_w0wa_lcdm=(c_w - c_l) + 2 * np.log(N),
            dbic_w0wa_lcdmeps=(c_w - c_le) + 1 * np.log(N),
            # LCDM+eps improvement over LCDM, 1 dof
            dchi2_eps_gain=c_l - c_le,
            sig_eps_gain=sigma_from_dchi2(c_l - c_le, dof=1),
            post_mass=POST_MASS[s],
            sig_post=(sigma_from_mass(POST_MASS[s][0]),
                      sigma_from_mass(POST_MASS[s][1])),
            mass_implied=(mass_from_sigma(sigma_from_dchi2(d_de)),
                          mass_from_sigma(sigma_from_dchi2(d_deps))),
        )

    # weighted mean of eps -- PRIMARY = the three public compilations
    PUB = SAMPLES
    w = np.array([1 / R[s]["eps_err"] ** 2 for s in PUB])
    x = np.array([R[s]["eps"] for s in PUB])
    mean = float(np.sum(w * x) / np.sum(w))
    err = float(1 / np.sqrt(np.sum(w)))
    scat = float(np.sum(w * (x - mean) ** 2))
    meta = dict(
        eps_mean=mean, eps_mean_err=err,
        scatter_chi2=scat, scatter_dof=len(PUB) - 1,
        scatter_p=float(chi2.sf(scat, len(PUB) - 1)),
        eps_mean_pull=abs(mean) / err,
        # conservative alternative: floor at best single measurement
        eps_err_floor=float(min(EPS_ERR.values())),
    )

    # inverse ladder
    H0rd = C_KMS / B_STAR
    H0rd_err = H0rd * B_STAR_ERR / B_STAR
    rd = H0rd / H0_CC
    rd_err = rd * np.hypot(B_STAR_ERR / B_STAR, H0_CC_ERR / H0_CC)
    meta.update(H0rd=H0rd, H0rd_err=H0rd_err, rd=rd, rd_err=rd_err,
                rd_shoes=H0rd / 73.0)

    # Method II
    dH0 = H0_SN - H0_BAO
    sH0 = np.hypot(H0_BAO_ERR, H0_SN_ERR)
    dmag = 5 * np.log10(H0_SN / H0_BAO)
    sdmag = (5 / np.log(10)) * np.hypot(H0_SN_ERR / H0_SN, H0_BAO_ERR / H0_BAO)
    meta.update(H0_diff_sigma=dH0 / sH0, dmag=dmag, dmag_err=sdmag,
                dmag_sigma=dmag / sdmag)

    # varying-G numbers (Sec. VII, Table IV)
    yr_per_kmsMpc = 1.0 / (9.77792e11)          # (km/s/Mpc) -> 1/yr
    H0_yr = H0_CC * yr_per_kmsMpc
    gdot = (4 * abs(EPS_REF) / 3) * H0_yr
    meta.update(
        Gdot_over_G=gdot,
        dG_z1=1 - (2.0) ** (4 * EPS_REF / 3),
        dG_rec=1 - (1101.0) ** (4 * EPS_REF / 3),
        tilt_z1_mag=5 * EPS_REF * np.log10(2.0),
        llr_hofmann_ratio=gdot / 7.6e-14,
        llr_hofmann_ratio_1sig_edge=gdot / (7.1e-14 + 7.6e-14),
        llr_biskupek_ratio=gdot / 1e-14,
    )
    return R, meta


# ----------------------------------------------------------------------
# LATEX TABLES
# ----------------------------------------------------------------------
def latex_tables(R, M):
    def f(x, n=2):
        return f"{x:.{n}f}"

    out = []

    # ---- Table I : data ------------------------------------------------
    out.append(r"""\begin{table}[t]
\caption{Datasets used in this work. The same DESI DR2 BAO vector is
combined with each supernova compilation.}
\label{tab:data}
\begin{ruledtabular}
\begin{tabular}{lccl}
dataset & $N$ & $z_{\max}$ & covariance \\
\hline
DESI DR2 BAO       & 13   & 2.33 & full $13\times13$ \\
Pantheon$+$        & 1580 & 2.26 & $\text{STAT}+\text{SYS}$ \\
DES-Dovekie        & 1820 & 1.14 & full (pre-inverted) \\
Union3 (UNITY1.5)  & 22   & 2.26 & full (binned) \\
Union3.1 (UNITY1.8)& 22   & 2.26 & full (binned) \\
Cosmic chronometers& 15   & 1.97 & full (BC03) \\
\end{tabular}
\end{ruledtabular}
\end{table}""")

    # ---- Table II : horse race ----------------------------------------
    rows = []
    for s in SAMPLES:
        r = R[s]
        nm = s.replace("Pantheon+", r"Pantheon$+$")
        rows.append(
            f"{nm:<15s}& ${f(r['dchi2_DE'])}\\,({f(r['sig_DE'],1)}\\sigma)$ "
            f"& ${f(r['dchi2_DE_eps'])}\\,({f(r['sig_DE_eps'],1)}\\sigma)$ "
            f"& ${f(r['dchi2_omk'])}\\,({f(r['sig_omk'],1)}\\sigma)$ \\\\")
    out.append(r"""\begin{table}[t]
\caption{Dark-energy preference $\Delta\chi^2_{\rm DE}$ ($2$ dof; equivalent
$\sigma$ in parentheses, see Eq.~(\ref{eq:sigconv})) with no extra parameter
and with a third parameter $X\in\{\eps,\Omega_k\}$ added to both models.}
\label{tab:horse}
\begin{ruledtabular}
\begin{tabular}{lccc}
sample & no $X$ & $X=\eps$ & $X=\Omega_k$ \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}""")

    # ---- Table III : stability ----------------------------------------
    rows = []
    for s in SAMPLES:
        r = R[s]
        nm = s.replace("Pantheon+", r"Pantheon$+$")
        rows.append(
            f"{nm:<15s}& ${ZMAX[s]:.2f}$ "
            f"& ${r['eps']:.3f}\\pm{r['eps_err']:.3f}$ "
            f"& ${f(r['sig_DE'],1)}\\sigma$ & ${f(r['sig_DE_eps'],1)}\\sigma$ "
            f"& ${100*r['absorbed']:.0f}\\%$ \\\\")
    out.append(r"""\begin{table}[t]
\caption{$\LCDM+\eps$ fits (DESI DR2 BAO $+$ each compilation, full
covariances, no CMB). ``absorbed'' is
$1-\Delta\chi^2_{\rm DE|\eps}/\Delta\chi^2_{\rm DE}$.}
\label{tab:stability}
\begin{ruledtabular}
\begin{tabular}{lccccc}
sample & $z_{\max}$ & $\eps$ & DE & DE$|\eps$ & absorbed \\
\hline
""" + "\n".join(rows) + f"""
\\hline
weighted mean  &      & ${M['eps_mean']:.3f}\\pm{M['eps_mean_err']:.3f}$ & & & \\\\
\\end{{tabular}}
\\end{{ruledtabular}}
\\end{{table}}""")

    # ---- Table IV : varying-G bounds -----------------------------------
    g = M["Gdot_over_G"]
    out.append(rf"""\begin{{table}}[t]
\caption{{The varying-$G$ interpretation of $\eps={EPS_REF}$ against direct
bounds; $|\dot G/G|_0\simeq{g*1e12:.1f}\times10^{{-12}}$~yr$^{{-1}}$ is
sign-independent.}}
\label{{tab:Gbounds}}
\begin{{ruledtabular}}
\begin{{tabular}}{{lcc}}
probe & bound (yr$^{{-1}}$) & status \\
\hline
LLR~\cite{{HofmannMuller2018}} & $(7.1\pm7.6)\times10^{{-14}}$ & exceeds $\times\!\sim\!{M['llr_hofmann_ratio_1sig_edge']:.0f}$\footnotemark[1] \\
LLR~\cite{{Biskupek2021}}      & $\lesssim10^{{-14}}$          & exceeds $\times\!\sim\!{M['llr_biskupek_ratio']:.0f}$ \\
SNIa direct~\cite{{Zhao2018}}  & $\lesssim\text{{few}}\times10^{{-11}}$ & consistent \\
BBN (extrap.)                & $|\Delta G/G|\lesssim0.1$   & excluded (${100*M['dG_rec']:.0f}\%$) \\
$\rd$ (this work)            & standard                   & late-time only \\
\end{{tabular}}
\footnotetext[1]{{Ratio taken against the $1\sigma$ upper edge of the quoted
interval, $1.47\times10^{{-13}}$; against the central value the factor is
$\sim\!42$.}}
\end{{ruledtabular}}
\end{{table}}""")
    return "\n\n".join(out)


# ----------------------------------------------------------------------
# FIGURE 3
# ----------------------------------------------------------------------
def figure3(R, M, path="fig3_eps_stability.pdf", show_posterior=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": 200, "savefig.bbox": "tight",
    })
    DARK, ACC, BAND = "#8C2D19", "#1F4E79", "#C7D9EA"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.3))
    y = np.arange(len(SAMPLES))[::-1]
    labels = [s.replace("Pantheon+", "Pantheon+") for s in SAMPLES]

    # ---------------- left: eps ----------------
    m, me = M["eps_mean"], M["eps_mean_err"]
    axL.axvspan(m - me, m + me, color=BAND, zorder=0)
    axL.axvline(m, color=ACC, lw=0.9, zorder=1)
    axL.axvline(0.0, color="0.25", ls="--", lw=0.8, zorder=1)

    for yy, s in zip(y, SAMPLES):
        r = R[s]
        axL.errorbar(r["eps"], yy, xerr=2 * r["eps_err"], fmt="none",
                     ecolor=DARK, elinewidth=0.7, capsize=2.0, zorder=2)
        axL.errorbar(r["eps"], yy, xerr=r["eps_err"], fmt="o", ms=3.4,
                     color=DARK, ecolor=DARK, elinewidth=1.9, capsize=0,
                     mfc=DARK, zorder=3)

    axL.set_yticks(y); axL.set_yticklabels(labels)
    axL.tick_params(axis="y", length=0)
    axL.set_ylim(-1.75, len(SAMPLES) - 0.35)
    axL.set_xlabel(r"$\varepsilon$   ($\Lambda$CDM$+\varepsilon$ fit)")
    axL.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    axL.set_axisbelow(True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axL.legend(handles=[
        Patch(facecolor=BAND, edgecolor=ACC,
              label=rf"weighted mean ${m:.3f}\pm{me:.3f}$"),
        Line2D([], [], color=DARK, lw=1.9,
               label=r"inner bar $1\sigma$, outer $2\sigma$"),
        Line2D([], [], color="0.25", ls="--", lw=0.8,
               label=r"$\varepsilon=0$ (no DDR violation)"),
    ], loc="lower left", frameon=True, framealpha=0.95)

    # ---------------- right: DE preference ----------------
    h = 0.34
    for yy, s in zip(y, SAMPLES):
        r = R[s]
        for off, val, col in ((h / 2, r["sig_DE"], ACC),
                              (-h / 2, r["sig_DE_eps"], DARK)):
            axR.barh(yy + off, val, height=h, color=col, zorder=2)
            axR.text(val + 0.04, yy + off, f"{val:.2f}", va="center",
                     ha="left", fontsize=6.5)
        if show_posterior:
            for off, val in ((h / 2, r["sig_post"][0]),
                             (-h / 2, r["sig_post"][1])):
                axR.plot([val], [yy + off], marker="|", ms=9, mew=1.3,
                         color="k", zorder=4)

    axR.set_yticks(y); axR.set_yticklabels([])
    axR.set_ylim(-1.75, len(SAMPLES) - 0.35)
    axR.set_xlim(0, 3.15)
    axR.set_xlabel(r"preference for evolving dark energy  [$\sigma$]")
    axR.grid(axis="x", color="0.9", lw=0.5, zorder=0)
    axR.set_axisbelow(True)
    handles = [Patch(facecolor=ACC, label=r"standard"),
               Patch(facecolor=DARK, label=r"with $\varepsilon$ free")]
    if show_posterior:
        handles.append(Line2D([], [], color="k", marker="|", ls="none",
                              ms=9, mew=1.3, label="posterior mass"))
    axR.legend(handles=handles, loc="lower right", frameon=True,
               framealpha=0.95)

    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
if __name__ == "__main__":
    R, M = build()
    with open("results.json", "w") as fh:
        json.dump({"samples": R, "meta": M}, fh, indent=2, default=float)
    with open("tables_generated.tex", "w") as fh:
        fh.write(latex_tables(R, M) + "\n")
    figure3(R, M, "fig2.pdf")
    
    print("wrote results.json, tables_generated.tex, fig2.pdf(+png)")
