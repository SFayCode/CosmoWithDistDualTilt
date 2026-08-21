#!/usr/bin/env python3
"""
Corner plots of the w0waCDM+eps posterior (Figs. 4 and 5).

The published Figs. 4-5 were produced by the Mathematica pipeline; this script
draws the same panels from the same chains, as a cross-check and as a route for
readers without a Mathematica licence.

    python fig45_corner.py ../chain/<chainfile> -o fig4_pantheon.pdf \
           --title "Pantheon+"

Accepted chain formats: .npy, .npz (first array), or whitespace/comma-separated
text.  Columns are assumed to be (Om, w0, wa, eps) in that order; use --cols to
select and reorder if the file carries extra columns, e.g. --cols 1 2 3 4 for a
file whose first column is the step index or the log-likelihood.

Contours are the 68%, 95% and 99% credible regions and the marker is the
posterior median, matching the caption of Fig. 4.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

LABELS = [r"$\Omega_{0\rm m}$", r"$w_0$", r"$w_a$", r"$\varepsilon$"]
LEVELS = (0.68, 0.95, 0.99)
FILL = ["#7FD4D4", "#4FA3C7", "#3C6FA8"]  # 68 innermost -> 99 outermost


def load_chain(path, cols, burn):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        arr = z[list(z.keys())[0]]
    else:
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    arr = np.atleast_2d(arr)
    if cols:
        arr = arr[:, list(cols)]
    if arr.shape[1] != 4:
        raise SystemExit(
            f"{path}: got {arr.shape[1]} columns, expected 4 "
            f"(Om, w0, wa, eps). Use --cols to select them.")
    if burn > 0:
        arr = arr[int(burn * len(arr)):]
    return arr


def kde_levels(x, y, grid=140):
    gx = np.linspace(x.min(), x.max(), grid)
    gy = np.linspace(y.min(), y.max(), grid)
    X, Y = np.meshgrid(gx, gy)
    Z = gaussian_kde(np.vstack([x, y]))(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)
    s = np.sort(Z.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    lv = [s[np.searchsorted(c, f)] for f in LEVELS]
    return X, Y, Z, sorted(lv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chain")
    ap.add_argument("-o", "--out", default="corner.pdf")
    ap.add_argument("--title", default="")
    ap.add_argument("--cols", type=int, nargs=4, default=None)
    ap.add_argument("--burn", type=float, default=0.0,
                    help="fraction of the chain to discard (default 0)")
    a = ap.parse_args()

    ch = load_chain(a.chain, a.cols, a.burn)
    med = np.median(ch, axis=0)
    n = ch.shape[1]

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 6,
                         "ytick.labelsize": 6, "axes.linewidth": 0.6,
                         "figure.dpi": 200, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(n, n, figsize=(5.2, 5.2))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                k = gaussian_kde(ch[:, i])
                g = np.linspace(ch[:, i].min(), ch[:, i].max(), 220)
                ax.plot(g, k(g), color="k", lw=0.8)
                ax.set_yticks([])
                ax.set_xlim(g[0], g[-1])
            else:
                X, Y, Z, lv = kde_levels(ch[:, j], ch[:, i])
                ax.contourf(X, Y, Z, levels=lv + [Z.max()], colors=FILL[::-1])
                ax.contour(X, Y, Z, levels=lv, colors="k", linewidths=0.4)
                ax.plot(med[j], med[i], marker="o", ms=2.2, color="k")
            if i == n - 1:
                ax.set_xlabel(LABELS[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS[i])
            else:
                ax.set_yticklabels([])

    if a.title:
        fig.suptitle(a.title, fontsize=9, y=0.94)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png")
    print(f"{a.chain}: {len(ch)} samples -> {a.out}")
    for lab, m, lo, hi in zip(("Om", "w0", "wa", "eps"), med,
                              np.percentile(ch, 16, axis=0),
                              np.percentile(ch, 84, axis=0)):
        print(f"  {lab:>4} = {m:+.4f}  (+{hi-m:.4f} -{m-lo:.4f})")


if __name__ == "__main__":
    main()
