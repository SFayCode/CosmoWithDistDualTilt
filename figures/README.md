# Figure scripts

Code that reproduces Figs. 1–3 of *A few-percent distance-duality tilt can
absorb the DESI evolving-dark-energy signal* (Fay, Artola & Perivolaropoulos).

## Quick start

```bash
pip install -r requirements.txt
./download_data.sh          # ~40 MB from the public releases

python results.py           # -> fig2.pdf, results.json, tables_generated.tex (seconds)
python tomography.py        # -> tomo.npz  (prerequisite for Fig. 3)
python fig3_tomography.py   # -> fig3.pdf  (~8 min)

python mc.py noeps 20000    # -> bk_noeps.h5
python mc.py eps   20000    # -> bk_eps.h5
python flatten.py           # -> flat_noeps.npy, flat_eps.npy
python fig1.py              # -> fig1.pdf
```

`mc.py` takes the run tag (`noeps` or `eps`) and the number of steps, and
appends to an existing backend if one is present, so a run can be extended by
calling it again with the same tag.

## What each script does

| file | role |
|---|---|
| `pipeline.py` | shared module: data loaders, background expansion, analytic-marginalized BAO/SnIa likelihoods. Not run directly. |
| `mc.py` | emcee chains for `w0waCDM` with and without ε (DESI DR2 BAO + Pantheon+, full covariance) |
| `flatten.py` | burn-in, thinning and flattening of the backends; prints the chain diagnostics |
| `fig1.py` | **Fig. 1** — joint (w₀, wₐ) posteriors, 68% / 95%, with and without ε |
| `results.py` | **Fig. 2** — ε across the four compilations and the dark-energy preference before/after. Also writes `results.json` and the LaTeX for Tables II–V. Needs no data: it is driven by the χ² ladder tabulated at the top of the file. |
| `tomography.py` | absolutely-calibrated H₀ from BAO (Planck r_d) and SnIa (SH0ES M_B); writes `tomo.npz` |
| `fig3_tomography.py` | **Fig. 3** — distance moduli relative to the BAO ΛCDM template, with the analysis-choice bands. Scans 58 configurations at fixed bin edges. |

`fig45_corner.py` redraws the corner plots of **Figs. 4–5** from a chain, as a
cross-check on the Mathematica versions and as a route for readers without a
Mathematica licence:

```bash
python fig45_corner.py ../chain/<chainfile> -o fig4_pantheon.pdf --title "Pantheon+"
```

It assumes columns (Ω_0m, w₀, wₐ, ε); use `--cols` to select and reorder if the
chain carries a step index or a log-likelihood column. The published Figs. 4–5
are the Mathematica ones.

## Reproduced values

Running the above gives, as printed to stdout:

- `results.py` — weighted mean ε = −0.0320 ± 0.0090, scatter χ² = 0.684 for 3 dof,
  p = 0.88; sound horizon r_d = 147.8 ± 8.6 Mpc
- `tomography.py` — H₀^BAO = 69.03 ± 0.23, H₀^SN = 73.25 ± 0.92, χ²/dof = 0.879,
  ΔH₀ = 4.45σ, magnitude gap 0.1288 ± 0.0282 mag
- `fig3_tomography.py` — 58 acceptable configurations; per-bin systematic widths
  0.037–0.060 mag for the seven lower bins and 0.150 mag for the highest

## Sampling

`mc.py` uses emcee's affine-invariant ensemble sampler with 32 walkers. The
prior on Ω_0m for the chains is [0.05, 0.60], deliberately wider than the range
over which constraints are quoted: the chains reach the lower edge, and
restricting to Ω_0m ∈ [0.1, 0.5] afterwards changes the enclosed posterior mass
by < 0.01σ (Appendix A of the paper). The remaining priors are those of
Sec. III D: w₀ ∈ [−3, 0.5], wₐ ∈ [−4, 3], ε ∈ [−0.4, 0.4].

`flatten.py` prints the integrated autocorrelation time, the burn-in and
thinning applied, and the effective sample size; `mc.py` prints the acceptance
fraction.

## Conventions

The chains and the χ² ladder evaluate both the (1+z) prefactor and the comoving
distance at the cosmological redshift `z_HD` (`use_zhel=False` in
`pipeline.Like`), for comparability across the four compilations and the three
independent implementations. Passing `use_zhel=True` switches to the
two-redshift form; Sec. III A of the paper quantifies the difference.

`tomography.py` and `fig3_tomography.py` use `use_zhel=True`, since the
absolutely-calibrated comparison of Sec. VIII is made against the released
Pantheon+ convention; the redshift convention is one of the analysis choices
varied inside the shaded bands of Fig. 3.

Pantheon+ selection: `zHD > 0.01` and `IS_CALIBRATOR == 0`, giving N = 1580
and z_max = 2.2614.

## Notes

`results.py` is the single source of truth for Fig. 2 and for Tables II–V: the
figure and the tables are emitted from one dictionary, so they cannot drift
apart. If a χ² changes, edit `CHI2` at the top and rerun.

The DES covariance file stores the packed upper triangle of the *inverse*
covariance; the Union files are FITS arrays whose first row is z, first column
is μ, and remainder the inverse covariance. `pipeline.py` handles both.
