# CosmoWithDistDualTilt
Analysis code and data for

> **A few-percent distance-duality tilt can absorb the DESI evolving-dark-energy signal**
> S. Fay, M. Artola and L. Perivolaropoulos

The paper compares DESI DR2 BAO distances with four Type Ia supernova
compilations, allowing a distance-duality factor η(z) = α (1+z)^ε, and asks how
much of the reported preference for evolving dark energy that single tilt
absorbs.

## Layout

```
mathematica/     reference pipeline: chains, chi^2 ladder, Figs. 4-5
figures/         Python pipeline: Figs. 1-3, Tables II-V
chain/           MCMC chains
data/            data used by the Mathematica notebooks
```

The Python code in `figures/` downloads its own copy of the public data with
`figures/download_data.sh`; nothing is vendored.

## What produces what

| paper item | produced by |
|---|---|
| Fig. 1 — (w₀, wₐ) posteriors, Pantheon+ | `figures/mc.py` → `flatten.py` → `fig1.py` |
| Fig. 2 — ε across compilations, preference before/after | `figures/results.py` |
| Fig. 3 — absolutely-calibrated distance moduli | `figures/tomography.py` → `fig3_tomography.py` |
| Figs. 4–5 — corner plots, w₀wₐCDM+ε | `mathematica/` (cross-check: `figures/fig45_corner.py`) |
| Tables II–V | `figures/results.py` (writes `tables_generated.tex`) |
| Tables VII–VIII | `mathematica/` |
| χ² ladder, parameter constraints | `mathematica/`, reproduced by `figures/` |

## Quick start (Python side)

```bash
cd figures
pip install -r requirements.txt
./download_data.sh          # ~40 MB

python results.py           # -> fig2.pdf, results.json, tables_generated.tex
python tomography.py        # -> tomo.npz
python fig3_tomography.py   # -> fig3.pdf              (~8 min)

python mc.py noeps 20000    # -> bk_noeps.h5
python mc.py eps   20000    # -> bk_eps.h5
python flatten.py           # -> flat_noeps.npy, flat_eps.npy
python fig1.py              # -> fig1.pdf
```

See `figures/README.md` for what each script does, the sampling settings, the
redshift conventions, and the values each script should print.

## Data

All inputs are public: DESI DR2 BAO (13-point mean and covariance),
Pantheon+ (distances and STAT+SYS covariance), DES-Dovekie (distances and
inverse covariance), Union3 and Union3.1 (binned distance matrices), and the
Moresco cosmic-chronometer compilation with its covariance recipe.
`figures/download_data.sh` lists the canonical URLs.

## Licence

MIT, see `LICENSE`.
