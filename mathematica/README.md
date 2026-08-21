# Mathematica pipeline

The reference implementation. It produces the χ² ladder of Tables II–V and
VII–VIII, the chains in `../chain/`, and Figs. 4–5.

| file | role |
|---|---|
| `mcmc_*.nb` | runs the chains |
| `results_*.nb` | reads a chain and produces the parameter constraints and the corner plots |

To use a chain, copy it into the same directory as the notebook.

A PDF or `.m` export of each notebook is provided alongside the `.nb`, so that
the code can be read without a Mathematica licence.
