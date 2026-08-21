#!/usr/bin/env python3
"""
Convert the emcee backends written by mc.py into the flat arrays fig1.py reads.

    python mc.py noeps 20000     ->  bk_noeps.h5
    python mc.py eps   20000     ->  bk_eps.h5
    python flatten.py            ->  flat_noeps.npy, flat_eps.npy

Burn-in and thinning are set from the integrated autocorrelation time when
emcee can estimate it (discard 5 tau, thin tau/2), and fall back to a fixed
25% burn-in with no thinning when the chain is too short for an estimate.
The diagnostics printed here are the ones quoted in Appendix A of the paper.
"""
import numpy as np
import emcee

COLUMNS = "om w0 wa eps"  # mc.py parameter order; fig1.py uses columns 1:3


def flatten(tag):
    bk = emcee.backends.HDFBackend(f"bk_{tag}.h5", read_only=True)
    n_iter = bk.iteration
    if n_iter == 0:
        raise SystemExit(f"bk_{tag}.h5 is empty -- run mc.py {tag} <nsteps> first")

    try:
        tau = bk.get_autocorr_time()
        tau_max = float(np.max(tau))
        discard, thin = int(5 * tau_max), max(1, int(tau_max / 2))
        note = f"tau_max={tau_max:.1f}"
    except emcee.autocorr.AutocorrError as err:
        tau_max = float(np.max(err.tau)) if hasattr(err, "tau") else float("nan")
        discard, thin = n_iter // 4, 1
        note = (f"tau not converged (estimate {tau_max:.1f}); "
                f"using 25% burn-in, no thinning")

    flat = bk.get_chain(discard=discard, thin=thin, flat=True)
    np.save(f"flat_{tag}.npy", flat)

    nwalkers = bk.shape[0]
    n_eff = n_iter * nwalkers / tau_max if np.isfinite(tau_max) and tau_max > 0 else np.nan
    print(f"{tag}: {n_iter} iterations x {nwalkers} walkers, "
          f"discard={discard}, thin={thin}, {flat.shape[0]} samples "
          f"({COLUMNS.split()[:flat.shape[1]]})")
    print(f"      {note}; effective sample size ~ {n_eff:.0f}")
    return flat


if __name__ == "__main__":
    for tag in ("noeps", "eps"):
        flatten(tag)
