"""Grain Orientation Envelope (Nygren 2019) helpers.

The envelope is a binary support over orientations consistent with the
measured spot centroids within tolerance — see Nygren et al. 2019. Here
we provide the envelope as both a baseline observable (binary support
volume vs. macroscopic load is constant by construction since centroid
positions are insensitive to intra-grain spread) and as an
\emph{initialization} for the K-particle ParticleODF: sampling K
particle positions uniformly from the envelope places them inside the
orientation feasibility set instead of at random points in the trust
region.

The envelope itself is computed by a per-grid-point forward pass; see
``dev/paper/figures/park22_goe_envelope.py``. This module just provides
the loading / sampling utilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_envelope(path: str | Path) -> dict:
    """Load an envelope .npz produced by ``park22_goe_envelope.py``."""
    p = Path(path)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def sample_envelope(env: dict, K: int, seed: Optional[int] = None,
                     fill: str = "random") -> torch.Tensor:
    """Sample K axis-angle particles from the envelope's binary support.

    Parameters
    ----------
    env : dict
        Loaded envelope (must contain ``aa_grid`` and ``in_envelope``).
    K : int
        Number of particles to draw.
    seed : int, optional
        RNG seed for reproducibility.
    fill : {"random", "ball"}
        What to do when the envelope is smaller than K.
        ``"random"`` (default): draw with replacement from the envelope
        cells (some particles may coincide).
        ``"ball"``: pad the K-K_env shortfall with uniform-in-trust-region
        random axis-angles, taking the trust region from
        ``env["theta_max_rad"]`` if present.

    Returns
    -------
    Tensor (K, 3)  float64 axis-angle vectors.
    """
    aa = env["aa_grid"]                               # (G, 3)
    mask = env["in_envelope"]
    aa_in = aa[mask.astype(bool)]
    n_env = len(aa_in)
    if n_env == 0:
        raise ValueError("Envelope is empty; cannot sample particles.")

    rng = np.random.default_rng(seed)

    if n_env >= K or fill == "random":
        idx = rng.choice(n_env, size=K, replace=(n_env < K))
        out = aa_in[idx]
    else:
        # mix envelope-uniform with trust-region-uniform fill
        n_pad = K - n_env
        out_env = aa_in[rng.permutation(n_env)]
        theta_max = float(env.get("theta_max_rad", 0.5 * np.pi / 180))
        dirs = rng.normal(size=(n_pad, 3))
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        mags = rng.uniform(size=(n_pad, 1)) * theta_max
        out_pad = dirs * mags
        out = np.concatenate([out_env, out_pad], axis=0)
        rng.shuffle(out)

    return torch.tensor(out, dtype=torch.float64)


def envelope_stats(env: dict) -> dict:
    """RMS / max axis-angle magnitudes inside the envelope, plus volume."""
    aa = env["aa_grid"]
    mask = env["in_envelope"].astype(bool)
    aa_in = aa[mask]
    n_in = int(mask.sum())
    n_total = int(len(mask))
    if n_in == 0:
        return {"n_in": 0, "n_total": n_total, "fraction": 0.0,
                "rms_aa_deg": 0.0, "max_aa_deg": 0.0}
    norms = np.linalg.norm(aa_in, axis=-1)
    return {
        "n_in": n_in,
        "n_total": n_total,
        "fraction": n_in / n_total,
        "rms_aa_deg": float(np.degrees(np.sqrt((aa_in ** 2).sum(axis=-1).mean()))),
        "max_aa_deg": float(np.degrees(norms.max())),
    }
