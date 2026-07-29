"""Plotting helpers (save-to-file only; never shows interactively).

Per project convention, figures are written to disk and the path is returned so
the caller can report it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["plot_rod"]


def plot_rod(l_grid, intensities, labels, out_path, *, title="Thickness fringes",
             logy=True):
    """Plot one or more rod-intensity curves vs the continuous ``l`` index.

    Parameters
    ----------
    l_grid : 1-D array/tensor
        The continuous ``l`` scan coordinate.
    intensities : sequence of 1-D arrays/tensors
        One curve per entry (e.g. N3 = 3, 4, 5 monolayers).
    labels : sequence of str
        Legend labels, parallel to ``intensities``.
    out_path : str
        Where to save the PNG.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except ImportError:
            pass
        return np.asarray(x)

    l = _np(l_grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for I, lab in zip(intensities, labels):
        ax.plot(l, _np(I), label=lab, lw=1.4)
    ax.set_xlabel("continuous Miller index  l")
    ax.set_ylabel("intensity (arb.)")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
