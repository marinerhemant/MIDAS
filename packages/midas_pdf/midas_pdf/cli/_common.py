"""Shared helpers for the midas-pdf CLI commands."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_two_column(path: Path, *, has_sigma: Optional[bool] = None
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a whitespace-separated ASCII file with 2 or 3 columns.

    Skips leading comment lines (``#``, ``//``, or ``@``).  Returns
    ``(x, y, sigma_or_None)`` — the third column, if present, is
    interpreted as ``sigma`` unless ``has_sigma`` is set explicitly.
    """
    path = Path(path)
    rows = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if not s: continue
            if s[0] in "#/@": continue
            parts = s.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                # header — skip
                continue
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: need at least 2 numeric columns")
    x = arr[:, 0]
    y = arr[:, 1]
    if arr.shape[1] >= 3 and has_sigma is not False:
        sigma = arr[:, 2]
    else:
        sigma = None
    return x, y, sigma


def print_json(payload: dict) -> None:
    """Print a JSON summary to stdout with 2-space indent."""
    print(json.dumps(payload, indent=2, default=_json_default))


def fallback_sigma(G, sigma_np) -> "tuple":
    """Return (sigma, is_fallback).

    If ``sigma_np`` is None (the input .gr had only two columns), fabricate
    a per-point σ = 5% of |G|.max() and set ``is_fallback=True`` so the
    caller can emit a warning banner *to stderr* and label the reported
    ``chi2_reduced`` as arbitrary.  Silently swallowing this is a common
    source of "why is my χ² off by 100x?" confusion; making it explicit
    is what a PDF specialist expects.
    """
    import torch
    if sigma_np is not None:
        return (torch.tensor(sigma_np, dtype=torch.float64), False)
    fallback = torch.full_like(torch.as_tensor(G, dtype=torch.float64),
                                0.05 * float(torch.as_tensor(G).abs().max()))
    return (fallback, True)


def maybe_warn_fallback_sigma(is_fallback: bool, gr_path) -> None:
    """Emit a stderr banner when σ was fabricated as a fixed fraction of G_max."""
    import sys
    if is_fallback:
        print(
            f"⚠️  midas-pdf: {gr_path} has no σ column — "
            f"using σ = 5% × |G|_max as a placeholder.\n"
            f"    chi2_reduced is therefore arbitrary; supply a third σ column "
            f"for a physically meaningful χ²/ndof.",
            file=sys.stderr,
        )


def _json_default(obj):
    """Fallback for numpy / torch scalars in JSON dumps."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
