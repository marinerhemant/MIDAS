"""Cromer-Mann (IT92) X-ray atomic form factors.

f(s) = Σᵢ aᵢ exp(-bᵢ s²) + c    where s = sin(θ)/λ = 1/(2d)   [Å⁻¹]

Coefficients shipped in ``data/cromer_mann.json`` (gemmi IT92 export, neutral
atoms Z = 1..98).  Backend-agnostic: works on numpy arrays or torch tensors;
when torch is given, the result is differentiable.
"""
from __future__ import annotations

import json
import re
import warnings
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "form_factor",
    "form_factor_batch",
    "available_elements",
    "coefficients",
    "register_ion",
    "registered_ions",
]


# ---------------------------------------------------------------- table loading

@lru_cache(maxsize=1)
def _table() -> dict[str, dict]:
    raw = json.loads(files("midas_hkls").joinpath("data/cromer_mann.json").read_text())
    return raw


def available_elements() -> list[str]:
    return sorted(_table().keys())


_CHARGE_RE = re.compile(r"^([A-Z][a-z]?)(?:([0-9]+)?([+-]))?$")

_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
          "VIII": 8, "IX": 9, "X": 10}

#: Runtime registry of *ionic* Cromer-Mann coefficients, keyed by the species
#: string exactly as written (e.g. ``"Fe3+"``). Empty by default: this package
#: ships the IT92 **neutral-atom** table only, and inventing ionic coefficients
#: would be worse than folding to neutral, because a wrong number looks right.
#: Populate it with :func:`register_ion` — ``midas_pdf.ionic_form_factors``
#: does exactly that at import for the species it has verified.
_ION_TABLE: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

_FOLD_WARNED: set[str] = set()


def _parse_species(symbol: str) -> tuple[str, int]:
    """``'Fe3+'`` -> ``('Fe', +3)``; ``'O2-'`` -> ``('O', -2)``; ``'Ni'`` -> ``('Ni', 0)``.

    ``Fe(III)`` is read as Fe³⁺ (the Roman numeral is an oxidation state).
    """
    s = symbol.strip()
    if not s:
        raise ValueError("empty element symbol")

    charge = 0
    rm = re.search(r"\(([IVX]+)\)", s)
    if rm is not None:
        charge = _ROMAN.get(rm.group(1), 0)
        s = re.sub(r"\([IVX]+\)", "", s)

    m = _CHARGE_RE.match(s)
    if m is None:
        lead = re.match(r"^[A-Z][a-z]?", s)
        return (lead.group(0) if lead else s), charge
    el, mag, sign = m.group(1), m.group(2), m.group(3)
    if sign is not None:
        n = int(mag) if mag else 1
        charge = n if sign == "+" else -n
    return el, charge


def _normalize_symbol(symbol: str) -> str:
    """Map 'Fe', 'Fe2+', 'O2-', 'Fe(III)' → bare element 'Fe' / 'O'."""
    return _parse_species(symbol)[0]


def register_ion(species: str, a, b, c: float, *, source: str = "") -> None:
    """Register 4-Gaussian Cromer-Mann coefficients for an ionic species.

    ``f(s) = c + Σ a_i exp(-b_i s²)``, the same form as the neutral table.

    The electron-count sum rule ``f(0) = Σa + c ≈ Z - charge`` is enforced to
    1 %: f(0) *is* the electron count, so a table entry that violates it is
    simply wrong, and this is the one check that catches a transcription slip
    without a second source.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != 4 or b.size != 4:
        raise ValueError(f"{species}: need 4 a and 4 b coefficients")
    el, charge = _parse_species(species)
    if charge == 0:
        raise ValueError(
            f"{species!r} carries no charge — register_ion is for ions; the "
            "neutral table is shipped and must not be shadowed."
        )
    tbl = _table()
    if el not in tbl:
        raise KeyError(f"{species}: unknown element {el!r}")
    z = int(tbl[el]["Z"])
    expected = z - charge
    f0 = float(a.sum() + c)
    if expected <= 0:
        raise ValueError(f"{species}: Z - charge = {expected} is not a valid electron count")
    if abs(f0 - expected) / expected > 0.01:
        raise ValueError(
            f"{species}: f(0) = {f0:.4f} but Z - charge = {expected}. "
            f"The electron-count sum rule is violated by "
            f"{100 * abs(f0 - expected) / expected:.1f} % — the coefficients "
            f"are for a different species, or mistranscribed."
        )
    _ION_TABLE[species.strip()] = (a, b, float(c))
    _FOLD_WARNED.discard(species.strip())


def registered_ions() -> list[str]:
    """Ionic species with real coefficients, as opposed to a neutral fold."""
    return sorted(_ION_TABLE)


def coefficients(element: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (a, b, c) for a species, where a and b are length-4 arrays.

    An ionic species uses registered ionic coefficients when available. When it
    does not, it falls back to the **neutral** atom and warns once — the fold
    is an approximation, and silence about it is how a plausible wrong number
    reaches a paper. At Q = 0 the error is exactly the charge: ``O2-`` has 10
    electrons and neutral O has 8, a 25 % error in f(0); it shrinks with Q as
    the valence electrons stop contributing.
    """
    key = element.strip()
    if key in _ION_TABLE:
        a, b, c = _ION_TABLE[key]
        return a.copy(), b.copy(), c

    sym, charge = _parse_species(element)
    tbl = _table()
    if sym not in tbl:
        raise KeyError(
            f"no Cromer-Mann coefficients for element {element!r} "
            f"(normalized {sym!r})"
        )
    if charge != 0 and key not in _FOLD_WARNED:
        _FOLD_WARNED.add(key)
        z = tbl[sym].get("Z", 0)
        expected = (z - charge) if z else None
        detail = (
            f" f(0) will be {z} electrons instead of {expected} "
            f"({100 * abs(charge) / max(expected, 1):.0f} % high at Q=0)"
            if expected else ""
        )
        warnings.warn(
            f"{element!r}: no ionic form factor registered; using the NEUTRAL "
            f"{sym} coefficients.{detail} Register verified coefficients with "
            f"midas_hkls.form_factors.register_ion, or import "
            f"midas_pdf.ionic_form_factors for the species it ships.",
            RuntimeWarning,
            stacklevel=3,
        )
    e = tbl[sym]
    return np.asarray(e["a"], dtype=np.float64), np.asarray(e["b"], dtype=np.float64), float(e["c"])


# --------------------------------------------------------------- backend probe

def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


# ----------------------------------------------------------------- public API

def form_factor(s2: Any, element: str) -> Any:
    """f(s) for a single element evaluated at scalar or array s² = sin²θ/λ²  [Å⁻²].

    ``s2`` may be a Python float, numpy array, or torch tensor.  The return
    type matches the input.  Torch tensors flow gradients through ``s2``.
    """
    a, b, c = coefficients(element)
    if _is_torch_tensor(s2):
        import torch
        device = s2.device
        dtype = s2.dtype
        a_t = torch.as_tensor(a, dtype=dtype, device=device)
        b_t = torch.as_tensor(b, dtype=dtype, device=device)
        c_t = torch.as_tensor(c, dtype=dtype, device=device)
        s2u = s2.unsqueeze(-1)            # (..., 1)
        return torch.sum(a_t * torch.exp(-b_t * s2u), dim=-1) + c_t

    arr = np.asarray(s2, dtype=np.float64)
    s2u = arr[..., None]
    return float((a * np.exp(-b * s2u)).sum() + c) if arr.ndim == 0 else (a * np.exp(-b * s2u)).sum(axis=-1) + c


def form_factor_batch(s2: Any, elements: Iterable[str]) -> Any:
    """f(s, atom_j) over many atoms.  Returns shape (..., N_atoms).

    ``elements`` is a length-N iterable of element symbols (one per atom);
    duplicates are allowed.  ``s2`` is broadcast against the atom axis.
    """
    el_list = list(elements)
    if not el_list:
        raise ValueError("elements must be non-empty")
    coefs = [coefficients(e) for e in el_list]
    a_stack = np.stack([c[0] for c in coefs])  # (N, 4)
    b_stack = np.stack([c[1] for c in coefs])  # (N, 4)
    c_stack = np.array([c[2] for c in coefs])  # (N,)

    if _is_torch_tensor(s2):
        import torch
        device = s2.device
        dtype = s2.dtype
        a_t = torch.as_tensor(a_stack, dtype=dtype, device=device)   # (N, 4)
        b_t = torch.as_tensor(b_stack, dtype=dtype, device=device)
        c_t = torch.as_tensor(c_stack, dtype=dtype, device=device)   # (N,)
        s2u = s2.unsqueeze(-1).unsqueeze(-1)                         # (..., 1, 1)
        terms = a_t * torch.exp(-b_t * s2u)                          # (..., N, 4)
        return terms.sum(dim=-1) + c_t                                # (..., N)

    arr = np.asarray(s2, dtype=np.float64)
    s2u = arr[..., None, None]                                        # (..., 1, 1)
    terms = a_stack * np.exp(-b_stack * s2u)                          # (..., N, 4)
    return terms.sum(axis=-1) + c_stack                               # (..., N)
