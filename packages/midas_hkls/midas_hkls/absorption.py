"""Element-specific X-ray linear absorption coefficient μ at a given wavelength.

Uses the NIST XCOM tabulated mass attenuation coefficient (μ/ρ) shipped at
``data/nist_mac.json`` (generated from xraylib's wrapper of the NIST tables;
Hubbell-Seltzer 1995 foundational dataset).

Interpolation is linear in (log E, log μ/ρ) space — the standard practice for
these tables and physically motivated (μ ∝ E^{-3} far from edges).  This is
fully differentiable through λ (and the optional density override) when those
inputs are torch tensors.

Convention notes
----------------

* Wavelength is in Ångströms throughout.
* μ is returned in cm⁻¹ (matches NIST convention).  Multiply by path length
  in cm to get the dimensionless absorption coefficient ``μ·t``.
* For a sample of thickness ``t_um`` (µm), transmission is
  ``exp(-μ * t_um * 1e-4)``.

Edge handling
-------------

K-, L-, M-edges have discontinuities in μ.  The shipped grid includes points
at ``E_edge ± 0.1%`` for the major K-edges (Z=14 through Z=92 covering all
HEDM-relevant elements).  Interpolation across an edge is therefore well-
behaved.  For sub-keV applications, query with caution — the table starts at
1 keV.

Public API::

    from midas_hkls.absorption import linear_absorption_coefficient
    mu = linear_absorption_coefficient("Ti", wavelength_A=0.173)   # cm⁻¹

Inputs may be Python floats, numpy scalars, or torch tensors.  Tensor inputs
produce tensor outputs with grad-flow preserved (CPU, CUDA, MPS).
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .anomalous import (
    Z_for,
    _HC_eV_A,
    _normalize_symbol,
    wavelength_to_energy_eV,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "linear_absorption_coefficient",
    "mass_attenuation_coefficient",
    "element_density",
    "atomic_mass",
    "available_elements_absorption",
]


# ------------------------------------------------------------------ tables


@lru_cache(maxsize=1)
def _properties() -> dict:
    """Element atomic mass + bulk density (Z=1-92)."""
    return json.loads(
        files("midas_hkls").joinpath("data/element_properties.json").read_text()
    )


@lru_cache(maxsize=1)
def _nist_mac() -> dict:
    """Shared NIST mass attenuation coefficient table."""
    return json.loads(
        files("midas_hkls").joinpath("data/nist_mac.json").read_text()
    )


@lru_cache(maxsize=1)
def _nist_energies_keV() -> np.ndarray:
    return np.asarray(_nist_mac()["_energy_keV"], dtype=np.float64)


@lru_cache(maxsize=1)
def _nist_log_E() -> np.ndarray:
    return np.log(_nist_energies_keV())


def available_elements_absorption() -> list[str]:
    """Elements with both density and NIST μ/ρ data available."""
    props = set(_properties().keys())
    mac = set(k for k in _nist_mac().keys() if not k.startswith("_"))
    return sorted(props & mac)


# ------------------------------------------------------------------ helpers


def _lookup_prop(element: str, key: str) -> float:
    sym = _normalize_symbol(element)
    table = _properties()
    if sym not in table:
        raise KeyError(
            f"no absorption properties for element {element!r} "
            f"(normalized {sym!r}); add it to data/element_properties.json"
        )
    val = table[sym].get(key)
    if val is None:
        raise KeyError(f"element {sym!r} is missing {key!r} in property table")
    return float(val)


def atomic_mass(element: str) -> float:
    """Standard atomic weight [g/mol]."""
    return _lookup_prop(element, "atomic_mass")


def element_density(element: str) -> float:
    """Tabulated bulk density [g/cm³] of the pure element at STP."""
    return _lookup_prop(element, "density")


def _mu_rho_grid(element: str) -> np.ndarray:
    """Return raw μ/ρ values at each grid energy for ``element``."""
    sym = _normalize_symbol(element)
    table = _nist_mac()
    if sym not in table:
        raise KeyError(f"no NIST μ/ρ data for element {element!r} (normalized {sym!r})")
    return np.asarray(table[sym]["mu_rho"], dtype=np.float64)


# ------------------------------------------------------------------ backend


def _is_torch(x) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _np_log_log_interp(log_E: float, log_E_grid: np.ndarray, log_mac_grid: np.ndarray
                       ) -> float:
    """Linear interpolation in (log E, log μ/ρ) space.

    Extrapolation clamped to end-points.  Gracefully handles entries with
    μ/ρ = 0 (out-of-tabulated-range below the K-edge etc.) by skipping
    them and using the nearest valid neighbor.
    """
    valid = np.isfinite(log_mac_grid) & (log_mac_grid > -1e30)
    if not np.any(valid):
        raise ValueError("no valid μ/ρ data in this energy range")
    xp = log_E_grid[valid]
    fp = log_mac_grid[valid]
    if log_E <= xp[0]:
        return float(fp[0])
    if log_E >= xp[-1]:
        return float(fp[-1])
    return float(np.interp(log_E, xp, fp))


def _torch_log_log_interp(log_E, log_E_grid_t, log_mac_grid_t):
    """torch-native log-log linear interpolation.  Differentiable in log_E.

    Out-of-range μ/ρ = 0 entries become -inf in log-space; we treat them as
    "no data" via masking.  For simplicity, we assume the in-range subset
    of the grid is contiguous (true for NIST tables: invalid points are at
    the low-energy end, below absorption edges).
    """
    import torch

    # Mask out -inf (log of 0) entries
    valid_mask = log_mac_grid_t > -1e30                        # bool, (K,)
    valid_idx = torch.nonzero(valid_mask, as_tuple=True)[0]    # int, (n_valid,)
    if valid_idx.numel() == 0:
        raise ValueError("no valid μ/ρ data")
    log_E_g = log_E_grid_t[valid_idx]
    log_mac_g = log_mac_grid_t[valid_idx]

    log_E_clamped = torch.clamp(log_E, min=log_E_g[0], max=log_E_g[-1])
    # bin index: smallest i such that log_E_g[i] >= log_E_clamped
    idx_hi = torch.searchsorted(log_E_g, log_E_clamped.unsqueeze(-1)).squeeze(-1)
    idx_hi = torch.clamp(idx_hi, min=1, max=log_E_g.numel() - 1)
    idx_lo = idx_hi - 1
    x_lo = log_E_g[idx_lo]
    x_hi = log_E_g[idx_hi]
    y_lo = log_mac_g[idx_lo]
    y_hi = log_mac_g[idx_hi]
    t = (log_E_clamped - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


# ------------------------------------------------------------------ μ kernels


def mass_attenuation_coefficient(
    element: str,
    wavelength_A,
    *,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
):
    """σ_mass [cm²/g] for ``element`` at ``wavelength_A``.

    Looks up the NIST table at the corresponding energy and interpolates in
    (log E, log μ/ρ) space.  Differentiable through ``wavelength_A`` when it
    is a torch tensor.

    Returns tensor on requested device/dtype if input is a tensor, else float.
    """
    log_mac_grid_np = np.log(np.where(_mu_rho_grid(element) > 0,
                                       _mu_rho_grid(element), 1e-30))
    log_E_grid_np = _nist_log_E()

    use_torch = _is_torch(wavelength_A) or dtype is not None or device is not None
    if use_torch:
        import torch
        if not _is_torch(wavelength_A):
            wavelength_A = torch.as_tensor(
                float(wavelength_A), dtype=dtype or torch.float64, device=device
            )
        if dtype is None:
            dtype = wavelength_A.dtype
        if device is None:
            device = wavelength_A.device

        # E [keV] = (12398.42 / λ_Å) / 1000
        E_keV = _HC_eV_A / wavelength_A / 1000.0
        log_E = torch.log(E_keV)

        log_E_grid_t = torch.as_tensor(log_E_grid_np, dtype=dtype, device=device)
        log_mac_grid_t = torch.as_tensor(log_mac_grid_np, dtype=dtype, device=device)

        log_mac = _torch_log_log_interp(log_E, log_E_grid_t, log_mac_grid_t)
        return torch.exp(log_mac)

    # numpy / float path
    E_keV = float(_HC_eV_A / float(wavelength_A) / 1000.0)
    log_E = float(np.log(E_keV))
    log_mac = _np_log_log_interp(log_E, log_E_grid_np, log_mac_grid_np)
    return float(np.exp(log_mac))


def linear_absorption_coefficient(
    element: str,
    wavelength_A,
    *,
    density_g_cm3: Optional[Union[float, "torch.Tensor"]] = None,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
):
    """μ [cm⁻¹] = ρ · σ_mass(λ) for ``element`` at ``wavelength_A``.

    Uses the NIST XCOM tabulated μ/ρ via log-log interpolation.

    Parameters
    ----------
    element        : element symbol (case-sensitive, e.g. ``"Ti"``)
    wavelength_A   : Ångströms; float | numpy scalar | torch tensor.
                     Differentiable when a torch tensor.
    density_g_cm3  : override the tabulated bulk density.  Useful for
                     porous / alloy samples.  May be a torch tensor for
                     joint-refinement scenarios.
    dtype, device  : applied if either input is forced to torch.

    Returns
    -------
    μ same backend type as inputs (tensor in → tensor out).  Units cm⁻¹.
    """
    sigma_mass = mass_attenuation_coefficient(
        element, wavelength_A, dtype=dtype, device=device
    )
    rho = density_g_cm3 if density_g_cm3 is not None else element_density(element)

    if _is_torch(sigma_mass):
        import torch
        rho_t = (
            rho
            if _is_torch(rho)
            else torch.as_tensor(float(rho), dtype=sigma_mass.dtype, device=sigma_mass.device)
        )
        return sigma_mass * rho_t
    return float(sigma_mass) * float(rho)
