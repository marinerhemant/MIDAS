"""Material presets and reflection generation for XAF-HEDM.

Wraps :func:`midas_diffract.hkls_for_forward_model` (which itself wraps
:mod:`midas_hkls`) so callers only pass a material key + wavelength + 2theta
cutoff.  Zirconia is the lead material; a few high-symmetry references and the
fiducial metals are included so the sweep can compare.

Lattice parameters are room-pressure literature values (angstroms / degrees).
Under load these shift per the material's equation of state; the point here is
the *reflection geometry*, so nominal cell values are sufficient for design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class Material:
    """A crystallographic phase: space-group number + lattice parameters."""
    name: str
    space_group_number: int
    lattice: Tuple[float, float, float, float, float, float]  # a,b,c,al,be,ga
    note: str = ""


#: Registry of built-in materials.  Add entries here to expand the sweep.
MATERIALS: Dict[str, Material] = {
    # --- Zirconia (lead sample) -------------------------------------------
    "zirconia_monoclinic": Material(
        "zirconia_monoclinic", 14, (5.1505, 5.2116, 5.3173, 90.0, 99.23, 90.0),
        "Baddeleyite ZrO2, P2_1/c (#14), ambient phase"),
    "zirconia_tetragonal": Material(
        "zirconia_tetragonal", 137, (3.640, 3.640, 5.270, 90.0, 90.0, 90.0),
        "t-ZrO2, P4_2/nmc (#137)"),
    "zirconia_cubic": Material(
        "zirconia_cubic", 225, (5.090, 5.090, 5.090, 90.0, 90.0, 90.0),
        "c-ZrO2, Fm-3m (#225)"),
    "zirconia_ortho_oI": Material(
        "zirconia_ortho_oI", 61, (5.070, 5.079, 5.257, 90.0, 90.0, 90.0),
        "o-ZrO2 (orthoI, Pbca #61), high pressure"),
    # --- High-symmetry references -----------------------------------------
    "iron_bcc": Material(
        "iron_bcc", 229, (2.8665, 2.8665, 2.8665, 90.0, 90.0, 90.0),
        "alpha-Fe, Im-3m (#229)"),
    "steel_fcc": Material(
        "steel_fcc", 225, (3.597, 3.597, 3.597, 90.0, 90.0, 90.0),
        "gamma-Fe / austenite, Fm-3m (#225)"),
    "ruby": Material(
        "ruby", 167, (4.7607, 4.7607, 12.9947, 90.0, 90.0, 120.0),
        "Corundum Al2O3:Cr, R-3c (#167), DAC pressure standard"),
    "alumina": Material(
        "alumina", 167, (4.7607, 4.7607, 12.9947, 90.0, 90.0, 120.0),
        "Corundum Al2O3, R-3c (#167); same structure as ruby (undoped)"),
    # --- Geophysical cubic phases (large cell -> spot-rich despite cubic) ---
    "garnet_pyrope": Material(
        "garnet_pyrope", 230, (11.459, 11.459, 11.459, 90.0, 90.0, 90.0),
        "Pyrope Mg3Al2Si3O12, Ia-3d (#230), a=11.46 A (large cubic cell)"),
    "ringwoodite": Material(
        "ringwoodite", 227, (8.070, 8.070, 8.070, 90.0, 90.0, 90.0),
        "Ringwoodite Mg2SiO4 spinel, Fd-3m (#227), a=8.07 A"),
    "mgo": Material(
        "mgo", 225, (4.211, 4.211, 4.211, 90.0, 90.0, 90.0),
        "Periclase MgO, Fm-3m (#225), small cubic cell (spot-starved for HEDM)"),
    "cbn": Material(
        "cbn", 216, (3.615, 3.615, 3.615, 90.0, 90.0, 90.0),
        "Cubic boron nitride, F-43m (#216), small cell"),
    # --- Fiducial marker metals (high-Z, single-crystal) ------------------
    "tungsten": Material(
        "tungsten", 229, (3.1652, 3.1652, 3.1652, 90.0, 90.0, 90.0),
        "W, Im-3m (#229); high-Z fiducial"),
    "gold": Material(
        "gold", 225, (4.0782, 4.0782, 4.0782, 90.0, 90.0, 90.0),
        "Au, Fm-3m (#225); high-Z fiducial"),
    "tantalum": Material(
        "tantalum", 229, (3.3058, 3.3058, 3.3058, 90.0, 90.0, 90.0),
        "Ta, Im-3m (#229); high-Z fiducial"),
}


def get_material(key: str) -> Material:
    if key not in MATERIALS:
        raise KeyError(
            f"unknown material {key!r}; known: {sorted(MATERIALS)}")
    return MATERIALS[key]


def build_reflections(
    material_key: str,
    wavelength_A: float,
    two_theta_max_deg: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(hkls_cart, thetas, hkls_int)`` for a material.

    ``hkls_cart`` are Cartesian reciprocal G-vectors (1/A), ``thetas`` Bragg
    angles (rad), ``hkls_int`` integer Miller indices -- exactly the triple
    :class:`midas_diffract.HEDMForwardModel` consumes.  One row per
    Laue-equivalent reflection.
    """
    # Imported lazily so the base import stays light and errors are localised.
    import midas_diffract as md
    from midas_hkls import Lattice, SpaceGroup

    mat = get_material(material_key)
    sg = SpaceGroup.from_number(mat.space_group_number)
    latt = Lattice(*mat.lattice)
    return md.hkls_for_forward_model(
        sg, latt, wavelength_A=wavelength_A,
        two_theta_max_deg=two_theta_max_deg, dtype=dtype)
