"""Structure factors (|F|^2) and absorption for realistic spot intensities.

Replaces the Lorentz-polarisation x Debye-Waller *proxy* in the digital twin
with true |F|^2 * LP from the crystal structure (via ``midas_hkls``), so the
detectability floor keeps the reflections that are actually measurable.  This
matters most for large-cell phases such as garnet, whose many high-order
reflections are geometrically accessible but often weak.

Also provides DAC transmission (diamond + gasket + sample) vs energy, which sets
the usable flux and informs the energy choice.

Asymmetric-unit atom positions are literature Wyckoff coordinates; the space
group's symmetry operations expand them to the full cell.  Positions are
approximate -- adequate for *relative* intensities / detectability, not for
precise structure refinement.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch

from .crystal import get_material

# Asymmetric-unit atoms: material -> list of (element, (x,y,z), B_iso).
_ASU_ATOMS: Dict[str, list] = {
    "mgo": [("Mg", (0.0, 0.0, 0.0), 0.3), ("O", (0.5, 0.5, 0.5), 0.4)],
    "cbn": [("B", (0.0, 0.0, 0.0), 0.3), ("N", (0.25, 0.25, 0.25), 0.3)],
    "zirconia_cubic": [("Zr", (0.0, 0.0, 0.0), 0.4), ("O", (0.25, 0.25, 0.25), 0.6)],
    "ruby": [("Al", (0.0, 0.0, 0.3521), 0.3), ("O", (0.3064, 0.0, 0.25), 0.4)],
    "alumina": [("Al", (0.0, 0.0, 0.3521), 0.3), ("O", (0.3064, 0.0, 0.25), 0.4)],
    "zirconia_monoclinic": [
        ("Zr", (0.2758, 0.0400, 0.2100), 0.4),
        ("O", (0.070, 0.332, 0.345), 0.6),
        ("O", (0.450, 0.758, 0.480), 0.6)],
    "garnet_pyrope": [
        ("Mg", (0.0, 0.25, 0.125), 0.6),
        ("Al", (0.0, 0.0, 0.0), 0.4),
        ("Si", (0.375, 0.0, 0.25), 0.4),
        ("O", (0.0329, 0.0503, 0.6533), 0.6)],
    "ringwoodite": [
        ("Mg", (0.5, 0.5, 0.5), 0.5),
        ("Si", (0.125, 0.125, 0.125), 0.4),
        ("O", (0.2437, 0.2437, 0.2437), 0.6)],
    "iron_bcc": [("Fe", (0.0, 0.0, 0.0), 0.35)],
    "steel_fcc": [("Fe", (0.0, 0.0, 0.0), 0.35)],
    "tungsten": [("W", (0.0, 0.0, 0.0), 0.25)],
    "gold": [("Au", (0.0, 0.0, 0.0), 0.4)],
    "tantalum": [("Ta", (0.0, 0.0, 0.0), 0.3)],
}


def has_structure(material_key: str) -> bool:
    return material_key in _ASU_ATOMS


@lru_cache(maxsize=None)
def build_crystal(material_key: str):
    """A ``midas_hkls.Crystal`` with the material's asymmetric-unit atoms."""
    from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
    if material_key not in _ASU_ATOMS:
        raise KeyError(f"no atom structure defined for {material_key!r}")
    mat = get_material(material_key)
    latt = Lattice(*mat.lattice)
    sg = SpaceGroup.from_number(mat.space_group_number)
    atoms = [Atom(element=el, fract=xyz, B_iso=b)
             for (el, xyz, b) in _ASU_ATOMS[material_key]]
    return Crystal(lattice=latt, space_group=sg, atoms=atoms, name=material_key)


def reflection_intensities(material_key: str, hkls_int, thetas,
                           wavelength_A: float) -> np.ndarray:
    """|F|^2 * Lorentz-polarisation per reflection (normalised to max 1).

    Falls back to a Lorentz-polarisation x Debye-Waller proxy if no atom
    structure is defined for the material.
    """
    import midas_hkls as mh
    tt = torch.as_tensor(thetas, dtype=torch.float64) * 2.0
    lp = mh.lorentz_polarization(tt)
    if not has_structure(material_key):
        s = torch.sin(tt / 2.0) / wavelength_A
        inten = (lp * torch.exp(-2.0 * 0.5 * s * s)).cpu().numpy()
        return inten / inten.max()
    crystal_t = build_crystal(material_key).to_torch(dtype=torch.float64)
    hkl = torch.as_tensor(np.asarray(hkls_int), dtype=torch.float64)
    F = mh.structure_factors(crystal_t, hkl, wavelength_A=wavelength_A)
    fsq = (F.abs() ** 2).to(torch.float64)
    inten = (fsq * lp).cpu().numpy()
    m = inten.max()
    return inten / m if m > 0 else inten


# --------------------------------------------------------------------------- #
#  DAC transmission (absorption)                                              #
# --------------------------------------------------------------------------- #
def dac_transmission(
    energy_keV: float,
    *,
    diamond_mm: float = 4.0,          # two anvils, total path
    gasket_element: str = "Fe",
    gasket_mm: float = 0.05,          # steel gasket in the beam path
    sample_element: str = "Al",
    sample_mm: float = 0.1,
) -> Dict[str, float]:
    """Transmitted fraction through the DAC path (diamond + gasket + sample).

    Higher energy transmits better; this sets the usable flux and, with |F|^2,
    the exposure budget.
    """
    import midas_hkls.absorption as ab
    wl = 12.39842 / energy_keV                     # Angstrom
    def mu_cm(element):                            # linear atten. coeff (1/cm)
        return float(ab.linear_absorption_coefficient(element, wl))
    t_dia = np.exp(-mu_cm("C") * diamond_mm * 0.1)
    t_gas = np.exp(-mu_cm(gasket_element) * gasket_mm * 0.1)
    t_sam = np.exp(-mu_cm(sample_element) * sample_mm * 0.1)
    return {"energy_keV": energy_keV, "T_diamond": float(t_dia),
            "T_gasket": float(t_gas), "T_sample": float(t_sam),
            "T_total": float(t_dia * t_gas * t_sam)}
