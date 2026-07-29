"""Fluorescence diagnostics for total-scattering data.

Fluorescence is the dominant *smooth* background contaminant in PDF: if the
incident energy is above an absorption edge of a sample element, that element
fluoresces isotropically, adding a broad baseline that biases the normalization.
Quantitative fluorescence subtraction is geometry- and detector-dependent and is
out of scope here; the practical, robust treatment is a refinable smooth
background (see ``midas_pdf.refine.refine_normalization(bg_order=...)``).

What this module provides is the **diagnostic**: given a composition and the
incident energy, which elements will fluoresce, on which lines, and how strongly
(K-shell yield) -- so the user knows whether a smooth-background term is needed
and whether to consider an energy change. Backed by ``data/fluor_edges.json``
(xraylib edges/yields tabulated at build time; no runtime xraylib).
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Iterable, List

import numpy as np

__all__ = ["wavelength_to_energy_keV", "expected_fluorescence"]

_HC_KEV_A = 12.398419843320026


@lru_cache(maxsize=1)
def _edges() -> dict:
    return json.loads(files("midas_pdf").joinpath("data/fluor_edges.json").read_text())


def wavelength_to_energy_keV(wavelength_A: float) -> float:
    return _HC_KEV_A / float(wavelength_A)


def expected_fluorescence(
    elements: Iterable[str],
    *,
    incident_energy_keV: float | None = None,
    wavelength_A: float | None = None,
    min_yield: float = 0.01,
) -> List[dict]:
    """Lines a composition will fluoresce at the given incident energy.

    Provide either ``incident_energy_keV`` or ``wavelength_A``. Returns a list of
    ``{element, shell, edge_keV, line_keV, yield}`` for every shell whose
    absorption edge lies below the incident energy and whose fluorescence yield
    exceeds ``min_yield``, sorted by yield (strongest first). An empty list means
    no significant fluorescence is expected — the data should be clean of it.
    """
    if (incident_energy_keV is None) == (wavelength_A is None):
        raise ValueError("give exactly one of incident_energy_keV or wavelength_A")
    E0 = (incident_energy_keV if incident_energy_keV is not None
          else wavelength_to_energy_keV(wavelength_A))
    tbl = _edges()
    out: List[dict] = []
    for el in elements:
        rec = tbl.get(el)
        if rec is None:
            continue
        for shell, edge_key, line_key, y_key in (
            ("K", "K_edge_keV", "Ka1_keV", "K_yield"),
            ("L3", "L3_edge_keV", "La1_keV", None),
        ):
            edge = rec.get(edge_key)
            if edge is None or edge >= E0:
                continue
            yld = rec.get(y_key) if y_key else None
            if yld is not None and yld < min_yield:
                continue
            out.append({
                "element": el, "shell": shell,
                "edge_keV": edge, "line_keV": rec.get(line_key),
                "yield": yld,
            })
    out.sort(key=lambda d: (d["yield"] is None, -(d["yield"] or 0.0)))
    return out


def fluorescence_report_sample_and_container(
    sample_composition,
    container_composition=None,
    *,
    incident_energy_keV: float | None = None,
    wavelength_A: float | None = None,
    min_yield: float = 0.05,
) -> dict:
    """Fluorescence check for both the sample AND its container.

    Container fluorescence lines that fall in the detector's sensitive band
    contaminate the low-Q signal and can look like a real amorphous hump.
    For Kapton (C/H/N/O) at hard-X-ray energies there is nothing --- but for
    quartz (Si), borosilicate (B/Si + traces of Fe), or metal cans (steel,
    Cu) it matters.

    Returns a dict with three keys:
        sample_lines    : list of fluorescence lines from the sample
        container_lines : list of fluorescence lines from the container
                          (empty list if container_composition is None)
        clean           : True iff both lists are empty at the min_yield cutoff
    """
    sample_elements = (list(sample_composition.elements)
                       if hasattr(sample_composition, "elements")
                       else list(sample_composition))
    sample_lines = expected_fluorescence(
        sample_elements, incident_energy_keV=incident_energy_keV,
        wavelength_A=wavelength_A, min_yield=min_yield)
    container_lines: List[dict] = []
    if container_composition is not None:
        container_elements = (list(container_composition.elements)
                              if hasattr(container_composition, "elements")
                              else list(container_composition))
        container_lines = expected_fluorescence(
            container_elements, incident_energy_keV=incident_energy_keV,
            wavelength_A=wavelength_A, min_yield=min_yield)
    return {
        "sample_lines":    sample_lines,
        "container_lines": container_lines,
        "clean":           (not sample_lines) and (not container_lines),
    }
