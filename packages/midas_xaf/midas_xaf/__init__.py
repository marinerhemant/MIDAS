"""midas-xaf — Cross-Axis Faceted HEDM (XAF-HEDM) design & simulation toolkit.

A far-field HEDM measurement performed through a cubic diamond anvil cell with a
~15 deg opening on all six faces.  Narrow omega wedges are collected through the
four equatorial faces per mounting; the cell is remounted about an orthogonal
axis so the top/bottom faces reach the equator, and both mountings are merged
into one reciprocal-space reconstruction (each mounting fills the other's
missing cone).

This package is a thin orchestration layer over ``midas-diffract``'s
differentiable forward model: it adds the two-mounting geometry, the omega-wedge
and exit-aperture (shadowing) access gates, Friedel/fiducial bookkeeping, and
determinability metrics -- above all the autograd strain-sensitivity metric that
answers whether (and with what cell opening) the measurement can resolve strain.

Quick start
-----------
    from midas_xaf import XAFConfig, XAFForwardModel, make_sample
    from midas_xaf import metrics

    cfg = XAFConfig(opening_full_deg=15.0, material="zirconia_monoclinic")
    fwd = XAFForwardModel(cfg)
    grains = make_sample(cfg)
    sim = fwd.simulate(grains)
    print("accessible spots:", len(sim.table))
    print(metrics.cross_axis_gain(fwd, grains))
"""
from .config import XAFConfig
from .crystal import MATERIALS, Material, build_reflections, get_material
from .sample import GrainPopulation, make_sample
from .forward import SpotTable, XAFForwardModel, XAFSimulation
from . import (autonomy, geometry, metrics, merge, reconstruct, sweep, coverage,
               micromech, report, synth, indexing, pipeline, structure,
               robustness, budget)

__version__ = "0.1.0"

__all__ = [
    "XAFConfig",
    "MATERIALS", "Material", "build_reflections", "get_material",
    "GrainPopulation", "make_sample",
    "SpotTable", "XAFForwardModel", "XAFSimulation",
    "geometry", "metrics", "merge", "reconstruct", "sweep", "coverage",
    "micromech", "report", "synth", "indexing", "pipeline", "structure",
    "robustness", "budget",
    "__version__",
]
