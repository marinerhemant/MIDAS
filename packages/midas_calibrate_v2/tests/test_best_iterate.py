"""autocalibrate must return the BEST E<->M iterate, not the last one.

The alternating loop is not monotonic — the E-step re-extracts peaks at the
new geometry, so a late iteration can land in a worse basin than an earlier
one. Observed on real CeO2 (pokharel_jul26, GE5): iter 3 = 17.9 ue, iter 4 =
72.0 ue. Returning the last iterate shipped the 72 ue geometry.

v1 C keeps the best across nIterations (FF_HEDM/Example/Parameters.txt:
"Number of full optimization iterations (best result is kept)").
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_calibrate_v2.pipelines import single as single_mod


class _Rec:
    """Stand-in for IterRecord carrying only what the selection logic reads."""

    def __init__(self, strain):
        self.mean_strain_uE = strain


def test_best_iterate_is_adopted_over_a_worse_last(monkeypatch):
    """A run whose last iterate is worse than an earlier one must return the
    earlier geometry."""
    strains = [253.8, 296.4, 59.5, 17.9, 72.0]      # the real CeO2 sequence
    geoms = [1000.0 + i for i in range(len(strains))]

    best_strain = float("inf")
    best_unpacked = None
    history = []
    unpacked = None
    for s, g in zip(strains, geoms):
        unpacked = {"Lsd": torch.tensor([g], dtype=torch.float64)}
        history.append(_Rec(s))
        if s < best_strain:
            best_strain = s
            best_unpacked = {k: v.detach().clone() for k, v in unpacked.items()}

    # the selection rule as implemented in autocalibrate
    if best_unpacked is not None and best_strain < history[-1].mean_strain_uE:
        unpacked = best_unpacked

    assert best_strain == pytest.approx(17.9)
    assert float(unpacked["Lsd"]) == pytest.approx(1003.0), (
        "returned the last iterate (1004) instead of the best (1003)")


def test_source_selects_best_and_syncs_v1_params():
    """Guard the actual implementation, not just the rule.

    The adopted geometry must also be pushed back into v1_params: the
    residual-correction stage re-runs the E-step off v1_params, so a mismatch
    would extract peaks at one geometry and score them at another.
    """
    import inspect

    src = inspect.getsource(single_mod.autocalibrate)
    assert "best_strain" in src and "best_unpacked" in src, (
        "autocalibrate no longer tracks a best iterate")
    # the adoption block must write back into v1_params
    adopt = src.split("Adopt the best iterate", 1)
    assert len(adopt) == 2, "best-iterate adoption block is missing"
    assert "setattr(v1_params" in adopt[1].split("Post-MAP", 1)[0], (
        "best iterate is adopted but v1_params is left on the last iterate")
    # the residual-map baseline must be the adopted strain, not history[-1]
    assert "pre_strain = best_strain" in src, (
        "residual map is judged against the last iterate's strain")
