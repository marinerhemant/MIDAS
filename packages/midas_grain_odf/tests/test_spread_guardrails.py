"""E4 tests: spread-DOF guardrails.

(a) a spread parameter finishing PINNED at its physical ceiling flags the
    fit invalid (warning + result flag) — the emerson failure mode where
    both DOFs shot to their clamps at default LRs and the loss worsened;
(b) ``lr_*_spread="auto"`` scales the SGD step from the first gradient;
(c) robust particle-spread statistics (weighted median + weight-within)
    are reported alongside the theta_max-sensitive wRMS.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from conftest import make_model, random_orientation  # noqa: E402

# Reuse the microstrain phantom builder.
from test_grain_odf_microstrain import _build_synth_strain  # noqa: E402

from midas_grain_odf.inversion import fit_grain_odf  # noqa: E402
from midas_grain_odf.odf import ParticleODF, particle_spread_stats  # noqa: E402

DT = torch.float64


def _synth(planted=1e-3, patch_F=5, patch_P=21, sigma_yz=1.0, sigma_f=0.6):
    torch.manual_seed(0)
    model = make_model()
    position = torch.zeros(3, dtype=DT)
    R_avg = random_orientation(seed=7).to(DT)
    synth = _build_synth_strain(model, position, R_avg, sigma_yz, sigma_f,
                                patch_F, patch_P, planted)
    return model, position, R_avg, synth


def _fit(model, position, R_avg, synth, **over):
    odf = ParticleODF(R_avg=R_avg, K=1, theta_max=math.radians(0.1), seed=0,
                      init_axis_angle=torch.zeros(1, 3, dtype=DT))
    kw = dict(
        patch_F=5, patch_P=21, sigma_yz=1.0, sigma_f=0.6,
        delta_iters=1, inner_steps=60,
        lr_axis_angle=0.0, lr_logits=0.0, loss_norm="mean",
        refine_strain_spread=True,
        strain_spread_init=1e-4,
        strain_spread_microstrain_units=True,
    )
    kw.update(over)
    return fit_grain_odf(
        odf, model, position,
        synth["measured_y"], synth["measured_z"], synth["measured_f"],
        synth["measured_patches"], synth["spot_indexer"],
        **kw,
    )


def test_pinned_at_ceiling_warns_and_flags():
    """An absurdly large LR drives σ_ε into the ceiling → invalid-fit
    warning + result flag (E4a)."""
    model, position, R_avg, synth = _synth(planted=1e-3)
    with pytest.warns(UserWarning, match="PINNED at its ceiling"):
        res = _fit(model, position, R_avg, synth,
                   lr_strain_spread=1e3, inner_steps=20)
    assert res.strain_spread_pinned is True


def test_auto_lr_recovers_planted_spread():
    """lr_strain_spread='auto' must converge near the planted σ_ε without
    hand-tuning (E4b) — and must NOT pin."""
    model, position, R_avg, synth = _synth(planted=1e-3)
    res = _fit(model, position, R_avg, synth,
               lr_strain_spread="auto", inner_steps=200)
    assert res.strain_spread_pinned is False
    fit = float(res.strain_spread_fit.item())
    assert fit == pytest.approx(1e-3, rel=0.5), f"auto-lr recovered {fit:.2e}"


def test_spread_stats_reports_robust_alternatives():
    R_avg = random_orientation(seed=3).to(DT)
    odf = ParticleODF(R_avg=R_avg, K=24, theta_max=math.radians(2.0), seed=1)
    stats = particle_spread_stats(odf, within_deg=1.0)
    for key in ("wrms_deg", "weighted_median_deg", "weight_within",
                "within_deg", "n_particles"):
        assert key in stats
    assert stats["n_particles"] == 24
    assert 0.0 <= stats["weight_within"] <= 1.0 + 1e-9
    assert stats["weighted_median_deg"] <= math.degrees(math.radians(2.0)) + 1e-6
    # And it is reachable from a fit result.
    model, position, R_avg2, synth = _synth(planted=0.0)
    res = _fit(model, position, R_avg2, synth, inner_steps=5,
               lr_strain_spread=5e-4)
    st = res.spread_stats(within_deg=1.0)
    assert "weighted_median_deg" in st and "wrms_deg" in st
