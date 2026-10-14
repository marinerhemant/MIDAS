"""The ring-split gate must not mistake edge-truncated partial arcs for a bad basis.

On the Varex frame the plain inner-vs-outer ratio was 1.84x and the gate FAILED, with the
message 'analytical basis incomplete on rings >= 18'. A refuter then showed that 47 % of the
held-out fits lay within 50 px of the panel edge against 5 % of the training fits, that
edge-truncated peaks have a median residual ~4.6x an interior one's, and that dropping the
near-edge fits gives 1.33x. The outer rings are also partial corner arcs (8-34 of 72 azimuth
bins). The ratio was measuring which fits fall in the outer third.

The gate now (i) drops fits within `edge_margin_px` of a detector edge or module gap, (ii)
compares only rings holding >= `min_ring_fit_fraction` of the best-covered ring's fits, and
(iii) cannot fail unless `allow_fail=True`, because its thresholds were never calibrated.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from midas_calibrate_v2.pipelines import diagnostics

N_RING, N_PER = 12, 50
SHAPE = (1000, 1000)                     # (NrPixelsY, NrPixelsZ)


def _fits(near_edge_fraction=0.0, outer_fit_count=N_PER, edge_boost=6.0, seed=0):
    """Rings 0-7 are 'train', 8-11 'test'. Test-ring fits near the edge carry `edge_boost` x the
    residual of interior ones; interior residuals are identical across rings, so a correct
    gate sees NO basis problem."""
    rng = np.random.default_rng(seed)
    Y, Z, ring, resid = [], [], [], []
    for r in range(N_RING):
        n = N_PER if r < 8 else outer_fit_count
        base = np.abs(rng.normal(0, 10.0 * 1.4826, n))
        near = np.zeros(n, bool)
        if r >= 8:
            near[: int(round(near_edge_fraction * n))] = True
        y = np.where(near, 10.0, 500.0)              # 10 px from the edge vs the middle
        resid.append(np.where(near, base * edge_boost, base))
        Y.append(y); Z.append(np.full(n, 500.0)); ring.append(np.full(n, r))
    cat = np.concatenate
    f = SimpleNamespace(
        Y_pix=torch.tensor(cat(Y)), Z_pix=torch.tensor(cat(Z)),
        ring_two_theta_deg=torch.zeros(len(cat(Y)), dtype=torch.float64),
        ring_idx=torch.tensor(cat(ring), dtype=torch.long),
        rho_d=torch.tensor(1000.0), panel_idx=None)
    return f, cat(resid)


def _patch(monkeypatch, resid_uE):
    monkeypatch.setattr(diagnostics, "pseudo_strain_residual",
                        lambda *a, **k: torch.tensor(resid_uE * 1e-6))


def test_edge_truncated_outer_arcs_do_not_fail_the_basis(monkeypatch):
    f, r = _fits(near_edge_fraction=0.6)
    _patch(monkeypatch, r)
    raw = diagnostics.cross_validation_gate(f, {}, allow_fail=True)            # no shape: no cut
    assert raw.severity == "fail" and raw.metrics["ratio"] > 1.5, "premise: the uncut ratio is bad"
    g = diagnostics.cross_validation_gate(f, {}, detector_shape=SHAPE, allow_fail=True)
    assert g.severity == "ok", (g.severity, g.message)
    assert g.metrics["ratio_raw"] > 1.5 and g.metrics["ratio"] < 1.3
    assert g.metrics["n_edge_excluded"] > 0


def test_a_genuine_outer_ring_misfit_in_the_interior_is_still_seen(monkeypatch):
    """Sensitivity: the cut must not hide a real effect. Interior outer-ring fits with a
    2.5x residual still trip the gate."""
    f, r = _fits(near_edge_fraction=0.0)
    r = np.where(f.ring_idx.numpy() >= 8, r * 2.5, r)
    _patch(monkeypatch, r)
    g = diagnostics.cross_validation_gate(f, {}, detector_shape=SHAPE, allow_fail=True)
    assert g.severity == "fail" and g.metrics["ratio"] > 2.0


def test_sparsely_covered_rings_are_left_out_of_the_comparison(monkeypatch):
    """Rings with few fits are partial arcs; they are not compared, whatever their residual."""
    f, r = _fits(outer_fit_count=10)
    r = np.where(f.ring_idx.numpy() >= 8, r * 5.0, r)          # a loud, sparse outer set
    _patch(monkeypatch, r)
    g = diagnostics.cross_validation_gate(f, {}, detector_shape=SHAPE, allow_fail=True)
    assert g.metrics["n_rings_used"] == 8.0
    assert g.severity == "ok", (g.severity, g.message)


def test_it_cannot_fail_by_default(monkeypatch):
    f, r = _fits(near_edge_fraction=0.0)
    r = np.where(f.ring_idx.numpy() >= 8, r * 2.5, r)
    _patch(monkeypatch, r)
    g = diagnostics.cross_validation_gate(f, {}, detector_shape=SHAPE)
    assert g.severity == "warn" and "not calibrated" in g.message


def test_module_gaps_count_as_edges_at_fit_window_scale(monkeypatch):
    """A fit next to a module gap is truncated like one at the border, but only within about
    one fit window of it, not within the (much larger) border margin."""
    from midas_calibrate_v2.forward.panels import PanelLayout
    lay = PanelLayout.regular(2, 2, 40, 40, gap_y=20, gap_z=20)
    ny, nz = lay.panel_index_mask.shape
    d_border, d_gap = diagnostics._edge_distance_px(
        np.array([39.0, 20.0]), np.array([20.0, 20.0]), (ny, nz), lay)
    assert d_gap[0] < 5.0 and d_gap[1] > 15.0, d_gap   # next to the gap vs mid-panel
    assert d_border[1] >= 19.0, d_border               # the border is far from both


def test_a_tiled_detector_keeps_its_interior_fits(monkeypatch):
    """Regression: with ONE 100 px margin applied to gaps too, a Pilatus (modules 243 x 195 px,
    nothing farther than ~97 px from a gap) lost every fit and the gate said 'only 0 test fits'."""
    from midas_calibrate_v2.forward.panels import PanelLayout
    lay = PanelLayout.regular(3, 3, 200, 200, gap_y=7, gap_z=17)
    ny, nz = lay.panel_index_mask.shape
    rng = np.random.default_rng(0)
    # fits in module interiors: >= 30 px from any gap, ~ 300+ px from the border on average
    Y = rng.uniform(250, ny - 250, 400)
    Z = rng.uniform(250, nz - 250, 400)
    d_border, d_gap = diagnostics._edge_distance_px(Y, Z, (ny, nz), lay)
    interior = d_gap >= 10.0
    assert interior.mean() > 0.5, interior.mean()
    assert ((d_border < 100.0) | (d_gap < 10.0))[interior].sum() == 0
