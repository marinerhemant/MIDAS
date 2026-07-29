"""Phase 5 tests: inverse dislocation typing (slip system, Burgers sign, position)."""
import pytest
import torch

from midas_dfxm.detect import (
    identify_dislocation,
    match_residual,
    weak_beam_stack,
)
from midas_dfxm.dislocation import (
    cubic_stiffness,
    dislocation_deformation_field,
    stroh_dislocation,
)

DT = torch.float64
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
PLANE, BURGERS = (1, 1, 1), (1, -1, 0)
REFL = [(2, -2, 0), (2, 0, 0), (0, 2, 0), (1, 1, 1)]


def _grid(n=24, half=8.0):
    xs = torch.linspace(-half, half, n, dtype=DT)
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=DT), indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def _planted_edge(sign=1, core=(0.0, 0.0, 0.0)):
    b = tuple(sign * x for x in BURGERS)
    line = torch.linalg.cross(torch.tensor(PLANE, dtype=DT), torch.tensor(BURGERS, dtype=DT))
    return stroh_dislocation(CU, burgers=b, slip_normal=PLANE, line=line,
                             core_position=core, core_radius_um=0.4)


@pytest.mark.unit
def test_weak_beam_stack_shape():
    pts = _grid()
    field = dislocation_deformation_field(pts, _planted_edge())
    stack = weak_beam_stack(field, REFL)
    assert stack.shape == (len(REFL), pts.shape[0])
    assert torch.isfinite(stack).all()


@pytest.mark.unit
def test_burgers_sign_recovered():
    # The Borgi-2025 gap: sign of b. Wrong-sign template must have larger residual.
    pts = _grid()
    obs = weak_beam_stack(dislocation_deformation_field(pts, _planted_edge(sign=+1)), REFL)
    from midas_dfxm.detect import _normalize
    obs_n = _normalize(obs)
    r_right = match_residual(obs_n, weak_beam_stack(
        dislocation_deformation_field(pts, _planted_edge(sign=+1)), REFL))
    r_wrong = match_residual(obs_n, weak_beam_stack(
        dislocation_deformation_field(pts, _planted_edge(sign=-1)), REFL))
    assert float(r_right) < 1e-9
    assert float(r_wrong) > 100 * float(r_right + 1e-12)


@pytest.mark.slow
def test_identify_recovers_system_and_sign():
    pts = _grid()
    obs = weak_beam_stack(dislocation_deformation_field(pts, _planted_edge(sign=+1)), REFL)
    labels = identify_dislocation(obs, pts, CU, REFL, refine_position=False, top_k=1)
    best = labels[0]
    # Correct slip plane (up to sign), correct character, correct Burgers incl. sign.
    assert tuple(abs(x) for x in best.slip_normal) == (1, 1, 1)
    assert best.character == "edge"
    assert best.burgers in [(1, -1, 0), (-1, 1, 0)]
    # The identified b matches the planted +[1,-1,0] (not the -sign).
    assert best.burgers == (1, -1, 0)


@pytest.mark.slow
def test_identify_refines_core_position():
    pts = _grid()
    true_core = (2.0, -1.5, 0.0)
    obs = weak_beam_stack(
        dislocation_deformation_field(pts, _planted_edge(core=true_core)), REFL)
    labels = identify_dislocation(obs, pts, CU, REFL, refine_position=True, top_k=1)
    px, py, _ = labels[0].core_position
    assert abs(px - true_core[0]) < 1.0
    assert abs(py - true_core[1]) < 1.0
