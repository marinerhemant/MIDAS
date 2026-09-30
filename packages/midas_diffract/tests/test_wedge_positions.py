"""Issue #11/#17: G and grain positions turn together about the wedge axis.

MIDAS Wedge convention (``midas_diffract.forward`` module doc): with the
Parameters-file ``Wedge`` W the rotation axis in the lab is
``n = R_y(-W) e_z = (-sin W, 0, cos W)`` and orientation / position live in
the ROTATION-STAGE frame, which sits at ``S = R_y(-W)`` relative to the lab
at omega = 0. So every stage-frame vector v (G = O g, or a position p) is
seen in the lab as

    v_lab(omega) = R_n(omega) @ S @ v  ==  R_y(-W) @ R_z(omega) @ v,

the map the FF C refiner uses (``DisplacementInTheSpot`` for p,
``CorrectForOme`` for G). The sample is a rigid body: G and position alike.

These tests hold the eager forward to an independent calculation: build
``R_n(omega)`` by Rodrigues' formula about ``n`` and ``S`` by Rodrigues about
lab y (NOT via the factorisation the code uses), map G and the position, form
``k_out = k_in + G_lab`` and intersect the ray with the detector plane.
Nothing is taken from the model except the omega it solved for, and that
omega is itself checked against the elastic condition ``|k_out| = |k_in|``.

Run with:
    cd packages/midas_diffract
    python -m pytest tests/test_wedge_positions.py -v
"""
import math

import pytest
import torch

from midas_diffract.forward import (
    HEDMForwardModel, HEDMGeometry, ScanConfig, TriVoxelConfig,
)

DEG2RAD = math.pi / 180.0
WL = 0.172979          # A
A_AU = 4.08            # A
BC = 1024.0            # exact in fp32 (model stores BC as fp32)
PX = 1.5               # um

EULERS = torch.tensor([
    [0.3, 0.5, 0.7],
    [1.9, 1.1, 4.2],
    [4.7, 2.3, 0.4],
], dtype=torch.float64)

# Off-axis voxels, with a z component too (FF grains are 3-D).
POSITIONS = torch.tensor([
    [650.0, -300.0, 40.0],
    [-500.0, 450.0, -25.0],
    [120.0, 700.0, 0.0],
], dtype=torch.float64)


def _hkls():
    ints = []
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if (h, k, l) == (0, 0, 0):
                    continue
                par = (h % 2, k % 2, l % 2)
                if par in ((0, 0, 0), (1, 1, 1)):
                    ints.append((h, k, l))
    ints = torch.tensor(ints, dtype=torch.float64)
    cart = ints / A_AU
    thetas = torch.asin(WL * torch.linalg.norm(cart, dim=1) / 2.0)
    return cart, thetas


def _model(wedge_deg, *, flip_y=False, Lsd=(5000.0, 7000.0), scan=None):
    hkls, thetas = _hkls()
    geom = HEDMGeometry(
        Lsd=list(Lsd), y_BC=[BC] * len(Lsd), z_BC=[BC] * len(Lsd), px=PX,
        omega_start=-180.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=WL,
        flip_y=flip_y, wedge=wedge_deg,
    )
    return HEDMForwardModel(hkls, thetas, geom, scan_config=scan)


def _bragg(model):
    hkls, thetas = _hkls()
    R = model.euler2mat(EULERS)
    om, eta, tt, valid = model.calc_bragg_geometry(R, hkls_cart=hkls, thetas=thetas)
    return R, hkls, om, eta, tt, valid


def _rodrigues(n, omega):
    """Rotation by ``omega`` (rad, right-handed) about unit axis ``n``."""
    K = torch.tensor([[0.0, -n[2], n[1]],
                      [n[2], 0.0, -n[0]],
                      [-n[1], n[0], 0.0]], dtype=torch.float64)
    I = torch.eye(3, dtype=torch.float64)
    return I + math.sin(omega) * K + (1.0 - math.cos(omega)) * (K @ K)


def _rigid_body_pixels(wedge_deg, R, hkls, om, valid, positions, Lsd, flip_y):
    """Independent ray trace. Returns {(k, m): (bragg_resid, [(y, z) per d], y_lab)}."""
    W = wedge_deg * DEG2RAD
    n_hat = (-math.sin(W), 0.0, math.cos(W))           # R_y(-W) e_z
    S = _rodrigues((0.0, 1.0, 0.0), -W)                # stage -> lab at omega = 0
    N = positions.shape[0]
    k_in = torch.tensor([1.0 / WL, 0.0, 0.0], dtype=torch.float64)
    out = {}
    for k, m in torch.nonzero(valid > 0.5, as_tuple=False).tolist():
        n = k % N
        Rn = _rodrigues(n_hat, float(om[k, m]))
        G_lab = Rn @ (S @ (R[n] @ hkls[m]))
        k_out = k_in + G_lab
        resid = abs(float(torch.linalg.norm(k_out)) - 1.0 / WL) * WL
        p_lab = Rn @ (S @ positions[n])
        pix = []
        for L in Lsd:
            t = (L - p_lab[0]) / k_out[0]
            y = p_lab[1] + t * k_out[1]
            z = p_lab[2] + t * k_out[2]
            ysgn = -1.0 if flip_y else 1.0
            pix.append((BC + ysgn * float(y) / PX, BC + float(z) / PX))
        out[(k, m)] = (resid, pix, float(p_lab[1]))
    return out


def _model_pixels(spots, D):
    yp, zp = spots.y_pixel, spots.z_pixel
    if D == 1 and yp.dim() == 2:
        yp, zp = yp.unsqueeze(0), zp.unsqueeze(0)
    return yp, zp


# ---------------------------------------------------------------------------
#  (a) W = 0: bit-identical to the pre-fix about-z rotation
# ---------------------------------------------------------------------------

def test_w0_projection_bit_identical_to_about_z_formula():
    m = _model(0.0)
    assert not m._wedge_active()
    R, hkls, om, eta, tt, valid = _bragg(m)
    spots = m.project_to_detector(om, eta, tt, POSITIONS, valid)

    # The pre-fix expressions, verbatim.
    pos2 = torch.cat([POSITIONS, POSITIONS], dim=-2)
    cw, sw = torch.cos(om), torch.sin(om)
    px_, py_, pz_ = (pos2[:, i].unsqueeze(-1) for i in range(3))
    x = px_ * cw - py_ * sw
    y = px_ * sw + py_ * cw
    z = pz_.expand_as(x)
    tan2, se, ce = torch.tan(tt), torch.sin(eta), torch.cos(eta)
    for d in range(m.n_distances):
        L = m._Lsd_eff.to(om.dtype)[d]
        dist = L - x
        ydet = y - dist * tan2 * se
        zdet = z + dist * tan2 * ce
        y_ref = m._y_BC.to(om.dtype)[d] + ydet / m.px
        z_ref = m._z_BC.to(om.dtype)[d] + zdet / m.px
        assert torch.equal(spots.y_pixel[d], y_ref)
        assert torch.equal(spots.z_pixel[d], z_ref)


def test_w0_refined_wedge_matches_fixed_and_carries_gradient():
    """A refined wedge sitting at 0 takes the tilted-axis branch (so its
    gradient flows through the positions) but must give the same numbers."""
    m_fix = _model(0.0)
    m_ref = _model(0.0)
    m_ref.wedge.requires_grad_(True)
    assert m_ref._wedge_active() and not m_fix._wedge_active()

    s_fix = m_fix(EULERS, POSITIONS)
    s_ref = m_ref(EULERS, POSITIONS)
    assert torch.equal(s_fix.y_pixel, s_ref.y_pixel.detach())
    assert torch.equal(s_fix.z_pixel, s_ref.z_pixel.detach())

    loss = (s_ref.z_pixel * s_ref.valid).sum()
    loss.backward()
    assert m_ref.wedge.grad is not None and torch.isfinite(m_ref.wedge.grad)


# ---------------------------------------------------------------------------
#  (b) W != 0: eager projection == independent rigid-body ray trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wedge_deg", [0.05, -0.8, 3.0])
@pytest.mark.parametrize("flip_y,Lsd", [(False, (5000.0, 7000.0)),
                                         (True, (1_000_000.0,))])
def test_wedge_projection_matches_rigid_body(wedge_deg, flip_y, Lsd):
    m = _model(wedge_deg, flip_y=flip_y, Lsd=Lsd)
    R, hkls, om, eta, tt, valid = _bragg(m)
    assert int((valid > 0.5).sum()) > 50
    spots = m.project_to_detector(om, eta, tt, POSITIONS, valid)
    yp, zp = _model_pixels(spots, len(Lsd))

    ref = _rigid_body_pixels(wedge_deg, R, hkls, om, valid, POSITIONS,
                             Lsd, flip_y)
    worst = 0.0
    for (k, mm), (resid, pix, _) in ref.items():
        # The omega the model solved for satisfies the elastic condition
        # about the TILTED axis -- so the reference is self-contained.
        assert resid < 1e-12, (k, mm, resid)
        for d, (y_r, z_r) in enumerate(pix):
            worst = max(worst, abs(float(yp[d, k, mm]) - y_r),
                        abs(float(zp[d, k, mm]) - z_r))
    assert worst < 1e-9, f"max |model - rigid body| = {worst:.3e} px"


def test_about_z_rotation_fails_the_rigid_body_reference():
    """Null control: the pre-fix position rotation (about untilted z, G still
    wedge-rotated) must FAIL the same reference, even at W = 0.05 deg --
    otherwise the test above could not detect the bug."""
    wedge_deg = 0.05
    m = _model(wedge_deg)
    m._has_wedge = False            # force the about-z position branch
    assert not m._wedge_active()
    R, hkls, om, eta, tt, valid = _bragg(m)
    spots = m.project_to_detector(om, eta, tt, POSITIONS, valid)
    ref = _rigid_body_pixels(wedge_deg, R, hkls, om, valid, POSITIONS,
                             (5000.0, 7000.0), False)
    worst = max(abs(float(spots.z_pixel[0, k, mm]) - pix[0][1])
                for (k, mm), (_, pix, _) in ref.items())
    # ~ |pos| sin W / px = 700 um * 8.7e-4 / 1.5 um ~ 0.4 px
    assert worst > 0.1, worst


def test_filter_by_scan_lab_y_is_wedge_free():
    """pf beam gate. The axis tilts about lab y, so the rigid-body lab y of a
    voxel is the no-wedge ``px sin(omega) + py cos(omega)`` for any W; the
    gate must follow it. Beams are centred on the rigid-body lab y with a
    2 um width, and a second set is offset by 3 um (must all be OUT)."""
    wedge_deg = 3.0
    Lsd = (5000.0, 7000.0)
    m0 = _model(wedge_deg, Lsd=Lsd)
    R, hkls, om, eta, tt, valid = _bragg(m0)
    on_det = m0.project_to_detector(om, eta, tt, POSITIONS, valid).valid
    ref = _rigid_body_pixels(wedge_deg, R, hkls, om, on_det, POSITIONS,
                             Lsd, False)
    keys = sorted(ref)[:: max(1, len(ref) // 6)][:6]
    assert len(keys) >= 3
    for (k, mm) in keys:
        n = k % POSITIONS.shape[0]
        w = float(om[k, mm])
        about_z = (float(POSITIONS[n, 0]) * math.sin(w)
                   + float(POSITIONS[n, 1]) * math.cos(w))
        assert abs(about_z - ref[(k, mm)][2]) < 1e-9
    for shift, want in ((0.0, 1.0), (3.0, 0.0)):
        beam_y = torch.tensor([ref[k][2] + shift for k in keys],
                              dtype=torch.float64)
        scan = ScanConfig(beam_positions=beam_y, beam_size=2.0)
        m = _model(wedge_deg, Lsd=Lsd, scan=scan)
        spots = m.project_to_detector(om, eta, tt, POSITIONS, valid)
        spots = m.filter_by_scan(spots, POSITIONS)
        for s, (k, mm) in enumerate(keys):
            assert spots.scan_mask[s, k, mm] == want, (shift, s, k, mm)


def test_nf_triangles_follow_tilted_axis():
    """forward_nf_triangles (the C-parity rasterising path): a sub-pixel
    voxel's hit at the first distance must land within 1 px of the
    rigid-body ray trace of its centre. At W = 3 deg the about-z rotation
    misses by tens of pixels, so this cannot pass by accident."""
    wedge_deg = 3.0
    Lsd = (5000.0, 7000.0)
    m = _model(wedge_deg, Lsd=Lsd)
    centers = POSITIONS[:, :2].clone()
    cfg = TriVoxelConfig(
        edge_lengths=torch.full((3,), 1.0, dtype=torch.float64),
        ud=torch.ones(3, dtype=torch.float64),
    )
    hits = m.forward_nf_triangles(EULERS, centers, cfg)
    assert len(hits) > 20

    R, hkls, om, eta, tt, valid = _bragg(m)
    pos3 = torch.nn.functional.pad(centers, (0, 1))
    ref = _rigid_body_pixels(wedge_deg, R, hkls, om, valid, pos3, Lsd, False)
    # Index the reference by (voxel, omega) -> first-distance pixel.
    by_vox_om = {}
    for (k, mm), (_, pix, _) in ref.items():
        by_vox_om.setdefault(k % 3, []).append(
            (float(om[k, mm]) * 180.0 / math.pi, pix[0]))
    n_checked = 0
    for vox, d, fr, y, z, ome in hits:
        if d != 0:
            continue
        # forward_nf_triangles uses the model's fp32 hkls, _bragg fp64:
        # match the omega loosely (spots of one voxel are >> 1e-3 deg apart).
        cands = [p for o, p in by_vox_om[vox] if abs(o - ome) < 1e-3]
        assert cands, (vox, ome)
        assert any(abs(y - math.floor(py)) <= 1 and abs(z - math.floor(pz)) <= 1
                   for py, pz in cands), (vox, ome, y, z, cands)
        n_checked += 1
    assert n_checked > 10
