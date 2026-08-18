"""ESRF DCT interoperability.

The centrepiece is :func:`test_esrf_rotation_matches_midas`: the mapping from
ESRF's ``rotdir`` vector to our scalar omega sign is only valid if the two codes
build the *same* rotation from an (axis, angle) pair. That is asserted here
against a direct transcription of the MATLAB rather than taken on trust, because
a handedness mismatch would mirror every reconstruction with no other symptom.
"""
import math

import pytest
import torch
from midas_dfxm.conventions import rotation_matrix

from midas_dct_tt.conventions import DCT_OMEGA_SIGN_AERO, DCT_OMEGA_SIGN_CCW
from midas_dct_tt.esrf import (
    ESRF_LENGTH_UNIT_UM,
    ESRFGeometry,
    esrf_center_px,
    esrf_default_detgeo,
    esrf_detector,
    esrf_detector_basis,
    esrf_omega_sign,
    mm_to_um,
)
from midas_dct_tt.project import default_center_px


# --- reference transcription of the ESRF MATLAB ----------------------------
def gt_maths_rotation_tensor(om_deg, n):
    """Transcription of ``gtMathsRotationTensor`` + ``gtMathsRotationMatrixComp``.

    From ``src/dct/zUtil_Maths/``, ``rc = 'col'`` branch::

        const = n n^T
        cos   = I - n n^T
        sin   = [[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]]
        Srot  = const + cos*cosd(om) + sin*sind(om)
    """
    n = torch.as_tensor(n, dtype=torch.float64).reshape(3)
    n = n / torch.linalg.vector_norm(n)
    n1, n2, n3 = (float(x) for x in n)
    const = torch.outer(n, n)
    cosm = torch.eye(3, dtype=torch.float64) - const
    sinm = torch.tensor([[0.0, -n3, n2], [n3, 0.0, -n1], [-n2, n1, 0.0]],
                        dtype=torch.float64)
    c = math.cos(math.radians(om_deg))
    s = math.sin(math.radians(om_deg))
    return const + cosm * c + sinm * s


def test_esrf_rotation_matches_midas():
    """ESRF and MIDAS build the same rotation -- so ``sign = rotdir_z`` is valid.

    If this ever fails by a transpose, the ``rotdir`` -> omega-sign mapping in
    :func:`esrf_omega_sign` is inverted and every converted DCT scan is mirrored.
    """
    axes = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 3)]
    angles = [-137.0, -45.0, -1.0, 0.0, 7.5, 30.0, 90.0, 179.0]
    worst = 0.0
    for ax in axes:
        a = torch.as_tensor(ax, dtype=torch.float64)
        a = a / torch.linalg.vector_norm(a)
        for om in angles:
            ours = rotation_matrix(a, torch.as_tensor(om, dtype=torch.float64))
            theirs = gt_maths_rotation_tensor(om, a)
            worst = max(worst, float(torch.max(torch.abs(ours.to(torch.float64) - theirs))))
    assert worst < 1e-12, f"ESRF/MIDAS rotation mismatch {worst:.3e}"


def test_esrf_rotation_is_right_handed():
    """Sanity anchor: +90 deg about +z takes +x to +y in both codes."""
    r = gt_maths_rotation_tensor(90.0, (0.0, 0.0, 1.0))
    got = r @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(got, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12)


# --- units -----------------------------------------------------------------
def test_mm_to_um_scalar_tuple_tensor():
    assert mm_to_um(1.0) == 1000.0
    assert mm_to_um((1.0, 2.0)) == (1000.0, 2000.0)
    assert torch.allclose(mm_to_um(torch.tensor([1.0])), torch.tensor([1000.0]))
    assert ESRF_LENGTH_UNIT_UM == 1000.0


# --- the one-pixel trap ----------------------------------------------------
@pytest.mark.parametrize("n", [16, 64, 255, 256, 2048])
def test_esrf_default_reference_pixel_becomes_midas_default_centre(n):
    """ESRF's ``N/2 + 0.5`` (1-based) is exactly our ``(N-1)/2`` (0-based).

    This is the whole justification for the ``-1``: both codes mean 'the centre
    of the detector', they just index from different origins.
    """
    cu, cv = esrf_center_px(n / 2.0 + 0.5, n / 2.0 + 0.5)
    du, dv = default_center_px((n, n))
    assert cu == pytest.approx(du, abs=1e-12)
    assert cv == pytest.approx(dv, abs=1e-12)


def test_esrf_center_px_is_a_unit_shift():
    assert esrf_center_px(1.0, 1.0) == (0.0, 0.0)   # MATLAB pixel 1 = Python pixel 0


# --- omega sign ------------------------------------------------------------
def test_omega_sign_from_rotdir():
    assert esrf_omega_sign((0, 0, 1)) == DCT_OMEGA_SIGN_CCW
    assert esrf_omega_sign((0, 0, -1)) == DCT_OMEGA_SIGN_AERO
    assert esrf_omega_sign((0, 0, 5.0)) == DCT_OMEGA_SIGN_CCW      # unnormalised


def test_omega_sign_reproduces_esrf_stage_table():
    """``gtGeoLabDefaultParameters.m``: 'pmo' is [0 0 -1], the rest [0 0 +1].

    'pmo' therefore turns the same way as the 1-ID aero stage.
    """
    assert esrf_omega_sign((0, 0, -1)) == DCT_OMEGA_SIGN_AERO      # pmo
    for _ in ("diffrz", "omega", "mrsrot", "srot"):
        assert esrf_omega_sign((0, 0, 1)) == DCT_OMEGA_SIGN_CCW


def test_omega_sign_refuses_tilted_axis():
    with pytest.raises(ValueError, match="not parallel to lab z"):
        esrf_omega_sign((0.1, 0.0, 1.0))
    with pytest.raises(ValueError, match="zero vector"):
        esrf_omega_sign((0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="3 components"):
        esrf_omega_sign((0.0, 1.0))


# --- detector basis --------------------------------------------------------
def test_detector_basis_inline_default():
    """Inline default u=+y, v=-z: the OUTWARD normal is +x, i.e. cross(v,u).

    ``cross(u, v)`` would give -x, putting the detector behind the sample.
    """
    u, v, n = esrf_detector_basis((0, 1, 0), (0, 0, -1))
    assert torch.allclose(n, torch.linalg.cross(torch.as_tensor(v), torch.as_tensor(u)), atol=1e-12)
    assert torch.allclose(n, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    assert abs(float(torch.sum(u * v))) < 1e-12


def test_detector_basis_vertical_branch_also_points_outward():
    """The other ESRF branch: u=-z, v=+x, detrefpos=[0,dist,0]. Normal must be +y."""
    _, _, n = esrf_detector_basis((0, 0, -1), (1, 0, 0))
    assert torch.allclose(n, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12)


def test_detector_rejects_inward_normal():
    """Swapping detdiru/detdirv inverts the normal -- caught, not silently mirrored."""
    dg = esrf_default_detgeo(64, 64, pixelsize_mm=1e-3, dist_mm=10.0)
    dg["detdiru"], dg["detdirv"] = dg["detdirv"], dg["detdiru"]
    with pytest.raises(ValueError, match="points away from the detector"):
        esrf_detector(dg)


def test_detector_basis_rejects_non_orthogonal():
    with pytest.raises(ValueError, match="not orthogonal"):
        esrf_detector_basis((0, 1, 0), (0, 1, 1))


def test_detector_basis_normalises():
    u, v, n = esrf_detector_basis((0, 3, 0), (0, 0, -7))
    for w in (u, v, n):
        assert float(torch.linalg.vector_norm(w)) == pytest.approx(1.0, abs=1e-12)


# --- full detector conversion ---------------------------------------------
def test_esrf_detector_conversion_units_and_centre():
    dg = esrf_default_detgeo(2048, 2048, pixelsize_mm=1.4e-3, dist_mm=12.5)
    det, normal = esrf_detector(dg)
    assert det.pixel_um == pytest.approx(1.4, abs=1e-12)          # mm -> um
    assert det.shape == (2048, 2048)
    assert det.distance_um == pytest.approx(12500.0, abs=1e-9)    # mm -> um
    assert det.center_px == pytest.approx(default_center_px((2048, 2048)), abs=1e-12)
    assert torch.allclose(normal, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12)


def test_distance_is_projection_onto_normal_not_norm():
    """A laterally offset detector: ``|detrefpos|`` would overstate the distance."""
    dg = esrf_default_detgeo(64, 64, pixelsize_mm=1e-3, dist_mm=10.0)
    dg["detrefpos"] = (10.0, 3.0, 4.0)          # 5 mm off-axis, normal is +x
    det, _ = esrf_detector(dg)
    assert det.distance_um == pytest.approx(10000.0, abs=1e-6)
    assert det.distance_um != pytest.approx(math.hypot(10.0, 5.0) * 1000.0, abs=1.0)


def test_esrf_detector_rejects_anisotropic_pixels():
    dg = esrf_default_detgeo(64, 64, pixelsize_mm=1e-3, dist_mm=10.0)
    dg["pixelsizev"] = 2e-3
    with pytest.raises(ValueError, match="anisotropic"):
        esrf_detector(dg)


def test_esrf_detector_reports_missing_fields():
    dg = esrf_default_detgeo(64, 64, pixelsize_mm=1e-3, dist_mm=10.0)
    del dg["detdiru"]
    with pytest.raises(KeyError, match="detdiru"):
        esrf_detector(dg)


# --- geometry container ----------------------------------------------------
def test_geometry_accessors():
    geo = ESRFGeometry(
        labgeo={"beamdir": (1, 0, 0), "rotdir": (0, 0, -1), "labunit": "mm"},
        detgeo=esrf_default_detgeo(512, 512, pixelsize_mm=2e-3, dist_mm=8.0),
    )
    geo.check_frame()
    assert geo.omega_sign() == DCT_OMEGA_SIGN_AERO
    det, n = geo.detector()
    assert det.pixel_um == pytest.approx(2.0)
    assert det.distance_um == pytest.approx(8000.0)


def test_check_frame_rejects_foreign_beam_direction():
    geo = ESRFGeometry(labgeo={"beamdir": (0, 1, 0)})
    with pytest.raises(ValueError, match=r"not \[1,0,0\]"):
        geo.check_frame()


def test_check_frame_passes_unnormalised_beamdir():
    ESRFGeometry(labgeo={"beamdir": (3.0, 0.0, 0.0)}).check_frame()


def test_missing_rotdir_is_explicit():
    with pytest.raises(KeyError, match="rotdir"):
        ESRFGeometry(labgeo={"beamdir": (1, 0, 0)}).omega_sign()
