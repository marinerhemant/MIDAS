"""External cross-check: the projector against scikit-image's Radon transform.

``implementation_plan.md`` Phase 1 asks for validation against an *independent
implementation*, not only against our own NumPy oracle (which shares this
package's conventions by construction). ``skimage.transform.radon`` is a widely
used, independently written parallel-beam projector, so agreement with it is
evidence about the geometry rather than about our own consistency.

Convention adapter (established empirically, 2026-08-03)
--------------------------------------------------------
The two libraries differ in two pure bookkeeping choices, both fixed here and
neither of them physics:

1. **Rotation centre.** ``skimage`` rotates about ``N/2``; a voxel grid centred
   the natural way sits at ``(N-1)/2``. Half a pixel. Left uncorrected it looks
   exactly like a geometry error -- the profiles agree perfectly at 0 deg and
   then drift apart with angle (up to 1.2 px of centroid by 131 deg). Placing the
   object on ``N/2`` makes the discrepancy **constant in angle**, which is what
   identifies it as a convention rather than a rotation error.
2. **Profile origin.** ``skimage``'s output abscissa is offset by half a pixel
   from ``(len-1)/2``.

With both applied the agreement is 4e-12 relative at 0 and 90 deg -- the angles
at which ``skimage`` performs no interpolation, so only float64 round-off is
left -- and ~0.2% RMS elsewhere,
which is its bilinear image-rotation blur, not a geometric difference. Our
projector forward-splats exact voxel positions and never rotates an image, so it
does not incur that blur.
"""
import numpy as np
import pytest
import torch

from midas_dct_tt import GrainShape, parallel_projection
from midas_dct_tt.conventions import dct_sample_rotation

radon = pytest.importorskip(
    "skimage.transform", reason="scikit-image not installed"
).radon

DT = torch.float64
N = 96              # object grid
NU = 160            # detector, big enough that nothing falls off
BEAM = torch.tensor([1.0, 0.0, 0.0], dtype=DT)     # project along lab x
SKIMAGE_PROFILE_OFFSET_PX = -0.5                   # see module docstring


def _object():
    """An asymmetric two-disc phantom -- symmetry would hide a sign error."""
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    c = (N - 1) / 2.0
    img = np.zeros((N, N))
    for ox, oy, r, a in ((-14.0, -6.0, 11.0, 1.0), (16.0, 12.0, 6.0, 0.6)):
        img += a / (1 + np.exp((np.hypot(xx - c - ox, yy - c - oy) - r) / 0.8))
    return img


def _grain(img):
    """One z-layer grain whose occupancy is the phantom.

    Lab ``x`` maps to image axis 0 and lab ``y`` to axis 1, because
    ``skimage.radon`` at ``theta = 0`` sums along axis 0. The object is centred on
    ``N/2`` to match skimage's rotation centre.
    """
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    centre = N / 2.0
    X = (yy - centre).ravel()
    Y = (xx - centre).ravel()
    pos = torch.tensor(np.stack([X, Y, np.zeros_like(X)], -1), dtype=DT)
    chi = torch.tensor(img.ravel(), dtype=DT).clamp(1e-12, 1 - 1e-12)
    return GrainShape(positions=pos, logits=torch.log(chi / (1 - chi)), spacing_um=1.0)


def _profiles(img, grain, angle_deg):
    """Our projection profile and skimage's, on a common abscissa."""
    R = dct_sample_rotation(angle_deg, omega_sign=-1.0, dtype=DT)
    im = parallel_projection(
        grain.positions @ R.T, grain.occupancy, normal=BEAM,
        voxel_volume_um3=1.0, pixel_um=1.0, detector_shape=(NU, NU),
    )
    # Sum over the detector's v axis: the single z-layer straddles two pixel rows
    # (it lands at v = 79.5), so one row holds exactly half the mass.
    mine = im.sum(dim=1).numpy()
    sk = radon(img, theta=[angle_deg], circle=False)[:, 0]
    m_c = np.arange(NU) - (NU - 1) / 2.0
    s_c = np.arange(len(sk)) - (len(sk) - 1) / 2.0 + SKIMAGE_PROFILE_OFFSET_PX
    return mine, np.interp(m_c, s_c, sk), sk


@pytest.mark.unit
@pytest.mark.parametrize("angle", (0.0, 90.0))
def test_matches_skimage_to_roundoff_where_it_does_not_interpolate(angle):
    """At axis-aligned angles skimage does no interpolation, so only round-off remains.

    Measured 4.4e-12 relative -- float64 accumulation through skimage's warp, not
    a modelling difference. Two independently written projectors agreeing at that
    level is the strongest statement available about the geometry.
    """
    img = _object()
    mine, ref, _ = _profiles(img, _grain(img), angle)
    assert np.abs(mine - ref).max() / ref.max() < 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("angle", (23.0, 47.0, 131.0, 217.0))
def test_matches_skimage_within_its_interpolation_blur(angle):
    """Elsewhere the residual is skimage's bilinear image rotation, ~0.2% RMS."""
    img = _object()
    mine, ref, _ = _profiles(img, _grain(img), angle)
    rms = np.sqrt(np.mean((mine - ref) ** 2)) / ref.max()
    assert rms < 0.005


@pytest.mark.unit
@pytest.mark.parametrize("angle", (0.0, 23.0, 47.0, 90.0, 131.0, 217.0))
def test_total_projected_mass_matches_skimage(angle):
    """Mass is interpolation-independent, so it must agree far more tightly."""
    img = _object()
    mine, _, sk = _profiles(img, _grain(img), angle)
    assert abs(mine.sum() / sk.sum() - 1.0) < 1e-4


@pytest.mark.unit
def test_discrepancy_is_constant_in_angle_not_growing():
    """The test that separates 'convention' from 'wrong rotation'.

    A genuine rotation error grows with angle. A centre-convention offset does
    not. After the adapter, our centroid sits a fixed 0.5 px from skimage's at
    every angle -- flat, which is the signature of the former being absent.
    """
    img = _object()
    grain = _grain(img)
    diffs = []
    for angle in (0.0, 23.0, 47.0, 90.0, 131.0, 217.0):
        R = dct_sample_rotation(angle, omega_sign=-1.0, dtype=DT)
        im = parallel_projection(
            grain.positions @ R.T, grain.occupancy, normal=BEAM,
            voxel_volume_um3=1.0, pixel_um=1.0, detector_shape=(NU, NU),
        )
        mine = im.sum(dim=1).numpy()
        sk = radon(img, theta=[angle], circle=False)[:, 0]
        cm = (mine * (np.arange(NU) - (NU - 1) / 2.0)).sum() / mine.sum()
        cs = (sk * (np.arange(len(sk)) - (len(sk) - 1) / 2.0)).sum() / sk.sum()
        diffs.append(cm - cs)
    assert max(diffs) - min(diffs) < 0.01        # flat to 1/100 pixel
