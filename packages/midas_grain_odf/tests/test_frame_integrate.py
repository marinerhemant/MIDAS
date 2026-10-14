"""The splatter's ``frame_integrate`` option integrates the omega profile over
each frame's exposure instead of sampling it at the frame centre.

Known answers: a spot whose omega width is far below a frame lands entirely in
the frame that contains it (a sampled kernel splits it, or loses it, depending
on where it sits); total intensity matches the sampled kernel for a wide spot;
the default path is unchanged; and the gradient with respect to the spot's
frame coordinate is alive for a sharp spot near a frame edge.
"""
import math

import torch

from midas_grain_odf.spot_extract import SpotPatchSpec, splat_spots_to_patches_sparse


def _spec(sigma_f, F=7, P=9):
    z = torch.zeros(1, dtype=torch.float64)
    return SpotPatchSpec(n_spots=1, patch_F=F, patch_P=P, sigma_yz=1.0,
                         sigma_f=sigma_f, anchor_y=z, anchor_z=z, anchor_f=z.clone())


def _splat(spec, f, integrate, rf=3):
    t = lambda v: torch.tensor([[v]], dtype=torch.float64)
    return splat_spots_to_patches_sparse(
        spec, t(0.0), t(0.0), t(f), torch.ones(1, dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64), radius_yz=4, radius_f=rf,
        frame_integrate=integrate)[0]


def test_sharp_spot_lands_in_its_own_frame():
    spec = _spec(sigma_f=0.01)   # edges >= 5 sigma away for every f below
    for f in (-0.45, -0.2, 0.0, 0.3, 0.45):        # all inside the centre cell [-0.5, 0.5)
        per_frame = _splat(spec, f, True).sum(dim=(1, 2))
        assert per_frame[3] / per_frame.sum() > 0.999, f
    # just past the edge it moves to the next frame
    per_frame = _splat(spec, 0.55, True).sum(dim=(1, 2))
    assert per_frame[4] / per_frame.sum() > 0.99


def test_total_matches_sampled_kernel_for_a_wide_spot():
    spec = _spec(sigma_f=1.2, F=15)
    a = _splat(spec, 0.3, True, rf=7).sum()
    b = _splat(spec, 0.3, False, rf=7).sum()
    assert abs(float(a / b) - 1.0) < 2e-3


def test_default_path_unchanged():
    spec = _spec(sigma_f=0.7)
    a = _splat(spec, 0.25, False)
    loc = torch.arange(7, dtype=torch.float64) - 3
    expect_f = torch.exp(-0.5 * ((loc - 0.25) / 0.7) ** 2)
    got_f = a.sum(dim=(1, 2)) / a.sum(dim=(1, 2)).max()
    assert torch.allclose(got_f, expect_f / expect_f.max(), atol=1e-12)


def test_gradient_alive_for_sharp_spot_near_a_frame_edge():
    spec = _spec(sigma_f=0.15)
    f = torch.tensor([[0.45]], dtype=torch.float64, requires_grad=True)
    z = torch.zeros(1, 1, dtype=torch.float64)
    out = splat_spots_to_patches_sparse(spec, z, z, f, torch.ones(1, dtype=torch.float64),
                                        torch.ones(1, 1, dtype=torch.float64), radius_yz=4,
                                        radius_f=3, frame_integrate=True)[0]
    out.sum(dim=(1, 2))[4].backward()           # intensity leaking into the next frame
    assert f.grad.abs().item() > 0.1
    assert math.isfinite(f.grad.item())


def test_pixel_integrate_conserves_total_and_centre():
    spec = _spec(sigma_f=0.7, F=7, P=15)
    t = lambda v: torch.tensor([[v]], dtype=torch.float64)
    kw = dict(radius_yz=7, radius_f=3)
    one = torch.ones(1, dtype=torch.float64); v = torch.ones(1, 1, dtype=torch.float64)
    a = splat_spots_to_patches_sparse(spec, t(0.3), t(-0.2), t(0.0), one, v, pixel_integrate=True, **kw)[0]
    b = splat_spots_to_patches_sparse(spec, t(0.3), t(-0.2), t(0.0), one, v, **kw)[0]
    assert abs(float(a.sum() / b.sum()) - 1.0) < 2e-3
    yy = torch.arange(15, dtype=torch.float64) - 7
    img = a.sum(0)
    cy = float((img.sum(1) * yy).sum() / img.sum()); cz = float((img.sum(0) * yy).sum() / img.sum())
    assert abs(cy - 0.3) < 1e-3 and abs(cz + 0.2) < 1e-3
    # a pixel-integrated narrow PSF is wider than the centre-sampled one
    va = float((img.sum(1) * (yy - cy) ** 2).sum() / img.sum())
    imb = b.sum(0); vb = float((imb.sum(1) * (yy - 0.3) ** 2).sum() / imb.sum())
    assert abs(va - (vb + 1.0 / 12.0)) < 0.02
