"""Synthetic DAC-raster frames, with a known answer -- for the raster notebooks and their tests.

Nothing here reads a beamline convention or defaults to any one material. Every planted
reflection is placed with the SAME geometry primitives the real ingest chain and
:func:`midas_defect.domains.find_domains` consume (:func:`midas_defect.geometry.ewald_crossing_omegas`,
``qsample_to_qlab``, ``qlab_to_pixel``), so the synthetic frames exercise the identical code path a
real raster does end to end: mask, background, blob-finding, powder separation, indexing.

Not a physically exact forward model -- there is no structure factor, no absorption, no detector
point-spread beyond a placed Gaussian. It exists to give :mod:`midas_defect.raster` and its tests
a scan with a known answer, not to simulate a specific beamline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .geometry import (Geometry, detector_angle_maps, ewald_crossing_omegas,
                       qsample_to_qlab, qlab_to_pixel)

try:
    from midas_hkls import SpaceGroup, centring_allowed
except ImportError:  # pragma: no cover - midas_hkls is a required dependency in practice
    SpaceGroup = None
    centring_allowed = None

__all__ = ["PlantedReflection", "PositionTruth", "SyntheticRaster",
          "synthetic_position_frames", "synthetic_dac_raster"]


@dataclass
class PlantedReflection:
    """One planted spot: which domain, which hkl, and where it lands."""
    domain: int
    hkl: Tuple[int, int, int]
    row: float
    col: float
    frame: float                      # fractional frame index, stack-local


@dataclass
class PositionTruth:
    """The planted answer at one raster point."""
    U_list: List[np.ndarray]          # one orientation per domain, sample->crystal... see note
    labels: List[str]                 # e.g. "domain1", "domain2", "decoy" -- caller's own bookkeeping
    reflections: List[PlantedReflection] = field(default_factory=list)


def _b_matrix(a: float, b: float, c: float) -> np.ndarray:
    """q = 2*pi/d reciprocal basis for an orthorhombic/tetragonal DIAGONAL cell."""
    return np.diag([2 * math.pi / a, 2 * math.pi / b, 2 * math.pi / c])


def _hkl_candidates(hmax: int, kmax: int, lmax: int, space_group_number: Optional[int]) -> np.ndarray:
    h = np.arange(-hmax, hmax + 1)
    k = np.arange(-kmax, kmax + 1)
    l = np.arange(-lmax, lmax + 1)
    H, K, L = np.meshgrid(h, k, l, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=-1)
    hkl = hkl[np.any(hkl != 0, axis=1)]
    if space_group_number is not None and SpaceGroup is not None:
        sg = SpaceGroup.from_number(int(space_group_number))
        hkl = hkl[centring_allowed(hkl, sg)]
    return hkl


def synthetic_position_frames(
    U_list: Sequence[np.ndarray], *,
    a: float, b: float, c: float, space_group_number: int,
    geom: Geometry,
    hmax: int, kmax: int, lmax: int,
    domain_labels: Optional[Sequence[str]] = None,
    amplitude: float = 1500.0,
    pedestal: float = 150.0,
    gain: float = 1.0,
    blob_sigma_px: float = 1.6,
    blob_sigma_frames: float = 1.3,
    powder_two_theta_deg: Sequence[float] = (),
    powder_amplitude: float = 60.0,
    min_two_theta_deg: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, PositionTruth]:
    """Build one raster position's frame stack from a planted list of orientations.

    ``U_list`` are sample-frame orientation matrices (crystal -> sample, the same convention
    :func:`midas_defect.domains.find_domains` returns as ``Domain.U``): the reflection's
    sample-frame q is ``U @ B @ hkl``. ``a, b, c`` and ``space_group_number`` are required --
    there is no material default anywhere in this function, on purpose (see
    ``manuals/solve-cell/PACKAGE_NOTES.md`` on silent nickelate defaults elsewhere in this
    package).

    Returns ``(frames, truth)``: ``frames`` is ``(geom.n_frames, geom.n_pix_z, geom.n_pix_y)``
    float32, Poisson noise over a flat pedestal (and an optional powder ring, azimuthally
    uniform so it exercises :func:`midas_defect.ingest.detect_powder_rings`); ``truth`` records
    every reflection this function placed and where, so a caller can grade a reduction against
    a known answer with no real data at all.
    """
    rng = np.random.default_rng(seed)
    H, W = geom.n_pix_z, geom.n_pix_y
    n_frames = geom.n_frames
    omega_lo = math.radians(geom.omega_first_deg - 0.5 * geom.omega_step_deg)
    omega_hi = math.radians(geom.omega_first_deg + (n_frames - 0.5) * geom.omega_step_deg)
    if omega_lo > omega_hi:
        omega_lo, omega_hi = omega_hi, omega_lo

    B = _b_matrix(a, b, c)
    hkl_all = _hkl_candidates(hmax, kmax, lmax, space_group_number)
    labels = list(domain_labels) if domain_labels is not None else [f"domain{i}" for i in range(len(U_list))]

    clean = np.full((n_frames, H, W), float(pedestal), dtype=np.float64)
    reflections: List[PlantedReflection] = []

    for di, U in enumerate(U_list):
        U = np.asarray(U, dtype=np.float64)
        q_sample_all = (U @ B @ hkl_all.T).T                          # (n, 3)
        qmag = np.linalg.norm(q_sample_all, axis=1)
        keep_q = qmag > 1e-9
        for idx in np.flatnonzero(keep_q):
            hkl = hkl_all[idx]
            q_s = q_sample_all[idx]
            omegas = ewald_crossing_omegas(q_s, geom.wavelength_A)
            for w in omegas:
                w = float(w)
                # unwrap into whichever branch of (-pi, pi] + 2*pi*n falls in the swept range
                for n in (-1, 0, 1):
                    ww = w + 2 * math.pi * n
                    if not (omega_lo <= ww <= omega_hi):
                        continue
                    qlab = qsample_to_qlab(torch.as_tensor(q_s, dtype=torch.float64), ww)
                    try:
                        row, col = qlab_to_pixel(qlab.reshape(1, 3), geom, device="cpu")
                    except RuntimeError:
                        continue
                    row_f, col_f = float(row[0]), float(col[0])
                    if not (np.isfinite(row_f) and np.isfinite(col_f)):
                        continue
                    if not (0.0 <= row_f < H and 0.0 <= col_f < W):
                        continue
                    frame_f = (math.degrees(ww) - geom.omega_first_deg) / geom.omega_step_deg
                    if not (-0.5 <= frame_f <= n_frames - 0.5):
                        continue
                    two_theta = math.degrees(2.0 * math.asin(
                        min(1.0, max(-1.0, qmag[idx] * geom.wavelength_A / (4.0 * math.pi)))))
                    if two_theta < min_two_theta_deg:
                        continue
                    amp = amplitude / (1.0 + 0.35 * qmag[idx])
                    _add_gaussian_blob(clean, frame_f, row_f, col_f, amp,
                                      blob_sigma_frames, blob_sigma_px)
                    reflections.append(PlantedReflection(
                        domain=di, hkl=(int(hkl[0]), int(hkl[1]), int(hkl[2])),
                        row=row_f, col=col_f, frame=frame_f))

    if powder_two_theta_deg:
        tth, _az = detector_angle_maps(geom)
        tth = np.asarray(tth)
        ring = np.zeros((H, W), dtype=np.float64)
        for two_theta in powder_two_theta_deg:
            ring += powder_amplitude * np.exp(-0.5 * ((tth - two_theta) / 0.05) ** 2)
        clean += ring[None, :, :]

    frames = rng.poisson(np.clip(clean, 0.0, None) * gain) / gain
    truth = PositionTruth(U_list=[np.asarray(U, float) for U in U_list], labels=labels,
                          reflections=reflections)
    return frames.astype(np.float32), truth


def _add_gaussian_blob(stack: np.ndarray, f0: float, r0: float, c0: float,
                       amplitude: float, sig_f: float, sig_px: float) -> None:
    nF, H, W = stack.shape
    f_lo, f_hi = max(int(math.floor(f0 - 3 * sig_f)), 0), min(int(math.ceil(f0 + 3 * sig_f)) + 1, nF)
    r_lo, r_hi = max(int(math.floor(r0 - 3 * sig_px)), 0), min(int(math.ceil(r0 + 3 * sig_px)) + 1, H)
    c_lo, c_hi = max(int(math.floor(c0 - 3 * sig_px)), 0), min(int(math.ceil(c0 + 3 * sig_px)) + 1, W)
    if f_lo >= f_hi or r_lo >= r_hi or c_lo >= c_hi:
        return
    ff = np.arange(f_lo, f_hi)[:, None, None]
    rr = np.arange(r_lo, r_hi)[None, :, None]
    cc = np.arange(c_lo, c_hi)[None, None, :]
    g = amplitude * np.exp(-0.5 * (((ff - f0) / sig_f) ** 2 + ((rr - r0) / sig_px) ** 2
                                   + ((cc - c0) / sig_px) ** 2))
    stack[f_lo:f_hi, r_lo:r_hi, c_lo:c_hi] += g


@dataclass
class SyntheticRaster:
    """A synthetic raster: a callable ``loader(p)`` plus the planted answer at every point.

    ``geom`` is shared across every point (matches how a real raster's calibration works: one
    geometry per dataset). Frames are generated on demand, not stored, so a large raster does
    not have to fit in memory.
    """
    geom: Geometry
    a: float
    b: float
    c: float
    space_group_number: int
    n_points: int
    point_truth: List[PositionTruth]
    _build_kwargs: dict
    _seeds: List[int]

    def loader(self, p: int) -> np.ndarray:
        if not (0 <= p < self.n_points):
            raise IndexError(f"point {p} out of range [0, {self.n_points})")
        frames, truth = synthetic_position_frames(
            self.point_truth[p].U_list, a=self.a, b=self.b, c=self.c,
            space_group_number=self.space_group_number, geom=self.geom,
            domain_labels=self.point_truth[p].labels, seed=self._seeds[p],
            **self._build_kwargs)
        self.point_truth[p] = truth   # frozen the first time; reruns are identical (seeded)
        return frames


def synthetic_dac_raster(
    *, a: float, b: float, c: float, space_group_number: int,
    geom: Geometry, hmax: int, kmax: int, lmax: int,
    n_points: int,
    orientations_of_point: Callable[[int, np.random.Generator], Tuple[List[np.ndarray], List[str]]],
    seed: int = 0,
    **frame_kwargs,
) -> SyntheticRaster:
    """A whole synthetic raster: one call per point plants its own domain(s).

    ``orientations_of_point(p, rng) -> (U_list, labels)`` decides what is at point ``p`` --
    e.g. two fixed crystals at every point (to test the raster-wide "is this spread real"
    check against an IDENTICAL-crystal null), or an orientation that drifts smoothly across
    the raster. Nothing here plants a material; every one of ``a, b, c,
    space_group_number`` is required.
    """
    rng = np.random.default_rng(seed)
    point_truth: List[PositionTruth] = []
    seeds: List[int] = []
    for p in range(n_points):
        U_list, labels = orientations_of_point(p, rng)
        point_truth.append(PositionTruth(U_list=list(U_list), labels=list(labels)))
        seeds.append(int(rng.integers(0, 2**31 - 1)))
    return SyntheticRaster(geom=geom, a=a, b=b, c=c, space_group_number=space_group_number,
                           n_points=n_points, point_truth=point_truth,
                           _build_kwargs=dict(hmax=hmax, kmax=kmax, lmax=lmax, **frame_kwargs),
                           _seeds=seeds)
