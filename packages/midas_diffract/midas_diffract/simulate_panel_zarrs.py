"""Multi-panel forward simulation that writes per-panel ``.MIDAS.zip`` zarrs.

Replaces ``ForwardSimulationCompressed`` for true multi-detector pinwheel
layouts where the C tool's tilt convention does not project beam-axis
rotations correctly. Drives :class:`HEDMForwardModel` in
``multi_mode="panel"``, then rasterises each predicted spot into a 2-D
Gaussian on the panel where it landed and writes one zarr per panel with
the same on-disk layout the rest of the FF-HEDM pipeline expects.

Output per panel
----------------
``{out_dir}/{out_stem}_det{N}.analysis.MIDAS.zip`` containing:

  * ``/exchange/data``           — ``(n_frames, n_pixels, n_pixels)`` int32 image stack
  * ``/measurement/process/scan_parameters/{datatype, start, step}``
  * ``/analysis/process/analysis_parameters/{Lsd, YCen, ZCen, tx, ty, tz,
    p0..p4, RingThresh, NrPixels, PixelSize, Wavelength, OmegaStart,
    OmegaEnd, Wedge, ...}`` populated from the per-panel ``DetectorGeom``.

Plus ``GrainsSim.csv`` (truth orientations / positions / strains in the
canonical MIDAS Au-FCC format) and ``setup_truth.json`` (full truth
record + per-panel geometry the validation script can compare against).

Usage
-----
``simulate_panel_zarrs`` is the library entry point. The
``midas-diffract-simulate-panels`` CLI is the user-facing wrapper.
``midas_ff_pipeline.testing.generate_pinwheel_synthetic_dataset`` calls
the library function for the FF-pipeline's hydra-style smoke.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from .forward import HEDMForwardModel, HEDMGeometry

DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
#  Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PanelGeom:
    """One panel's geometry (multi-detector FF layout).

    Field names match :class:`midas_ff_pipeline.detector.DetectorConfig`
    so a list of ``DetectorConfig`` rows can be passed straight in.
    """

    det_id: int
    lsd: float                            # sample-to-detector distance (μm)
    y_bc: float                           # beam-center column (px)
    z_bc: float                           # beam-center row (px)
    tx: float = 0.0                       # tilt about beam axis (deg)
    ty: float = 0.0                       # tilt (deg)
    tz: float = 0.0                       # tilt (deg)
    p_distortion: List[float] = field(default_factory=lambda: [0.0] * 11)


@dataclass
class SimConfig:
    """Static scan + crystal config shared across all panels."""

    space_group: int
    lattice_a: float                      # Å (cubic for the FF-pipeline path)
    wavelength_A: float                   # Å
    n_pixels: int                         # square panel side (px)
    px_um: float                          # pixel size (μm)
    omega_start_deg: float
    omega_end_deg: float
    omega_step_deg: float                 # signed; n_frames = round(|end-start|/|step|)
    rings_max: int                        # bound on ring number to enumerate
    wedge_deg: float = 0.0
    rho_d: float = 2_000_000.0
    # Sample-volume bounds for grain-position sampling (μm)
    rsample_um: float = 200.0
    hbeam_um: float = 200.0
    # Per-spot intensity / shape for the rasteriser
    peak_intensity: float = 5000.0        # peak counts (single pixel) per spot
    gauss_sigma_px: float = 1.0           # 2D in-plane spot σ (px)
    # Frame-axis behaviour: ``omega_sigma_frames > 0`` spreads each spot
    # across ±3σ frames with Gaussian weights; ``= 0`` puts the spot
    # entirely in floor(frame_nr) (matches ForwardSimulationCompressed
    # which also deposits one spot per frame).
    omega_sigma_frames: float = 0.0
    pos_noise_px: float = 0.0             # extra Gaussian noise on (y, z) per spot
    # ring threshold for analysis_parameters/RingThresh (matches FF defaults)
    ring_thresh: int = 10


# ---------------------------------------------------------------------------
#  HKL helpers
# ---------------------------------------------------------------------------

def _build_au_fcc_hkls(
    lattice_a: float,
    wavelength_A: float,
    rings_max: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Au / FCC fully-expanded Miller indices out to ``rings_max`` rings.

    Returns
    -------
    cart : (M, 3) torch float64           — Cartesian G-vectors (1/Å)
    thetas : (M,) torch float64           — Bragg angles (rad)
    hkls_int : (M, 3) torch float64       — integer (h, k, l) per spot row
    ring_id : (M,) np.int32               — ring number per row, 1-based
    """
    families: list[tuple[int, int, int]] = []
    for h in range(-8, 9):
        for k in range(-8, 9):
            for ll in range(-8, 9):
                if h == 0 and k == 0 and ll == 0:
                    continue
                # FCC selection rule: all even or all odd
                if not ((h % 2 == k % 2 == ll % 2 == 0)
                        or (abs(h) % 2 == abs(k) % 2 == abs(ll) % 2 == 1)):
                    continue
                families.append((h, k, ll))
    int_hkls = np.array(families, dtype=np.float64)
    msq = (int_hkls ** 2).sum(axis=1)
    unique_m = np.unique(msq)
    unique_m.sort()
    chosen_m = unique_m[:rings_max]
    keep = np.isin(msq, chosen_m)
    int_hkls = int_hkls[keep]
    msq = msq[keep]
    order = np.lexsort(
        (int_hkls[:, 2], int_hkls[:, 1], int_hkls[:, 0], msq)
    )
    int_hkls = int_hkls[order]
    msq = msq[order]
    ring_id = np.searchsorted(chosen_m, msq) + 1

    cart = int_hkls / lattice_a              # B = (1/a) * I for cubic
    g_norm = np.linalg.norm(cart, axis=1)
    sin_th = wavelength_A * g_norm / 2.0
    valid = sin_th < 0.99
    if not valid.all():
        cart = cart[valid]
        int_hkls = int_hkls[valid]
        ring_id = ring_id[valid]
        sin_th = sin_th[valid]
    thetas = np.arcsin(sin_th)
    return (
        torch.tensor(cart, dtype=torch.float64),
        torch.tensor(thetas, dtype=torch.float64),
        torch.tensor(int_hkls, dtype=torch.float64),
        ring_id.astype(np.int32),
    )


# ---------------------------------------------------------------------------
#  Random grain population
# ---------------------------------------------------------------------------

def _sample_grains(
    n: int,
    rsample_um: float,
    hbeam_um: float,
    strain_mag: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-on-SO(3) Euler ZXZ + uniform position + deviatoric Voigt strain."""
    u1, u2, u3 = rng.uniform(0, 1, size=(3, n))
    phi1 = 2 * np.pi * u1
    Phi = np.arccos(1 - 2 * u2)
    phi2 = 2 * np.pi * u3
    eulers = np.stack([phi1, Phi, phi2], axis=-1)
    positions = np.stack([
        rng.uniform(-rsample_um, rsample_um, n),
        rng.uniform(-rsample_um, rsample_um, n),
        rng.uniform(-hbeam_um / 2.0, hbeam_um / 2.0, n),
    ], axis=-1)
    raw = rng.normal(0, 1, size=(n, 6))
    raw[:, [0, 3, 5]] -= raw[:, [0, 3, 5]].mean(axis=1, keepdims=True)
    norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
    target = rng.uniform(0, strain_mag, size=(n, 1))
    strains = raw / norms * target
    return eulers.astype(np.float64), positions.astype(np.float64), strains.astype(np.float64)


# ---------------------------------------------------------------------------
#  Euler -> orientation matrix (ZXZ convention matching MIDAS/C)
# ---------------------------------------------------------------------------

def _euler_zxz_to_om(eulers_rad: np.ndarray) -> np.ndarray:
    """``(N, 3)`` Euler ZXZ in radians → ``(N, 9)`` row-major orientation matrices."""
    phi1, Phi, phi2 = eulers_rad[:, 0], eulers_rad[:, 1], eulers_rad[:, 2]
    c1, s1 = np.cos(phi1), np.sin(phi1)
    cP, sP = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    om = np.zeros((eulers_rad.shape[0], 3, 3), dtype=np.float64)
    om[:, 0, 0] = c1 * c2 - s1 * cP * s2
    om[:, 0, 1] = -c1 * s2 - s1 * cP * c2
    om[:, 0, 2] = s1 * sP
    om[:, 1, 0] = s1 * c2 + c1 * cP * s2
    om[:, 1, 1] = -s1 * s2 + c1 * cP * c2
    om[:, 1, 2] = -c1 * sP
    om[:, 2, 0] = sP * s2
    om[:, 2, 1] = sP * c2
    om[:, 2, 2] = cP
    return om.reshape(-1, 9)


# ---------------------------------------------------------------------------
#  Per-frame Gaussian rasteriser (vectorised on GPU)
# ---------------------------------------------------------------------------

def _render_frame(
    img: np.ndarray,
    spots_y: np.ndarray,
    spots_z: np.ndarray,
    spots_amp: np.ndarray,
    sigma: float,
    n_pixels: int,
) -> None:
    """Add an int32 Gaussian for each spot into the ``(n_pixels, n_pixels)`` image.

    All inputs are CPU numpy arrays. Vectorised across spots; each spot
    contributes a ``(2K+1) x (2K+1)`` kernel where ``K = ceil(3*sigma)``.
    """
    if spots_y.size == 0:
        return
    K = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-K, K + 1, dtype=np.int64)             # (2K+1,)
    n_spots = spots_y.shape[0]
    yi_floor = np.floor(spots_y).astype(np.int64)              # (n_spots,)
    zi_floor = np.floor(spots_z).astype(np.int64)
    dy = (offsets[None, :, None].astype(np.float64) +
          yi_floor[:, None, None] - spots_y[:, None, None])    # (n_spots, K2, 1)
    dz = (offsets[None, None, :].astype(np.float64) +
          zi_floor[:, None, None] - spots_z[:, None, None])    # (n_spots, 1, K2)
    gauss = np.exp(-(dy ** 2 + dz ** 2) / (2.0 * sigma ** 2))  # (n_spots, K2, K2)
    contrib = (gauss * spots_amp[:, None, None]).astype(np.int64)
    yy = yi_floor[:, None, None] + offsets[None, :, None]      # (n_spots, K2, 1)
    zz = zi_floor[:, None, None] + offsets[None, None, :]      # (n_spots, 1, K2)
    yy_b = np.broadcast_to(yy, contrib.shape)
    zz_b = np.broadcast_to(zz, contrib.shape)
    in_bounds = (yy_b >= 0) & (yy_b < n_pixels) & (zz_b >= 0) & (zz_b < n_pixels)
    if not in_bounds.any():
        return
    flat_y = yy_b[in_bounds]
    flat_z = zz_b[in_bounds]
    flat_v = contrib[in_bounds]
    # Note: ``np.add.at(img, (flat_z, flat_y), ...)`` below — see comment.
    # On-disk zarr layout is (Z, Y) — peakfit transposes after read so its
    # analysis frame is (Y, Z). To make peakfit's good_coords mask match the
    # rasterised spots, write to img[z_pix, y_pix] (NOT img[y_pix, z_pix]).
    np.add.at(img, (flat_z, flat_y), flat_v.astype(img.dtype))


# ---------------------------------------------------------------------------
#  Zarr writer
# ---------------------------------------------------------------------------

def _write_panel_zarr(
    zip_path: Path,
    image_stack: Iterable[np.ndarray],     # iterable of (n_pixels, n_pixels) per frame
    cfg: SimConfig,
    panel: PanelGeom,
    n_frames: int,
) -> None:
    """Write a per-panel ``.MIDAS.zip`` zarr."""
    import zarr
    import numcodecs

    if zip_path.exists():
        zip_path.unlink()
    with zarr.ZipStore(str(zip_path), mode="w") as store:
        root = zarr.group(store=store, overwrite=True)
        # exchange/data
        ds = root.create_dataset(
            "exchange/data",
            shape=(n_frames, cfg.n_pixels, cfg.n_pixels),
            dtype=np.int32,
            chunks=(1, cfg.n_pixels, cfg.n_pixels),
            compressor=numcodecs.Blosc(cname="zstd", clevel=3),
        )
        for f, frame in enumerate(image_stack):
            ds[f, :, :] = frame.astype(np.int32, copy=False)

        # measurement/process/scan_parameters
        sp_meas = root.require_group("measurement/process/scan_parameters")
        sp_meas.create_dataset("datatype",
                               data=np.bytes_(b"int32"))
        sp_meas.create_dataset("start", data=np.array([cfg.omega_start_deg]))
        sp_meas.create_dataset("step", data=np.array([cfg.omega_step_deg]))

        # analysis/process/analysis_parameters
        ap = root.require_group("analysis/process/analysis_parameters")
        def _w(name: str, value, dtype=np.float64):
            ap.create_dataset(name, data=np.array(value, dtype=dtype))

        _w("Lsd", [panel.lsd])
        _w("YCen", [panel.y_bc])
        _w("ZCen", [panel.z_bc])
        _w("tx", [panel.tx])
        _w("ty", [panel.ty])
        _w("tz", [panel.tz])
        for i, pv in enumerate(panel.p_distortion[:11]):
            _w(f"p{i}", [pv])
        _w("Wavelength", [cfg.wavelength_A])
        _w("OmegaStart", [cfg.omega_start_deg])
        _w("OmegaEnd", [cfg.omega_end_deg])
        _w("OmegaStep", [cfg.omega_step_deg])
        _w("PixelSize", [cfg.px_um])
        _w("NrPixels", [cfg.n_pixels], dtype=np.int64)
        _w("Wedge", [cfg.wedge_deg])
        _w("RhoD", [cfg.rho_d])
        _w("Rsample", [cfg.rsample_um])
        _w("Hbeam", [cfg.hbeam_um])
        _w("BeamThickness", [cfg.hbeam_um])
        _w("MinEta", [6.0])
        _w("MarginEta", [500.0])
        _w("MarginOme", [0.5])
        _w("MarginRadial", [500.0])
        _w("MarginRadius", [500.0])
        _w("MargABC", [4.8])
        _w("MargABG", [4.8])
        _w("Width", [1500.0])
        _w("EtaBinSize", [0.1])
        _w("OmeBinSize", [0.1])
        _w("StepSizeOrient", [0.2])
        _w("StepSizePos", [100.0])
        _w("MinNrSpots", [3], dtype=np.int32)
        _w("MinMatchesToAcceptFrac", [0.8])
        _w("MaxOmeSpotIDsToIndex", [90.0])
        _w("MinOmeSpotIDsToIndex", [-90.0])
        _w("DiscModel", [0], dtype=np.int32)
        _w("DiscArea", [2_250_000.0])
        _w("ImTransOpt", [0], dtype=np.int32)
        _w("NrFilesPerSweep", [1], dtype=np.int32)
        _w("GaussWidth", [int(round(cfg.gauss_sigma_px))], dtype=np.int64)
        _w("BoxSizes", [[-1_000_000.0, 1_000_000.0,
                         -1_000_000.0, 1_000_000.0]])
        _w("LatticeParameter", [cfg.lattice_a, cfg.lattice_a, cfg.lattice_a,
                                90.0, 90.0, 90.0])
        _w("SpaceGroup", [cfg.space_group], dtype=np.int32)
        _w("UseFriedelPairs", [1], dtype=np.int32)
        _w("OverallRingToIndex", [2], dtype=np.int32)
        _w("Vsample", [10_000_000.0])
        _w("GlobalPosition", [0.0])
        _w("NumPhases", [1], dtype=np.int32)
        _w("PhaseNr", [1], dtype=np.int32)
        _w("UpperBoundThreshold", [70_000.0])
        _w("PeakIntensity", [cfg.peak_intensity])
        _w("WriteSpots", [1], dtype=np.int32)
        _w("nScans", [1], dtype=np.int32)
        _w("tInt", [0.3])
        _w("tGap", [0.15])
        _w("Twins", [0], dtype=np.int32)
        _w("TakeGrainMax", [0], dtype=np.int32)
        ap.create_dataset("InFileName", data=np.bytes_(b"GrainsSim.csv"))
        ap.create_dataset("OutFileName", data=np.bytes_(b"midas_diffract_pinwheel"))
        # RingThresh: rings 1..rings_max with the same threshold
        ring_thresh = np.array([[r, cfg.ring_thresh] for r in range(1, cfg.rings_max + 1)],
                               dtype=np.float64)
        _w("RingThresh", ring_thresh)
        _w("RingsToExclude", np.array([[0, 0]], dtype=np.float64))
        _w("OmegaRanges", np.array([[
            min(cfg.omega_start_deg, cfg.omega_end_deg),
            max(cfg.omega_start_deg, cfg.omega_end_deg),
        ]], dtype=np.float64))


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def _ray_plane_assign(
    omega_deg: np.ndarray,
    eta_deg: np.ndarray,
    twoth_deg: np.ndarray,
    panels: List[PanelGeom],
    n_pixels: int,
    px_um: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-spot ray-plane intersection against each panel.

    For each spot defined by (ω, η, 2θ) in degrees, build the lab-frame
    diffraction direction ``k_lab``, then for each panel ``d`` solve

        t * k_lab  ∈  (panel d's plane)

    with the panel plane defined by point ``Lsd_d * R_d[:,0]`` and normal
    ``R_d[:,0]`` (FF convention: lab +x = beam, panel face perpendicular
    to its rotated normal). The intersection point ``P_lab = t * k_lab``
    is converted into panel-local pixel coordinates via the inverse
    rotation and the FF flip-y panel pixel formulae:

        Y_panel = R_d[:,1] · P_lab
        Z_panel = R_d[:,2] · P_lab
        y_pix   = yBC_d - Y_panel / px_um
        z_pix   = zBC_d + Z_panel / px_um

    A spot lands on panel ``d`` iff ``t > 0`` and ``(y_pix, z_pix)``
    falls inside ``[0, n_pixels)``. The first panel that accepts the
    spot wins (deterministic for pinwheel layouts where panels don't
    overlap).

    Parameters
    ----------
    omega_deg, eta_deg, twoth_deg
        Per-spot angles. Same length.
    panels
        Per-panel geometry (FF Rx*Ry*Rz convention for tx/ty/tz).
    n_pixels
        Detector pixel count per side.
    px_um
        Pixel size (μm).

    Returns
    -------
    det_index : np.int64 array, ``-1`` for spots that miss every panel
    y_pix, z_pix : float64, fractional pixel coordinates on the assigned
                   panel (NaN where det_index == -1).
    """
    d2r = math.pi / 180.0
    omega = omega_deg * d2r
    eta = eta_deg * d2r
    twoth = twoth_deg * d2r

    # Diffraction direction at ω = 0, MIDAS convention:
    #   k0_x = cos 2θ
    #   k0_y = -sin 2θ · sin η      (η = -90° → +Y direction)
    #   k0_z =  sin 2θ · cos η      (η = 0   → +Z direction)
    s2t = np.sin(twoth)
    c2t = np.cos(twoth)
    se = np.sin(eta)
    ce = np.cos(eta)
    k0x = c2t
    k0y = -s2t * se
    k0z =  s2t * ce

    # Rotate by ω about the lab z-axis (sample → lab):
    #   k_lab = Rz(ω) · k0
    cw = np.cos(omega)
    sw = np.sin(omega)
    kx = cw * k0x - sw * k0y
    ky = sw * k0x + cw * k0y
    kz = k0z

    n = omega.size
    det_index = np.full(n, -1, dtype=np.int64)
    y_pix = np.full(n, np.nan, dtype=np.float64)
    z_pix = np.full(n, np.nan, dtype=np.float64)

    for d, panel in enumerate(panels):
        tx, ty, tz = panel.tx * d2r, panel.ty * d2r, panel.tz * d2r
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(tx), -math.sin(tx)],
                       [0, math.sin(tx),  math.cos(tx)]])
        Ry = np.array([[math.cos(ty), 0, math.sin(ty)],
                       [0,            1, 0],
                       [-math.sin(ty), 0, math.cos(ty)]])
        Rz_ = np.array([[math.cos(tz), -math.sin(tz), 0],
                        [math.sin(tz),  math.cos(tz), 0],
                        [0, 0, 1]])
        R = Rx @ (Ry @ Rz_)            # FF convention Rx*Ry*Rz

        n_normal = R[:, 0]                  # panel face normal (lab)
        # Ray-plane intersection: t * (k · n_normal) = Lsd_d
        denom = kx * n_normal[0] + ky * n_normal[1] + kz * n_normal[2]
        # Spots whose ray is going away from the panel (denom <= 0)
        # cannot hit; mark via large denom mask.
        valid_d = denom > 1e-9
        t = np.where(valid_d, panel.lsd / np.where(valid_d, denom, 1.0), 0.0)

        Plab_x = t * kx
        Plab_y = t * ky
        Plab_z = t * kz

        # Project P_lab into the panel's local frame.
        Y_panel = R[0, 1] * Plab_x + R[1, 1] * Plab_y + R[2, 1] * Plab_z
        Z_panel = R[0, 2] * Plab_x + R[1, 2] * Plab_y + R[2, 2] * Plab_z

        y_pix_d = panel.y_bc - Y_panel / px_um
        z_pix_d = panel.z_bc + Z_panel / px_um

        on_panel = (
            valid_d
            & (y_pix_d >= 0) & (y_pix_d < n_pixels - 0.5)
            & (z_pix_d >= 0) & (z_pix_d < n_pixels - 0.5)
            & (det_index < 0)              # first-wins (no overlap in pinwheel)
        )
        det_index = np.where(on_panel, d, det_index)
        y_pix     = np.where(on_panel, y_pix_d, y_pix)
        z_pix     = np.where(on_panel, z_pix_d, z_pix)

    return det_index, y_pix, z_pix


def simulate_panel_zarrs(
    out_dir: Path,
    *,
    n_grains: int,
    panels: List[PanelGeom],
    cfg: SimConfig,
    seed: int = 42,
    out_stem: str = "midas_diffract_pinwheel",
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    log: Optional[callable] = None,
) -> dict:
    """Generate per-panel ``.MIDAS.zip`` zarrs for a multi-detector pinwheel.

    Strategy: forward-simulate on a single tilt-free *virtual* detector
    (large enough to contain all panels), then for each predicted spot
    figure out which physical panel sees it. This decouples the forward
    model from the panel-level tilt convention, so peakfit / fit-setup
    only ever see geometry their per-panel readers already understand.

    Parameters
    ----------
    out_dir, n_grains, panels, cfg, seed, out_stem, device, dtype, log
        See module docstring.

    Returns
    -------
    summary : dict
        ``{"zips": [...], "detectors_json": ..., "grains_csv": ...,
            "setup_truth_json": ..., "n_spots_per_panel": [...] }``
    """
    if log is None:
        def log(msg):
            print(f"[simulate_panel_zarrs] {msg}", flush=True)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_panels = len(panels)
    if n_panels < 1:
        raise ValueError("at least one panel is required")

    # ---- HKLs ----------------------------------------------------------
    hkls_cart, thetas, hkls_int, ring_id = _build_au_fcc_hkls(
        cfg.lattice_a, cfg.wavelength_A, cfg.rings_max,
    )
    log(f"HKLs: {len(hkls_cart)} reflections across {ring_id.max()} rings")

    # ---- Per-panel forward simulation via layered single-distance ---
    # Strategy: HEDMForwardModel's ``multi_mode="layered"`` with a single
    # distance and ``apply_tilts=True`` does the proper ray-plane
    # intersection between each diffraction ray and the panel plane —
    # exactly what we want for a tilted FF panel. We call it once per
    # physical panel, each time with that panel's true (Lsd, BC, tx, ty,
    # tz). A spot is recorded on a panel iff layered mode says it landed
    # within that panel's pixel bounds. The first panel to claim a spot
    # wins (pinwheel layouts don't overlap in practice).
    n_frames = int(round(abs(cfg.omega_end_deg - cfg.omega_start_deg)
                         / abs(cfg.omega_step_deg)))

    # CUDA fallback
    if device.startswith("cuda") and not torch.cuda.is_available():
        log(f"requested device={device}, but CUDA unavailable; falling back to CPU")
        device = "cpu"
    dev_t = torch.device(device)

    # ---- Grain population (shared across all per-panel sims) ---------
    eulers, positions, strains = _sample_grains(
        n_grains, cfg.rsample_um, cfg.hbeam_um, strain_mag=0.0, rng=rng,
    )
    eulers_t = torch.tensor(eulers, device=dev_t, dtype=dtype)
    positions_t = torch.tensor(positions, device=dev_t, dtype=dtype)
    latc = torch.tensor([cfg.lattice_a] * 3 + [90.0] * 3,
                        device=dev_t, dtype=dtype).repeat(n_grains, 1)
    strain_t = torch.tensor(strains, device=dev_t, dtype=dtype)

    # ---- Per-panel forward sim --------------------------------------
    # Each panel uses its own (Lsd, BC, tx, ty, tz). multi_mode="panel"
    # routes through HEDMForwardModel._apply_ff_panel_tilt — the inverse
    # of peakfit's compute_rt_eta (R^T, not R). Using ``layered`` mode
    # here would call _apply_nf_tilt (R forward) instead, which composes
    # to R² when peakfit reads it back, giving a 2·tx error in observed
    # η; that breaks indexing on every tilted panel.
    per_panel_records: list[dict] = []
    panel_claim = None              # global flat index of "spot already claimed"
    for panel_idx, panel in enumerate(panels):
        geom = HEDMGeometry(
            Lsd=float(panel.lsd),
            y_BC=float(panel.y_bc),
            z_BC=float(panel.z_bc),
            tx=float(panel.tx), ty=float(panel.ty), tz=float(panel.tz),
            px=cfg.px_um,
            omega_start=cfg.omega_start_deg,
            omega_step=cfg.omega_step_deg,
            n_frames=n_frames,
            n_pixels_y=cfg.n_pixels,
            n_pixels_z=cfg.n_pixels,
            min_eta=6.0,
            wavelength=cfg.wavelength_A,
            flip_y=True,
            apply_tilts=True,                # full per-panel tilt (FF inverse)
            multi_mode="panel",
            wedge=cfg.wedge_deg,
        )
        model = HEDMForwardModel(
            hkls=hkls_cart, thetas=thetas, geometry=geom,
            hkls_int=hkls_int, device=dev_t,
        )
        log(f"Forward sim det {panel.det_id}: Lsd={panel.lsd:.0f}μm, "
            f"BC=({panel.y_bc:.1f},{panel.z_bc:.1f}), "
            f"tx={panel.tx:.2f}° ty={panel.ty:.2f}° tz={panel.tz:.2f}°")
        with torch.no_grad():
            # Per-grain fast path: O(N*M), no orientation x strain cross-product.
            # Output is (2N, M) -- already the per-grain diagonal, no mask needed.
            spots = model.forward_per_grain(eulers_t, positions_t,
                                            lattice_params=latc, strain=strain_t)

        valid_np = (spots.valid > 0.5).cpu().numpy()   # (2N, M)
        Kdim, Mdim = valid_np.shape                     # Kdim == 2 * n_grains

        # First-wins: drop spots that an earlier panel already took
        # (avoids double-counting in the rare overlap).
        flat_valid = valid_np.reshape(-1)
        if panel_claim is None:
            panel_claim = np.zeros_like(flat_valid)
        flat_take = flat_valid & (~panel_claim)
        panel_claim = panel_claim | flat_take
        flat_idx = np.where(flat_take)[0]
        n_p = flat_idx.size
        log(f"  det {panel.det_id} took {n_p} spots from forward output")

        if n_p == 0:
            per_panel_records.append({"y_pix": np.zeros(0), "z_pix": np.zeros(0),
                                       "frame_nr": np.zeros(0),
                                       "omega_deg": np.zeros(0), "eta_deg": np.zeros(0),
                                       "twotheta_deg": np.zeros(0),
                                       "ring_nr": np.zeros(0, dtype=np.int32),
                                       "grain_id": np.zeros(0, dtype=np.int32)})
            continue

        omega_np = spots.omega.cpu().numpy() / DEG2RAD
        eta_np   = spots.eta.cpu().numpy()   / DEG2RAD
        twoth_np = spots.two_theta.cpu().numpy() / DEG2RAD
        y_pix    = spots.y_pixel.cpu().numpy()
        z_pix    = spots.z_pixel.cpu().numpy()
        frame_nr = spots.frame_nr.cpu().numpy()
        # (2N, M): row k holds grain (k % n_grains); rows [0:N]/[N:2N] are the
        # two omega branches.
        grain_ids = np.broadcast_to((np.arange(Kdim) % n_grains)[:, None],
                                    (Kdim, Mdim))
        hkl_ids = np.broadcast_to(np.arange(Mdim)[None, :], (Kdim, Mdim))

        rec = {
            "grain_id":     grain_ids.reshape(-1)[flat_idx].astype(np.int32),
            "ring_nr":      ring_id[hkl_ids.reshape(-1)[flat_idx]].astype(np.int32),
            "omega_deg":    omega_np.reshape(-1)[flat_idx].astype(np.float64),
            "eta_deg":      eta_np.reshape(-1)[flat_idx].astype(np.float64),
            "twotheta_deg": twoth_np.reshape(-1)[flat_idx].astype(np.float64),
            "y_pix":        y_pix.reshape(-1)[flat_idx].astype(np.float64),
            "z_pix":        z_pix.reshape(-1)[flat_idx].astype(np.float64),
            "frame_nr":     frame_nr.reshape(-1)[flat_idx].astype(np.float64),
        }
        if cfg.pos_noise_px > 0.0:
            rec["y_pix"] += rng.normal(0, cfg.pos_noise_px, rec["y_pix"].size)
            rec["z_pix"] += rng.normal(0, cfg.pos_noise_px, rec["z_pix"].size)
        in_f = (rec["frame_nr"] >= 0) & (rec["frame_nr"] < n_frames - 0.001)
        for k in rec:
            rec[k] = rec[k][in_f]
        per_panel_records.append(rec)

    # Flatten back into a single record set with det_id (panel index)
    spot_records = {k: np.concatenate([r[k] for r in per_panel_records])
                    for k in per_panel_records[0]}
    spot_records["det_id"] = np.concatenate([
        np.full(r["y_pix"].size, panel_idx, dtype=np.int32)
        for panel_idx, r in enumerate(per_panel_records)
    ])
    log(f"total spots across panels: {spot_records['y_pix'].size}")

    # ---- Per-panel rasterisation -------------------------------------
    zip_paths: list[Path] = []
    n_spots_per_panel: list[int] = []
    for panel_idx, panel in enumerate(panels):
        det_mask = spot_records["det_id"] == panel_idx
        n_p = int(det_mask.sum())
        n_spots_per_panel.append(n_p)
        log(f"  det {panel.det_id}: rendering {n_p} spots into "
            f"{n_frames} frames @ {cfg.n_pixels}x{cfg.n_pixels}")
        ys = spot_records["y_pix"][det_mask]
        zs = spot_records["z_pix"][det_mask]
        fs = spot_records["frame_nr"][det_mask]
        floor_f = np.floor(fs).astype(np.int64)
        frac_f = (fs - floor_f).astype(np.float64)            # ∈ [0, 1)

        # Build per-frame spot lists. Two modes:
        #   omega_sigma_frames == 0  → full intensity in floor(frame_nr)
        #                             (matches ForwardSimulationCompressed)
        #   omega_sigma_frames  > 0  → Gaussian smear across ±3σ frames
        per_frame: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        if cfg.omega_sigma_frames <= 0:
            for tf in np.unique(floor_f):
                idx = np.where(floor_f == tf)[0]
                if 0 <= int(tf) < n_frames:
                    per_frame.setdefault(int(tf), []).append((
                        ys[idx], zs[idx],
                        np.full(idx.size, cfg.peak_intensity, dtype=np.float64),
                    ))
        else:
            sigma_f = cfg.omega_sigma_frames
            K_f = max(1, int(math.ceil(3.0 * sigma_f)))
            for k in range(-K_f, K_f + 1):
                target = floor_f + k
                d = (fs - target.astype(np.float64))
                w = np.exp(-(d ** 2) / (2.0 * sigma_f ** 2))
                sel = (target >= 0) & (target < n_frames) & (w > 1e-3)
                if not sel.any():
                    continue
                t_sel = target[sel]
                for tf in np.unique(t_sel):
                    idx = np.where((target == tf) & sel)[0]
                    per_frame.setdefault(int(tf), []).append((
                        ys[idx], zs[idx],
                        (cfg.peak_intensity * w[idx]).astype(np.float64),
                    ))

        def _frame_iter():
            for f in range(n_frames):
                contribs = per_frame.get(f)
                if not contribs:
                    yield np.zeros((cfg.n_pixels, cfg.n_pixels), dtype=np.int32)
                    continue
                img = np.zeros((cfg.n_pixels, cfg.n_pixels), dtype=np.int32)
                for ys_t, zs_t, amp in contribs:
                    _render_frame(img, ys_t, zs_t, amp, cfg.gauss_sigma_px, cfg.n_pixels)
                yield img

        zip_path = out_dir / f"{out_stem}_det{panel.det_id}.analysis.MIDAS.zip"
        _write_panel_zarr(zip_path, _frame_iter(), cfg, panel, n_frames)
        zip_paths.append(zip_path)

    # ---- GrainsSim.csv (truth, MIDAS Au-FCC format) ------------------
    grains_csv = out_dir / "GrainsSim.csv"
    om_truth = _euler_zxz_to_om(eulers)             # (n_grains, 9)
    n_cols = 47
    sim_table = np.zeros((n_grains, n_cols), dtype=np.float64)
    sim_table[:, 0] = np.arange(1, n_grains + 1)            # GrainID
    sim_table[:, 1:10] = om_truth                            # OM
    sim_table[:, 10:13] = positions                          # X Y Z
    # Strain Voigt (E11 E22 E33 E12 E13 E23) → cols 13:19
    sim_table[:, 13:19] = strains
    # Lattice constants (cols 19:25) — set to nominal
    sim_table[:, 19:22] = cfg.lattice_a
    sim_table[:, 22:25] = 90.0
    # GrainRadius (col 25) — placeholder
    sim_table[:, 25] = 50.0
    # Confidence
    sim_table[:, 26] = 1.0
    with grains_csv.open("w") as fp:
        fp.write(f"%NumGrains {n_grains}\n")
        fp.write("%BeamCenter 0.000000\n")
        fp.write("%BeamThickness 200.000000\n")
        fp.write("%GlobalPosition 0.000000\n")
        fp.write("%NumPhases 1\n")
        fp.write(f"%PhaseInfo\n")
        fp.write(f"%\tSpaceGroup:{cfg.space_group}\n")
        fp.write(f"%\tLattice Parameter:{cfg.lattice_a:.6f}\t{cfg.lattice_a:.6f}\t{cfg.lattice_a:.6f}\t90.000000\t90.000000\t90.000000\n")
        for row in sim_table:
            fp.write("\t".join(f"{v:.6f}" for v in row) + "\n")

    # ---- detectors.json + setup_truth.json ---------------------------
    detectors_json = out_dir / "detectors.json"
    detectors_records = []
    for panel, zip_path in zip(panels, zip_paths):
        rec = asdict(panel)
        rec["zarr_path"] = str(zip_path)
        detectors_records.append(rec)
    with detectors_json.open("w") as fp:
        json.dump(detectors_records, fp, indent=2)

    setup_truth = {
        "seed": seed,
        "n_grains": n_grains,
        "n_panels": n_panels,
        "config": asdict(cfg),
        "panels": [asdict(p) for p in panels],
        "grains": {
            "eulers_zxz_rad": eulers.tolist(),
            "positions_um": positions.tolist(),
            "strains_voigt": strains.tolist(),
            "orient_mat": om_truth.tolist(),
        },
        "n_spots_per_panel": n_spots_per_panel,
    }
    setup_truth_json = out_dir / "setup_truth.json"
    with setup_truth_json.open("w") as fp:
        json.dump(setup_truth, fp, indent=2)

    log(f"wrote {len(zip_paths)} zarrs + GrainsSim.csv + detectors.json + setup_truth.json")
    return {
        "zips": zip_paths,
        "detectors_json": detectors_json,
        "grains_csv": grains_csv,
        "setup_truth_json": setup_truth_json,
        "n_spots_per_panel": n_spots_per_panel,
        "n_frames": n_frames,
    }
