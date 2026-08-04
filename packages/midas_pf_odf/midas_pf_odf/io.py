"""Real-data ingest for pf-HEDM joint peak-shape inversion.

Bridges a MIDAS pf-HEDM run (the artifacts under ``LayerNr_<n>/``) to the
:class:`~midas_pf_odf.simulate.GrainPatchData` contract the inversion driver
consumes. Three layers:

1. **Geometry / model** — build a correct :class:`HEDMGeometry` +
   :class:`ScanConfig` + :class:`HEDMForwardModel` directly from
   ``paramstest.txt`` (we do NOT reuse ``midas_fit_grain._build_model``: it
   carries placeholder geometry defaults — ``y_BC=1024``, ``omega_step=0.25`` —
   that are wrong for real data).
2. **Grain loader** — :func:`load_pf_grain` reads ``voxel_grid.csv`` (the
   voxel→grain map + positions) and the per-voxel refined ``Results/`` CSVs
   (orientation + lattice warm-start) for one grain.
3. **Patch assembly** — :func:`assemble_grain_patch_data` runs the forward at
   the warm-start to fix the per-spot anchors + observed mask in the model's
   ``S = 2M`` (branch, hkl) layout, then fills the measured ``(S, Σ, F, P, P)``
   tensor from a pluggable source (a synthetic patch tensor for tests, or a
   raw-frame reader for real data via :func:`crop_patches_from_frames`).

Units follow MIDAS: micrometres, degrees, Ångströms (wavelength + lattice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry, ScanConfig

from midas_pf_odf.simulate import GrainPatchData, _voxel_summed_spots


# ---------------------------------------------------------------------------
# paramstest.txt parsing
# ---------------------------------------------------------------------------

def parse_paramstest(path: str | Path) -> Dict[str, List[List[str]]]:
    """Parse a MIDAS ``paramstest.txt`` into ``{key: [tokens, ...], ...}``.

    Trailing semicolons are stripped. Repeated keys (e.g. ``RingNumbers``)
    accumulate one token-list per occurrence. Values stay as strings; callers
    coerce. Robust to blank lines and comments (``#``).
    """
    out: Dict[str, List[List[str]]] = {}
    with open(path, "r") as fp:
        for raw in fp:
            line = raw.split("#", 1)[0].strip().rstrip(";").strip()
            if not line:
                continue
            toks = line.split()
            if not toks:
                continue
            out.setdefault(toks[0], []).append(toks[1:])
    return out


def _first_float(params: Dict, key: str, default: Optional[float] = None) -> float:
    if key in params and params[key] and params[key][0]:
        return float(params[key][0][0])
    if default is not None:
        return float(default)
    raise KeyError(f"paramstest missing required key {key!r}")


# ---------------------------------------------------------------------------
# Geometry / scan config / model
# ---------------------------------------------------------------------------

def distortion_from_paramstest(params: Dict) -> Tuple[Optional[list], Optional[float]]:
    """Extract calibrated detector distortion from a parsed paramstest.

    Returns ``(p_v2, rho_d_um)`` in the form :class:`HEDMGeometry` expects
    (``p_distortion`` = 15 coeffs in midas_distortion ``P_COEF_NAMES``
    v2-canonical order; ``rho_d`` in µm), or ``(None, None)`` when the file
    carries no distortion.

    ⚠️ paramstest ``p0..p14`` are **v1-ordered** — p3 is ``phi4`` (a PHASE,
    degrees), not an amplitude; a naive v2-order copy explodes into
    meter-scale shifts (midas_distortion core.py V1_TO_V2_DISTORTION is
    the authoritative map, applied via ``v2_coeffs_from_named``). v2 names
    (iso_R2.., a1.., phi1.., written by midas-transforms >= 0.8.0 for
    calibrate-v2-native archives) take precedence over pN on collision.
    """
    from midas_distortion import P_COEF_NAMES, v2_coeffs_from_named

    named: Dict[str, float] = {}
    for i in range(15):
        k = f"p{i}"
        if k in params and params[k] and params[k][0]:
            v = float(params[k][0][0])
            if v != 0.0:
                named[k] = v
    for nm in P_COEF_NAMES:
        if nm in params and params[nm] and params[nm][0]:
            v = float(params[nm][0][0])
            if v != 0.0:
                named[nm] = v
    if not named:
        return None, None
    p_v2 = [float(x) for x in v2_coeffs_from_named(named)]
    rho_d = None
    for key in ("RhoD", "MaxRingRad"):
        if key in params and params[key] and params[key][0]:
            v = float(params[key][0][0])
            if v > 0:
                rho_d = v
                break
    return p_v2, rho_d


def geometry_from_paramstest(
    params: Dict,
    *,
    n_pixels_y: int,
    n_pixels_z: int,
    n_frames: Optional[int] = None,
    omega_step: Optional[float] = None,
    apply_distortion: bool = False,
) -> HEDMGeometry:
    """Build a correct :class:`HEDMGeometry` from a parsed paramstest.

    Maps the MIDAS keys: ``Distance``/``LsdFit`` → Lsd, ``YBCFit`` → y_BC,
    ``ZBCFit`` → z_BC, ``OmegaRange`` → omega_start, ``tyFit``/``tzFit`` →
    ty/tz tilts (tx assumed 0), ``Wedge`` → wedge. ``flip_y=True`` (FF/pf
    DetCor convention). Detector pixel dimensions must be supplied by the
    caller (paramstest does not record them).

    **Frame mapping (important):** the per-frame omega step is the
    *acquisition* step — NOT ``OmeBinSize`` (the indexing omega-bin size,
    generally different: datasetC has OmeBinSize 0.1 but 1440 frames over
    360° = 0.25°/frame). Resolution order:

    1. caller kwargs ``n_frames`` (= NrFilesPerSweep) / ``omega_step``;
    2. explicit ``OmegaStart``/``OmegaStep`` paramstest keys
       (midas-transforms ≥ 0.8.0 writes them);
    3. a SINGLE ``OmegaRange`` span (legacy inference);
    4. multiple ``OmegaRange`` lines with no explicit step → **ValueError**.
       Shadow-gapped multi-range files (e.g. SOH: −180/−106, −76/74,
       105/180) made the old span inference return 0.0514° instead of
       0.25° — every forward anchor then landed on empty frames.

    When ranges are present the omega origin/coverage always spans ALL
    ranges (min start → max end): the acquisition frames are continuous in
    omega; the ranges only mark valid windows.
    """
    import warnings

    # Lsd: prefer the fitted value if present.
    if "LsdFit" in params:
        Lsd = _first_float(params, "LsdFit")
    else:
        Lsd = _first_float(params, "Distance")
    y_BC = _first_float(params, "YBCFit") if "YBCFit" in params else _first_float(params, "YBC", 0.0)
    z_BC = _first_float(params, "ZBCFit") if "ZBCFit" in params else _first_float(params, "ZBC", 0.0)
    px = _first_float(params, "px")
    wavelength = _first_float(params, "Wavelength")

    ranges = [(float(r[0]), float(r[1])) for r in params.get("OmegaRange", []) if r]
    n_ranges = len(ranges)
    key_start = _first_float(params, "OmegaStart", 0.0) if "OmegaStart" in params else None
    key_step = _first_float(params, "OmegaStep", 0.0) if "OmegaStep" in params else None
    if key_step is not None and key_step == 0.0:
        key_step = None            # a false 0.0 (E5 class) is not a step

    if ranges:
        # Full acquisition span across ALL ranges (frames are continuous
        # in omega; shadow gaps only mark invalid windows).
        starts = [r[0] for r in ranges]
        ends = [r[1] for r in ranges]
        sign = 1.0 if ranges[0][1] >= ranges[0][0] else -1.0
        om0 = min(starts) if sign > 0 else max(starts)
        om_range = abs(max(max(starts), max(ends)) - min(min(starts), min(ends)))
    else:
        sign = 1.0
        om0, om_range = -180.0, 360.0
    if key_start is not None:
        om0 = key_start

    if n_frames is not None:
        n_frames = int(n_frames)
        omega_step = math.copysign(om_range / n_frames, sign)
    elif omega_step is not None:
        omega_step = float(omega_step)
        n_frames = int(round(om_range / abs(omega_step)))
    elif key_step is not None:
        # Explicit paramstest keys (midas-transforms >= 0.8.0).
        omega_step = float(key_step)
        n_frames = int(round(om_range / abs(omega_step)))
    elif n_ranges > 1:
        raise ValueError(
            "geometry_from_paramstest: paramstest has "
            f"{n_ranges} OmegaRange lines and no explicit OmegaStep key — "
            "refusing to infer the frame step from shadow-gapped spans "
            "(this returned 0.0514° instead of 0.25° on real data). Pass "
            "n_frames=NrFilesPerSweep or omega_step=..., or regenerate "
            "paramstest with midas-transforms >= 0.8.0 (writes "
            "OmegaStart/OmegaStep)."
        )
    else:
        omega_step = _first_float(params, "OmeBinSize")
        n_frames = int(round(om_range / abs(omega_step)))
        warnings.warn(
            "geometry_from_paramstest: no n_frames/omega_step given; falling "
            "back to OmeBinSize for the frame step. This is the indexing bin "
            "size, not the acquisition frame step, and is likely wrong for "
            "patch frame-cropping. Pass n_frames=NrFilesPerSweep.",
            stacklevel=2,
        )
    min_eta = _first_float(params, "MinEta", _first_float(params, "ExcludePoleAngle", 6.0))
    ty = _first_float(params, "tyFit", _first_float(params, "ty", 0.0))
    tz = _first_float(params, "tzFit", _first_float(params, "tz", 0.0))
    tx = _first_float(params, "txFit", _first_float(params, "tx", 0.0))
    wedge = _first_float(params, "Wedge", 0.0)
    # P1-4: raw-frame consumers must model the calibrated detector
    # distortion (measured effect on SOH Varex: 4-166 µm ring shifts,
    # ~100 µε on Stage-1 strain). Opt-in: the experimental pipeline
    # pre-corrects distortion at transforms time, so ideal-frame
    # consumers must NOT re-apply it.
    p_v2, rho_d = (None, None)
    if apply_distortion:
        p_v2, rho_d = distortion_from_paramstest(params)
        if p_v2 is None:
            warnings.warn(
                "geometry_from_paramstest: apply_distortion=True but the "
                "paramstest carries no distortion coefficients (p0..p14 "
                "all zero/absent, no v2 names). Proceeding undistorted.",
                stacklevel=2,
            )
    return HEDMGeometry(
        Lsd=Lsd, y_BC=y_BC, z_BC=z_BC, px=px,
        omega_start=om0, omega_step=omega_step, n_frames=n_frames,
        n_pixels_y=int(n_pixels_y), n_pixels_z=int(n_pixels_z),
        min_eta=min_eta, wavelength=wavelength,
        tx=tx, ty=ty, tz=tz, wedge=wedge, flip_y=True,
        apply_distortion=bool(apply_distortion and p_v2 is not None),
        p_distortion=p_v2, rho_d=rho_d,
    )


def scan_config_from_positions(
    positions_csv: str | Path,
    beam_size_um: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> ScanConfig:
    """Build a :class:`ScanConfig` from ``positions.csv`` (one beam-y per line)."""
    pos = np.loadtxt(positions_csv).reshape(-1)
    return ScanConfig(
        beam_positions=torch.as_tensor(pos, dtype=dtype),
        beam_size=float(beam_size_um),
    )


def build_model_from_paramstest(
    layer_dir: str | Path,
    *,
    n_pixels_y: int,
    n_pixels_z: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    max_two_theta_deg: Optional[float] = None,
    n_frames: Optional[int] = None,
    omega_step: Optional[float] = None,
    apply_distortion: bool = False,
    ring_numbers: Optional[Sequence[int]] = None,
) -> Tuple[HEDMForwardModel, np.ndarray]:
    """Build the pf-HEDM forward model (geometry + scan_config + hkls).

    Reads ``paramstest.txt``, ``hkls.csv`` and ``positions.csv`` from
    ``layer_dir``. Returns ``(model, ring_nr)`` where ``ring_nr`` is the
    per-reflection MIDAS ring number (length M).

    ``n_frames`` (= NrFilesPerSweep from the master params) or ``omega_step``
    set the acquisition frame step for patch cropping; see
    :func:`geometry_from_paramstest`.

    ``apply_distortion=True`` (P1-4) plumbs the calibrated detector
    distortion from paramstest into the forward (raw-frame prediction);
    see :func:`distortion_from_paramstest` for the v1→v2 mapping caveat.

    ``ring_numbers`` (P1-5) overrides the paramstest ``RingNumbers``
    (``None`` = paramstest behaviour). hkls.csv already carries all rings,
    and the rings used for INDEXING are generally not the optimal rings
    for strain (SOH matrix: indexing used 3,4; rings 1,2 are 3-4× brighter
    and fully on-detector; 5,7,8 are corner-clipped).
    """
    from midas_fit_grain.driver import _read_hkls_csv, _cartesian_B_matrix

    layer = Path(layer_dir)
    params = parse_paramstest(layer / "paramstest.txt")

    lat = [float(x) for x in params["LatticeParameter"][0][:6]]
    if ring_numbers is not None:
        ring_numbers = [int(r) for r in ring_numbers]
    else:
        ring_numbers = [int(r[0]) for r in params.get("RingNumbers", [])]
    if not ring_numbers:
        raise ValueError("paramstest has no RingNumbers")
    beam_size = _first_float(params, "BeamSize")

    geom = geometry_from_paramstest(
        params, n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
        n_frames=n_frames, omega_step=omega_step,
        apply_distortion=apply_distortion,
    )
    if max_two_theta_deg is None:
        if "MaxRingRad" in params:
            mrr = _first_float(params, "MaxRingRad")
            max_two_theta_deg = 2.0 * math.degrees(math.atan(mrr / geom.Lsd))
        else:
            max_two_theta_deg = 180.0

    hkls_int, thetas_deg, ring_nr = _read_hkls_csv(
        layer / "hkls.csv", ring_numbers, max_two_theta_deg
    )
    B = _cartesian_B_matrix(tuple(lat))                       # (3, 3), 1/Å
    hkls_cart = (B @ hkls_int.astype(np.float64).T).T          # (M, 3)
    thetas_rad = thetas_deg * (math.pi / 180.0)

    scan_cfg = scan_config_from_positions(layer / "positions.csv", beam_size, dtype=dtype)

    model = HEDMForwardModel(
        torch.from_numpy(hkls_cart),
        torch.from_numpy(thetas_rad),
        geom,
        hkls_int=torch.from_numpy(hkls_int.astype(np.float64)),
        scan_config=scan_cfg,
        device=device,
    )
    return model, ring_nr


# ---------------------------------------------------------------------------
# Grain dataset
# ---------------------------------------------------------------------------

@dataclass
class PFGrainDataset:
    """Everything the inversion needs for one grain, from a real pf-HEDM run."""
    grain_id: int
    voxel_idx: np.ndarray            # (G,) global voxel indices (into voxel_grid.csv)
    voxel_pos: torch.Tensor          # (G, 3) sample-frame µm
    R_init: torch.Tensor             # (G, 3, 3)
    eps_init: torch.Tensor           # (G, 6) Voigt, crystal frame
    lattice_init: torch.Tensor       # (6,) [a,b,c,α,β,γ]
    model: HEDMForwardModel
    grid_shape: Tuple[int, int]      # bounding-box (G_x, G_y) for plotting (may be sparse)
    grid_ij: np.ndarray              # (G, 2) row/col of each voxel in the full scan grid
    metadata: dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_pos.shape[0])

    @property
    def scan_positions(self) -> torch.Tensor:
        return self.model.scan_config.beam_positions


def _canonicalize_oms(oms: np.ndarray, space_group: int, ref: int = 0) -> np.ndarray:
    """Align every per-voxel OM to ONE symmetry variant (that of ``oms[ref]``).

    The pf-HEDM Results CSVs store per-voxel orientations scattered across
    symmetry-equivalent variants (sub-0.03° in misorientation but different
    matrices). Symmetry-equivalent OMs produce *identical physical diffraction*,
    but a fixed spot index ``(branch, hkl)`` maps to a DIFFERENT physical
    reflection per variant — so a multi-voxel forward must put all voxels in the
    same variant, else per-spot anchors (the voxel mean) are meaningless.

    Each OM is replaced by the crystal-symmetry-equivalent quaternion closest to
    the reference; this preserves the real <0.03° intragranular spread while
    making the spot layout consistent. Returns ``(G, 3, 3)`` float64.
    """
    from midas_stress.orientation import (
        make_symmetries, orient_mat_to_quat, quat_to_orient_mat,
        quaternion_product,
    )
    _, sym = make_symmetries(int(space_group))
    sym = np.asarray(sym, dtype=np.float64).reshape(-1, 4)        # (n_sym, 4)
    # midas_stress orientation primitives use FLAT 9-element row-major OMs.
    q_ref = np.asarray(orient_mat_to_quat(oms[ref].reshape(9)),
                       dtype=np.float64).ravel()
    out = np.empty_like(oms)
    for v in range(oms.shape[0]):
        q = np.asarray(orient_mat_to_quat(oms[v].reshape(9)),
                       dtype=np.float64).ravel()
        best_q, best_dot = q, -1.0
        for s in sym:
            qe = np.asarray(quaternion_product(q, s), dtype=np.float64).ravel()
            dot = abs(float(qe @ q_ref))
            if dot > best_dot:
                best_dot, best_q = dot, qe
        out[v] = np.asarray(quat_to_orient_mat(best_q),
                            dtype=np.float64).reshape(3, 3)
    return out


def _read_voxel_result(
    results_dir: Path, v: int, dtype: torch.dtype,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read (OM 3x3, lattice 6) from Result_OrientPos_voxel_<v>.csv.

    Returns None if the file is absent/empty/short.
    """
    p = results_dir / f"Result_OrientPos_voxel_{v}.csv"
    if not p.exists():
        return None
    try:
        row = np.loadtxt(p, skiprows=1)
    except Exception:
        return None
    if row.ndim != 1 or row.size < 21:
        return None
    om = row[1:10].reshape(3, 3)
    latc = row[15:21]
    return om, latc


def load_pf_grain(
    layer_dir: str | Path,
    grain_id: int,
    *,
    n_pixels_y: int,
    n_pixels_z: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    model: Optional[HEDMForwardModel] = None,
    n_frames: Optional[int] = None,
    omega_step: Optional[float] = None,
    canonicalize_symmetry: bool = True,
    space_group: Optional[int] = None,
) -> PFGrainDataset:
    """Load one grain's voxels + per-voxel warm-start from a pf-HEDM run.

    Reads ``Output/voxel_grid.csv`` (voxel_idx, x_um, y_um, z_um, grain_id)
    and per-voxel ``Results/Result_OrientPos_voxel_<v>.csv`` (refined
    orientation + lattice). Voxels with no usable Results row are dropped.
    ``eps_init`` is initialised to zero (deviatoric strain; the lattice
    carries the bulk), ``lattice_init`` is the per-grain median lattice.

    Pass a prebuilt ``model`` to avoid re-reading paramstest/hkls per grain.
    """
    layer = Path(layer_dir)
    if model is None:
        model, _ = build_model_from_paramstest(
            layer, n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
            device=device, dtype=dtype,
            n_frames=n_frames, omega_step=omega_step,
        )

    vg = np.loadtxt(layer / "Output" / "voxel_grid.csv", skiprows=1)
    vidx_all = vg[:, 0].astype(np.int64)
    xyz_all = vg[:, 1:4].astype(np.float64)
    grain_all = vg[:, 4].astype(np.int64)

    # Infer the full square scan grid (N x N) from voxel count.
    nv = grain_all.size
    N = int(round(math.sqrt(nv)))
    sel = np.where(grain_all == grain_id)[0]
    if sel.size == 0:
        raise ValueError(f"grain {grain_id} not found in voxel_grid.csv")

    results_dir = layer / "Results"
    keep_idx: List[int] = []
    oms: List[np.ndarray] = []
    lats: List[np.ndarray] = []
    for v in sel.tolist():
        res = _read_voxel_result(results_dir, int(vidx_all[v]), dtype)
        if res is None:
            continue
        om, latc = res
        keep_idx.append(v)
        oms.append(om)
        lats.append(latc)
    if not keep_idx:
        raise ValueError(
            f"grain {grain_id}: no usable Results CSVs in {results_dir}"
        )

    keep = np.array(keep_idx, dtype=np.int64)
    voxel_pos = torch.as_tensor(xyz_all[keep], dtype=dtype, device=device)
    om_stack = np.stack(oms)
    if canonicalize_symmetry:
        if space_group is None:
            params = parse_paramstest(layer / "paramstest.txt")
            space_group = int(params["SpaceGroup"][0][0]) if "SpaceGroup" in params else 225
        om_stack = _canonicalize_oms(om_stack, int(space_group))
    R_init = torch.as_tensor(om_stack, dtype=dtype, device=device)
    eps_init = torch.zeros(len(keep), 6, dtype=dtype, device=device)
    lattice_init = torch.as_tensor(np.median(np.stack(lats), axis=0),
                                   dtype=dtype, device=device)

    grid_ij = np.stack([vidx_all[keep] // N, vidx_all[keep] % N], axis=1)
    gx = int(grid_ij[:, 0].max() - grid_ij[:, 0].min() + 1)
    gy = int(grid_ij[:, 1].max() - grid_ij[:, 1].min() + 1)

    return PFGrainDataset(
        grain_id=int(grain_id),
        voxel_idx=vidx_all[keep],
        voxel_pos=voxel_pos,
        R_init=R_init,
        eps_init=eps_init,
        lattice_init=lattice_init,
        model=model,
        grid_shape=(gx, gy),
        grid_ij=grid_ij,
        metadata={"n_scan": N, "n_dropped": int(sel.size - keep.size)},
    )


# ---------------------------------------------------------------------------
# Patch assembly
# ---------------------------------------------------------------------------

def _forward_anchors(
    dataset: PFGrainDataset,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run the warm-start forward → per-spot anchors + validity in the
    model's S=2M layout. Returns (anchor_y, anchor_z, anchor_f, spot_valid
    (G,S), spot_observed (S,), S)."""
    with torch.no_grad():
        spots = _voxel_summed_spots(
            dataset.model, dataset.R_init, dataset.eps_init,
            dataset.lattice_init, dataset.voxel_pos, apply_scan_filter=False,
        )
    G = dataset.n_voxels
    M = int(spots.y_pixel.shape[-1])
    S = 2 * M
    sy = spots.y_pixel.reshape(G, S)
    sz = spots.z_pixel.reshape(G, S)
    sf = spots.frame_nr.reshape(G, S)
    sv = spots.valid.reshape(G, S).to(dtype)
    # Robust per-spot anchor: MEDIAN over valid voxels (immune to the rare
    # mis-canonicalized / Ewald-marginal voxel that predicts the spot far away;
    # a mean is pulled by such outliers, a median is not).
    valid_mask = sv > 0.5
    nan = torch.full((), float("nan"), dtype=dtype, device=sy.device)
    sy_m = torch.where(valid_mask, sy, nan)
    sz_m = torch.where(valid_mask, sz, nan)
    sf_m = torch.where(valid_mask, sf, nan)
    # Spots with NO valid voxel have a nanmedian of NaN; they are unobserved
    # (masked by spot_observed) but their anchor must stay finite or the splat
    # produces NaNs that poison the loss. Send them to 0 (an in-bounds pixel).
    anchor_y = torch.nan_to_num(sy_m.nanmedian(dim=0).values, nan=0.0)
    anchor_z = torch.nan_to_num(sz_m.nanmedian(dim=0).values, nan=0.0)
    anchor_f = torch.nan_to_num(sf_m.nanmedian(dim=0).values, nan=0.0)
    spot_observed = sv.sum(dim=0) > 0.5
    return anchor_y, anchor_z, anchor_f, valid_mask, spot_observed, S


def assemble_grain_patch_data(
    dataset: PFGrainDataset,
    *,
    measured_patches: Optional[torch.Tensor] = None,
    frame_reader: Optional[Callable[[int, int], np.ndarray]] = None,
    patch_F: int = 5,
    patch_P: int = 15,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    saturation_threshold: Optional[float] = None,
    subtract_background: bool = False,
    background_border: int = 3,
) -> GrainPatchData:
    """Assemble :class:`GrainPatchData` for one grain.

    Anchors + observed mask come from the warm-start forward (the model's
    S=2M layout, matching what the inversion re-predicts). The measured
    ``(S, Σ, F, P, P)`` tensor is supplied one of two ways:

    * ``measured_patches`` — a ready tensor (synthetic tests / cached real data).
    * ``frame_reader(scan_idx, frame_idx) -> 2D ndarray`` — raw detector
      frames; patches are cropped around the per-spot anchors via
      :func:`crop_patches_from_frames`.

    ``subtract_background`` (default ``False``): pf-odf's loss has no additive
    background term (the closed-form per-spot scale is purely multiplicative), so
    a flat pedestal under every spot biases the fit — the scale inflates and the
    residual is dominated by unmatched background pixels. Set ``True`` when the
    patches are **raw** (NOT dark-subtracted): a per-(spot, scan) background is
    estimated from the central frame's outer ``background_border``-pixel ring
    (away from the centred spot), subtracted from all frames, and clamped ≥ 0.
    ⚠️ Leave ``False`` for already-dark-subtracted caches (e.g. the SOH reader
    stores ``raw − dark``) — double-subtracting would clip real signal. Measured
    impact on a raw att5 Ni scan: median strain −30% and a materially different
    per-voxel field (raw vs corrected only ~0.26-correlated).

    ``saturation_threshold`` (P2-7): counts, IN THE UNITS OF THE PATCHES
    AS STORED. ⚠️ Patch caches are typically DARK-SUBTRACTED (the SOH
    reader stores ``raw − dark``), so raw-clamped pixels sit just BELOW
    the detector ``UpperBoundThreshold`` — passing the raw 64000 masks
    NOTHING (measured on SOH: 0 pixels ≥ 64000 in a 96%-saturated ring
    set). Use ``threshold − max(dark)`` or the diagnosis convention
    (≥ 60000 for the 64000-clamp Varex). Pixels with measured ≥
    threshold get a hard 0 weight in the data MSE AND in the per-spot
    scale fit — on the SOH Varex rings 1/2 are 96-98% saturated; fitting
    narrow Gaussian splats against those flat-tops floors the loss and
    the strain runs away (ε grew 4237 → 5959 µε with more steps). None =
    no masking.
    """
    a_y, a_z, a_f, spot_valid, spot_observed, S = _forward_anchors(
        dataset, dtype, device
    )
    Sigma = int(dataset.scan_positions.numel())

    if measured_patches is not None:
        meas = measured_patches.to(dtype).to(device)
        if tuple(meas.shape) != (S, Sigma, patch_F, patch_P, patch_P):
            raise ValueError(
                f"measured_patches shape {tuple(meas.shape)} != "
                f"expected {(S, Sigma, patch_F, patch_P, patch_P)}"
            )
    elif frame_reader is not None:
        meas = crop_patches_from_frames(
            frame_reader, a_y, a_z, a_f, Sigma, spot_observed,
            patch_F=patch_F, patch_P=patch_P,
            n_pixels_y=int(dataset.model.n_pixels_y),
            n_pixels_z=int(dataset.model.n_pixels_z),
            dtype=dtype, device=device,
        )
    else:
        raise ValueError("supply either measured_patches or frame_reader")

    if subtract_background:
        b = int(background_border)
        if 2 * b >= patch_P:
            raise ValueError(
                f"background_border={b} too large for patch_P={patch_P}"
            )
        cf = meas[:, :, patch_F // 2]                     # (S, Σ, P, P) central frame
        border = torch.ones(patch_P, patch_P, dtype=torch.bool, device=meas.device)
        border[b:patch_P - b, b:patch_P - b] = False      # True on the outer ring
        bg = meas.new_zeros(meas.shape[0], meas.shape[1])
        obs = spot_observed.reshape(-1).bool()
        if obs.any():
            # index the two trailing (P, P) dims with the border mask
            bg[obs] = cf[obs][:, :, border].median(dim=-1).values
        meas = (meas - bg[:, :, None, None, None]).clamp_(min=0.0)

    sat_mask = None
    if saturation_threshold is not None:
        sat_mask = (meas < float(saturation_threshold)).detach()
        n_sat = int((~sat_mask).sum())
        if n_sat:
            frac = n_sat / sat_mask.numel()
            LOG_FN = print   # io has no logger; keep the report visible
            LOG_FN(f"[assemble] saturation mask: {n_sat} px "
                   f"({100*frac:.1f}%) >= {saturation_threshold:g} masked")

    return GrainPatchData(
        anchor_y=a_y.detach(),
        anchor_z=a_z.detach(),
        anchor_f=a_f.detach(),
        scan_positions=dataset.scan_positions.to(dtype).to(device).detach(),
        measured_patches=meas.detach(),
        spot_valid=spot_valid.detach(),
        spot_observed=spot_observed.detach(),
        spot_indexer=torch.arange(S, dtype=torch.long, device=device),
        sigma_yz=float(sigma_yz),
        sigma_f=float(sigma_f),
        patch_F=int(patch_F),
        patch_P=int(patch_P),
        saturation_mask=sat_mask,
    )


def saturation_threshold_from_paramstest(params: Dict) -> Optional[float]:
    """P2-7 threshold source: the detector ``UpperBoundThreshold`` key
    (paramstest or master params, e.g. SOH Varex: 64000). Returns None
    when absent/zero — callers may still pass an explicit
    ``saturation_threshold`` to :func:`assemble_grain_patch_data`."""
    v = _first_float(params, "UpperBoundThreshold", 0.0)
    return float(v) if v > 0 else None


# ---------------------------------------------------------------------------
# Zarr (.MIDAS.zip) ingest — authoritative geometry + raw frames
# ---------------------------------------------------------------------------
#
# A MIDAS .zip is a zarr store with:
#   exchange/data  (n_frames, n_raw_rows, n_raw_cols) uint  — RAW frames
#   exchange/dark  (1, n_raw_rows, n_raw_cols)              — dark
#   analysis/process/analysis_parameters/*                  — full geometry
#   measurement/process/scan_parameters/{start,step}        — omega per frame
#
# The stored frames are RAW: the MIDAS image transform (ImTransOpt + the
# detector transpose) and dark subtraction are applied at READ time. For the
# datasetC data ImTransOpt=[2,0] ⇒ ``flipud`` then the always-transpose, i.e.
# ``corrected = flipud(raw).T - flipud(dark).T`` giving a ``[y, z]`` frame
# (matches utils/radial_integration_comparison.py and pyFAI/MIDAS parity).

def read_zarr_params(zip_path: str | Path) -> Dict[str, np.ndarray]:
    """Read the analysis + scan parameters from a ``.MIDAS.zip`` zarr store."""
    import zarr
    z = zarr.open(zarr.ZipStore(str(zip_path), mode="r"), mode="r")
    ap = z["analysis/process/analysis_parameters"]
    out: Dict[str, np.ndarray] = {}
    for k in ap.keys():
        out[k] = np.array(ap[k])
    sp = z["measurement/process/scan_parameters"]
    out["_omega_start"] = float(np.array(sp["start"]).ravel()[0])
    out["_omega_step"] = float(np.array(sp["step"]).ravel()[0])
    return out


def _scalar(zp: Dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(zp[key]).ravel()[0])


def geometry_from_zarr(zip_path: str | Path) -> HEDMGeometry:
    """Build :class:`HEDMGeometry` from a ``.MIDAS.zip`` — the authoritative
    source for the frame↔omega mapping, detector size, and tilts.

    Uses ``scan_parameters.start/step`` for omega (NOT the analysis-params
    ``OmegaStart``/``omegaStep`` which can be unset), ``numPxY/numPxZ`` for the
    transposed ``[y, z]`` detector size, ``YCen``/``ZCen`` for the beam centre.
    """
    zp = read_zarr_params(zip_path)
    om_start = zp["_omega_start"]
    om_step = zp["_omega_step"]
    n_frames = int(_scalar(zp, "numFilesPerScan"))
    # The MIDAS analysis frame is square-padded to NrPixels = max(numPxY, numPxZ)
    # (see midas_peakfit.preprocess.make_square_image); observed spot (y, z)
    # pixel coords live in that square frame, so the forward must use it too.
    n_sq = max(int(_scalar(zp, "numPxY")), int(_scalar(zp, "numPxZ")))
    return HEDMGeometry(
        Lsd=_scalar(zp, "Lsd"),
        y_BC=_scalar(zp, "YCen"), z_BC=_scalar(zp, "ZCen"),
        px=_scalar(zp, "PixelSize"),
        omega_start=om_start, omega_step=om_step, n_frames=n_frames,
        n_pixels_y=n_sq, n_pixels_z=n_sq,
        min_eta=_scalar(zp, "MinEta"),
        wavelength=_scalar(zp, "Wavelength"),
        tx=0.0, ty=_scalar(zp, "ty"), tz=_scalar(zp, "tz"),
        wedge=_scalar(zp, "Wedge"), flip_y=True,
    )


def build_model_from_zarr(
    zip_path: str | Path,
    hkls_csv: str | Path,
    ring_numbers: Sequence[int],
    lattice: Sequence[float],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    max_two_theta_deg: Optional[float] = None,
    scan_positions: Optional[Sequence[float]] = None,
    beam_size_um: Optional[float] = None,
) -> Tuple[HEDMForwardModel, np.ndarray]:
    """Build the forward model with geometry from a ``.MIDAS.zip``.

    The authoritative real-data path: geometry + frame mapping come from the
    zarr; ``hkls.csv`` (from the layer dir) gives the reflections. ``lattice``,
    ``ring_numbers`` come from the layer paramstest / zarr.
    """
    from midas_fit_grain.driver import _read_hkls_csv, _cartesian_B_matrix
    geom = geometry_from_zarr(zip_path)
    if max_two_theta_deg is None:
        max_two_theta_deg = 180.0
    hkls_int, thetas_deg, ring_nr = _read_hkls_csv(
        hkls_csv, list(ring_numbers), max_two_theta_deg
    )
    B = _cartesian_B_matrix(tuple(lattice))
    hkls_cart = (B @ hkls_int.astype(np.float64).T).T
    thetas_rad = thetas_deg * (math.pi / 180.0)
    scan_cfg = None
    if scan_positions is not None and beam_size_um is not None:
        scan_cfg = ScanConfig(
            beam_positions=torch.as_tensor(np.asarray(scan_positions).ravel(),
                                           dtype=dtype),
            beam_size=float(beam_size_um),
        )
    model = HEDMForwardModel(
        torch.from_numpy(hkls_cart), torch.from_numpy(thetas_rad), geom,
        hkls_int=torch.from_numpy(hkls_int.astype(np.float64)),
        scan_config=scan_cfg, device=device,
    )
    return model, ring_nr


def save_model_inputs_from_zarr(
    zip_path: str | Path,
    hkls_csv: str | Path,
    ring_numbers: Sequence[int],
    lattice: Sequence[float],
    scan_positions: Sequence[float],
    beam_size_um: float,
    out_path: str | Path,
    *,
    max_two_theta_deg: Optional[float] = None,
) -> None:
    """Persist the forward-model BUILD INPUTS (geometry + hkls) so a GPU fit on
    a machine that can't see the run dir (alleppey: scratch not mounted) can
    rebuild the model from shared home. Geometry is shared across all grains of
    a dataset, so one file suffices. Mirrors :func:`build_model_from_zarr`."""
    import dataclasses
    from midas_fit_grain.driver import _read_hkls_csv, _cartesian_B_matrix
    geom = geometry_from_zarr(zip_path)
    if max_two_theta_deg is None:
        max_two_theta_deg = 180.0
    hkls_int, thetas_deg, _ = _read_hkls_csv(hkls_csv, list(ring_numbers), max_two_theta_deg)
    B = _cartesian_B_matrix(tuple(lattice))
    hkls_cart = (B @ hkls_int.astype(np.float64).T).T
    torch.save({
        "geom": dataclasses.asdict(geom),
        "hkls_cart": torch.from_numpy(hkls_cart),
        "thetas_rad": torch.from_numpy(thetas_deg * (math.pi / 180.0)),
        "hkls_int": torch.from_numpy(hkls_int.astype(np.float64)),
        "scan_positions": torch.as_tensor(np.asarray(scan_positions).ravel(), dtype=torch.float64),
        "beam_size": float(beam_size_um),
    }, str(out_path))


def build_model_from_inputs(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> HEDMForwardModel:
    """Rebuild the forward model from a :func:`save_model_inputs` file (no zarr)."""
    d = torch.load(str(path), weights_only=False)
    geom = HEDMGeometry(**d["geom"])
    sc = None
    if d.get("scan_positions") is not None:
        sc = ScanConfig(beam_positions=d["scan_positions"].to(dtype),
                        beam_size=float(d["beam_size"]))
    hkls_int = d["hkls_int"].to(dtype) if d.get("hkls_int") is not None else None
    return HEDMForwardModel(
        d["hkls_cart"].to(dtype), d["thetas_rad"].to(dtype), geom,
        hkls_int=hkls_int, scan_config=sc, device=device,
    )


class ZarrFrameSource:
    """Lazy reader of per-scan ``.MIDAS.zip`` frames as corrected ``[y, z]``.

    Scan index ``k`` maps to file number ``start_file_nr + k * files_per_sweep``
    and the zip ``<root>/<num>/<stem>_0<num>.MIDAS.zip``. Each frame is
    square-padded, ImTransOpt-transformed, transposed and dark-subtracted using
    **midas_peakfit's own primitives** (``make_square_image`` /
    ``apply_image_transformations`` / ``transpose_square`` / ``prepare_dark``),
    so the pixel convention is bit-identical to the peak-finding / indexing path
    that produced the observed spots. The peak-finding *threshold* is
    deliberately skipped — we keep the full peak shape (tails) for fitting.
    Open stores + per-scan dark are cached.
    """

    def __init__(
        self,
        root: str | Path,
        file_stem: str,
        start_file_nr: int,
        files_per_sweep: int,
        *,
        ext: str = "MIDAS.zip",
        num_pad: int = 6,
    ) -> None:
        self.root = Path(root)
        self.file_stem = file_stem
        self.start_file_nr = int(start_file_nr)
        self.files_per_sweep = int(files_per_sweep)
        self.ext = ext
        self.num_pad = num_pad
        self._stores: Dict[int, object] = {}
        self._dark: Dict[int, np.ndarray] = {}
        self._geom: Optional[Tuple[int, int, int, List[int]]] = None  # NrPixels, Y, Z, TransOpt

    def _zip_path(self, scan_idx: int) -> Path:
        num = self.start_file_nr + scan_idx * self.files_per_sweep
        return self.root / str(num) / f"{self.file_stem}_{num:0{self.num_pad+1}d}.{self.ext}"

    def _open(self, scan_idx: int):
        if scan_idx not in self._stores:
            import zarr
            from midas_peakfit.preprocess import prepare_dark
            z = zarr.open(zarr.ZipStore(str(self._zip_path(scan_idx)), mode="r"),
                          mode="r")
            self._stores[scan_idx] = z
            if self._geom is None:
                ap = z["analysis/process/analysis_parameters"]
                ny = int(np.array(ap["numPxY"]).ravel()[0])
                nz = int(np.array(ap["numPxZ"]).ravel()[0])
                nsq = max(ny, nz)
                ito = np.array(ap["ImTransOpt"]).ravel().astype(int).tolist()
                self._geom = (nsq, ny, nz, ito)
            nsq, ny, nz, ito = self._geom
            dark_raw = np.asarray(z["exchange/dark"][:]).astype(np.float64)
            self._dark[scan_idx] = prepare_dark(dark_raw, nsq, ny, nz, ito)
        return self._stores[scan_idx]

    def __call__(self, scan_idx: int, frame_idx: int) -> Optional[np.ndarray]:
        from midas_peakfit.preprocess import (
            make_square_image, apply_image_transformations, transpose_square,
        )
        z = self._open(scan_idx)
        nsq, ny, nz, ito = self._geom
        data = z["exchange/data"]
        if frame_idx < 0 or frame_idx >= data.shape[0]:
            return None
        raw = np.asarray(data[frame_idx]).astype(np.float64)        # (nz, ny)
        sq = make_square_image(raw, nsq, ny, nz)
        img = transpose_square(apply_image_transformations(sq, ito))  # [y, z]
        return img - self._dark[scan_idx]


def zarr_frame_reader(
    scan_zarr_paths: Sequence[str | Path],
    dataset_key: str = "exchange/data",
) -> Callable[[int, int], np.ndarray]:
    """Build a ``frame_reader(scan_idx, frame_idx) -> 2D ndarray`` over per-scan
    zarr files, for use with :func:`assemble_grain_patch_data`.

    ``scan_zarr_paths`` is one zarr store per scan position, in scan order
    (``scan_idx`` indexes into it). Each store is opened once and cached. Reading
    from the Blosc-compressed MIDAS zarr (``*.MIDAS.zip``, ``exchange/data``) is
    ~2× less disk I/O than the raw uncompressed h5 (byte-identical frames), which
    dominates pf-HEDM patch extraction.

    Example
    -------
    >>> paths = sorted(Path(raw_farm).glob("*/Ni_..._[0-9]*.MIDAS.zip"))
    >>> reader = zarr_frame_reader(paths)
    >>> data = assemble_grain_patch_data(dataset, frame_reader=reader)
    """
    import zarr

    paths = [str(p) for p in scan_zarr_paths]
    cache: dict[int, object] = {}

    def reader(scan_idx: int, frame_idx: int) -> np.ndarray:
        d = cache.get(scan_idx)
        if d is None:
            d = zarr.open(paths[scan_idx], mode="r")[dataset_key]
            cache[scan_idx] = d
        return np.asarray(d[frame_idx])

    return reader


def crop_patches_from_frames(
    frame_reader: Callable[[int, int], np.ndarray],
    anchor_y: torch.Tensor,             # (S,) pixel
    anchor_z: torch.Tensor,             # (S,) pixel
    anchor_f: torch.Tensor,             # (S,) fractional frame
    n_scans: int,
    spot_observed: torch.Tensor,        # (S,) bool
    *,
    patch_F: int,
    patch_P: int,
    n_pixels_y: int,
    n_pixels_z: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Crop measured ``(S, Σ, F, P, P)`` patches from raw detector frames.

    ``frame_reader(scan_idx, frame_idx)`` must return a 2D ``(n_y, n_z)``
    detector frame. Each (spot, scan) patch is a ``(F, P, P)`` window centred
    on the spot's anchor frame and (y, z) pixel; out-of-range reads are
    zero-padded. Unobserved spots get zero patches. The y/z crop axes follow
    the detector array convention ``frame[y, z]``; callers must ensure the
    ``flip_y`` / ImTransOpt handling matches how the geometry produced the
    anchors (validated against observed spots on real data).
    """
    S = int(anchor_y.numel())
    halfP = patch_P // 2
    halfF = patch_F // 2
    out = torch.zeros(S, n_scans, patch_F, patch_P, patch_P,
                      dtype=dtype, device=device)
    ay = anchor_y.round().to(torch.int64).tolist()
    az = anchor_z.round().to(torch.int64).tolist()
    af = anchor_f.round().to(torch.int64).tolist()
    obs = spot_observed.tolist()
    for s in range(S):
        if not obs[s]:
            continue
        y0, z0, f0 = ay[s], az[s], af[s]
        for sigma in range(n_scans):
            for fi in range(patch_F):
                fr = f0 + (fi - halfF)
                if fr < 0:
                    continue
                try:
                    frame = frame_reader(sigma, fr)
                except (IndexError, KeyError):
                    continue
                if frame is None:
                    continue
                ylo, yhi = y0 - halfP, y0 - halfP + patch_P
                zlo, zhi = z0 - halfP, z0 - halfP + patch_P
                ys0, ys1 = max(ylo, 0), min(yhi, n_pixels_y)
                zs0, zs1 = max(zlo, 0), min(zhi, n_pixels_z)
                if ys0 >= ys1 or zs0 >= zs1:
                    continue
                patch = torch.as_tensor(
                    np.asarray(frame)[ys0:ys1, zs0:zs1], dtype=dtype, device=device
                )
                out[s, sigma, fi, ys0 - ylo:ys1 - ylo, zs0 - zlo:zs1 - zlo] = patch
    return out
