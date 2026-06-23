"""Generic differentiable forward model for all HEDM modalities (NF, FF, pf).

The core Bragg geometry, omega solver, and detector projection are identical
across modalities. Modality differences (detector distance, output mode, scan
strategy) are handled via configuration, not subclassing.

Physics pipeline:
    euler_angles, positions [, lattice_params]
        -> orientation matrices             (euler2mat)
        -> G-vectors in crystal frame       (calc_bragg_geometry)
        -> [optional: strained G-vectors]   (correct_hkls_latc)
        -> omega solver (quadratic)         (calc_bragg_geometry)
        -> eta computation                  (calc_bragg_geometry)
        -> position-dependent projection    (project_to_detector)
        -> validity filtering               (project_to_detector)
        -> SpotDescriptors

Output modes:
    SpotDescriptors -> predict_images()       [NF: Gaussian splatting]
    SpotDescriptors -> predict_spot_coords()  [FF/pf: angular coordinates]

Reference C code:
    CorrectHKLsLatC:       FF_HEDM/src/FitPosOrStrainsDoubleDataset.c:214-252
    CalcDiffrSpots_Furnace: NF_HEDM/src/CalcDiffractionSpots.c:87-183
    DisplacementSpots:      NF_HEDM/src/SharedFuncsFit.c:269-292
    Beam proximity filter:  FF_HEDM/src/FitOrStrainsScanningOMP.c:1048-1058
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical orientation + strain-frame primitives. midas_stress is the single
# source of truth for this math (Bunge ZXZ orientation algebra, sample<->crystal
# strain rotation); its torch backend is differentiable end-to-end and
# device-portable, so the forward model delegates rather than re-porting.
# NOTE: midas_stress's Voigt is Voigt-MANDEL (sqrt2 on shears); this model uses
# PLAIN-Voigt / raw 3x3 strain, so we only delegate the *rotation*, never the
# Voigt packing (see rotate_strain_sample_to_crystal / correct_hkls_latc).
from midas_stress.orientation import euler_to_orient_mat as _ms_euler_to_orient_mat
from midas_stress.tensor import strain_lab_to_grain as _ms_strain_lab_to_grain


# ---------------------------------------------------------------------------
#  Configuration data classes
# ---------------------------------------------------------------------------

@dataclass
class HEDMGeometry:
    """Detector and scan geometry configuration.

    Units: distances in micrometers, angles in degrees (converted internally).

    NF-HEDM uses multiple detector distances (nDistances=2-4), each with its
    own Lsd, y_BC, z_BC.  FF-HEDM and pf-HEDM use a single distance.
    When lists are provided for Lsd/y_BC/z_BC, each entry is one "layer"
    (detector distance).  A spot is valid only if it falls on the detector
    at **every** distance (the AllDistsFound logic in the C code).
    """
    Lsd: "float | list[float]"     # Sample-detector distance(s) (um)
    y_BC: "float | list[float]"    # Beam center y (pixels), per distance/detector
    z_BC: "float | list[float]"    # Beam center z (pixels), per distance/detector
    px: float                      # Pixel size (um) -- shared
    omega_start: float             # Omega start (degrees)
    omega_step: float              # Omega step (degrees, may be negative)
    n_frames: int                  # Frames per distance (NrFilesPerDistance)
    n_pixels_y: int                # Detector pixels in y
    n_pixels_z: int                # Detector pixels in z
    min_eta: float                 # Minimum eta angle (degrees)
    wavelength: float = 0.0        # X-ray wavelength (Angstroms)
    # Detector tilts (degrees). Applied only in NF mode (flip_y=False) by
    # default; FF/pf workflows apply a DetCor correction at peak-finding
    # time so their centroids are already tilt-corrected, and the forward
    # model ignores tilts in FF mode to avoid double-correcting. To override
    # this (e.g. to simulate raw multi-panel FF data with no DetCor), set
    # ``apply_tilts=True`` -- tilts are then applied in FF mode too.
    #
    # Each tilt field accepts either a scalar (shared across all
    # distances/detectors) or a list of length ``n_distances`` (one value
    # per detector). Combined with ``multi_mode="panel"`` this enables
    # full multi-panel FF-HEDM forward simulation.
    tx: "float | list[float]" = 0.0
    ty: "float | list[float]" = 0.0
    tz: "float | list[float]" = 0.0
    flip_y: bool = True            # FF/PF: True (DetHor = yBC - ydet/px).
                                   # NF:    False (pixel = yBC + ydet/px).
                                   # Validated against C code conventions.
    wedge: float = 0.0             # Wedge angle (deg): non-orthogonality
                                   # of the rotation axis and the X-ray beam.
                                   # Single global value (not per-detector).
                                   # Wedge=0 means rotation axis ⟂ beam.
                                   # Implementation matches CorrectWedge() in
                                   # FF_HEDM/src/ForwardSimulationCompressed.c.
    apply_tilts: bool = False      # Force tilt application even in FF mode.
                                   # Default False preserves the existing
                                   # behaviour: NF applies tilts, FF skips.
                                   # Set True for raw multi-panel simulation.
    # Radial detector distortion (canonical midas_distortion v2 model). Like
    # tilts, this maps an IDEAL prediction to the RAW detector position and is
    # OFF by default -- the FF/pf experimental pipeline pre-corrects distortion
    # at peak-finding time (transforms), so the forward must NOT re-apply it for
    # the indexer/fit-grain. Raw-pixel-patch consumers (pf_odf, grain_odf) that
    # never go through transforms set ``apply_distortion=True`` and supply the
    # calibrated v2 coefficients to predict in the raw frame.
    apply_distortion: bool = False
    p_distortion: "list[float] | None" = None   # 15 v2 coeffs (midas_distortion P_COEF_NAMES order); None/zeros => no-op
    rho_d: "float | None" = None                # distortion radius normalization (um); None => resolve from detector corner
    multi_mode: str = "layered"    # "layered" (default): NF semantics --
                                   # spot must land on the detector at
                                   # EVERY distance (AllDistsFound).
                                   # "panel": FF multi-detector semantics --
                                   # spot is valid if it lands on at least
                                   # one detector; output gains a det_id
                                   # field naming which one.

    @property
    def n_distances(self) -> int:
        return len(self.Lsd) if isinstance(self.Lsd, list) else 1

    @property
    def n_detectors(self) -> int:
        """Alias of ``n_distances`` for FF multi-panel readability."""
        return self.n_distances

    def _as_list(self, attr):
        v = getattr(self, attr)
        if isinstance(v, list):
            return v
        # Broadcast scalar across n_distances so per-detector code paths can
        # always assume a list-of-N. For scalar attrs that have not yet been
        # broadcast, this yields a single-element list (n_distances=1).
        return [v] * self.n_distances


@dataclass
class ScanConfig:
    """Multi-scan configuration for pf-HEDM (beam translation positions).

    In pf-HEDM the pencil beam is translated to different Y-positions.
    Each scan is a full omega sweep at one beam Y-position.
    NF-HEDM and FF-HEDM do NOT use this (they have a single beam position).
    """
    beam_positions: torch.Tensor  # (S,) beam y-positions per scan (um)
    beam_size: float              # Beam height (um)


@dataclass
class TriVoxelConfig:
    """Triangular voxel grid configuration for NF-HEDM.

    NF-HEDM uses equilateral triangle voxels.  Each voxel is defined by
    a center ``(x, y)``, an ``edge_length``, and an up/down flag ``ud``.
    The three vertices are computed as:

    .. code-block:: text

        gs = edge_length / 2
        dy1 =  edge_length / sqrt(3)   (flipped if ud < 0)
        dy2 = -edge_length / (2*sqrt(3))

        V0 = (x,      y + dy1)
        V1 = (x - gs,  y + dy2)
        V2 = (x + gs,  y + dy2)

    Matches ``simulateNF.c`` lines 556-572.
    """
    edge_lengths: torch.Tensor   # (N,) per-voxel edge length in um
    ud: torch.Tensor             # (N,) up/down flag (+1 or -1)


@dataclass
class SpotDescriptors:
    """Output of the forward model: all information about predicted spots.

    All angular quantities are in radians.  Pixel coordinates are fractional.
    Shape convention: ``(..., K, M)`` where ``K = 2*N`` (two omega solutions
    per position) and ``M`` = number of HKL reflections.

    For multi-distance NF-HEDM (``multi_mode="layered"``), ``y_pixel``,
    ``z_pixel``, and ``layer_valid`` have an extra leading ``D``
    (n_distances) dimension; ``valid`` combines the angular validity with
    ALL-distances-found.

    For multi-panel FF-HEDM (``multi_mode="panel"``), ``y_pixel``,
    ``z_pixel`` are collapsed to a single ``(..., K, M)`` tensor giving the
    pixel coordinates on the panel where each spot landed; ``det_id`` names
    that panel (int64 in [0, D)). ``valid`` is True if the spot landed on
    at least one panel.
    """
    omega: torch.Tensor                      # (..., K, M) radians
    eta: torch.Tensor                        # (..., K, M) radians
    two_theta: torch.Tensor                  # (..., K, M) radians
    y_pixel: torch.Tensor                    # (D, ..., K, M) layered, or (..., K, M) single-distance / panel
    z_pixel: torch.Tensor                    # (D, ..., K, M) layered, or (..., K, M) single-distance / panel
    frame_nr: torch.Tensor                   # (..., K, M) fractional frame (same at all distances)
    valid: torch.Tensor                      # (..., K, M) float mask
    layer_valid: Optional[torch.Tensor] = None  # (D, ..., K, M) per-distance validity (layered mode only)
    scan_mask: Optional[torch.Tensor] = None    # (..., S, K, M) per-beam-position validity (pf)
    det_id: Optional[torch.Tensor] = None       # (..., K, M) int64 panel index (panel mode only)


# ---------------------------------------------------------------------------
#  Forward model
# ---------------------------------------------------------------------------

class HEDMForwardModel(nn.Module):
    """Generic differentiable forward model for NF / FF / pf-HEDM.

    Parameters
    ----------
    hkls : Tensor (M, 3)
        Reciprocal-space G-vectors in Cartesian coordinates (1/Angstroms).
        These are the *nominal* (unstrained) G-vectors, already transformed
        through the B matrix for the reference lattice.
    thetas : Tensor (M,)
        Nominal Bragg angles in radians corresponding to ``hkls``.
    geometry : HEDMGeometry
        Detector / scan geometry.
    hkls_int : Tensor (M, 3), optional
        Integer Miller indices. Required for ``correct_hkls_latc`` (strain).
        If None, strain correction is unavailable.
    scan_config : ScanConfig, optional
        Multi-scan configuration. None for single-scan (standard NF / FF).
    device : torch.device
        Target device.
    """

    # Match the C code: both now use M_PI/180.0 (previously the C code
    # used hardcoded 13-digit constants that caused precision loss).
    DEG2RAD = math.pi / 180.0
    RAD2DEG = 180.0 / math.pi

    def __init__(
        self,
        hkls: torch.Tensor,
        thetas: torch.Tensor,
        geometry: HEDMGeometry,
        hkls_int: Optional[torch.Tensor] = None,
        scan_config: Optional[ScanConfig] = None,
        device: torch.device = torch.device("cpu"),
        compile: "bool | str" = False,
    ):
        """Build the forward model.

        Parameters
        ----------
        compile : bool | str, optional
            If truthy and ``device.type == "cuda"``, wrap the forward path
            with ``torch.compile``. Pass ``True`` for the default mode
            ("reduce-overhead", best for repeated fixed-shape calls) or a
            string mode such as ``"max-autotune"``. Bit-exact agreement
            against the eager path was validated on A6000 at fp64; CPU and
            MPS callers see no benefit and the flag is ignored. The first
            forward call after construction pays the JIT-trace cost; all
            subsequent calls run from cache.
        """
        super().__init__()

        self.register_buffer("hkls", hkls.to(device).float())
        self.register_buffer("thetas", thetas.to(device).float())

        if hkls_int is not None:
            self.register_buffer("hkls_int", hkls_int.to(device).float())
        else:
            self.hkls_int = None

        # Geometry -- per-distance arrays stored as tensors for vectorised projection
        Lsd_list = geometry._as_list("Lsd")
        yBC_list = geometry._as_list("y_BC")
        zBC_list = geometry._as_list("z_BC")
        self.n_distances = len(Lsd_list)
        # Per-detector geometry stored as nn.Parameter (requires_grad=False
        # by default) so callers can opt them into gradient-based refinement.
        self._Lsd = nn.Parameter(
            torch.tensor(Lsd_list, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self._y_BC = nn.Parameter(
            torch.tensor(yBC_list, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self._z_BC = nn.Parameter(
            torch.tensor(zBC_list, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        # Lsd reparameterisation for the optimiser.
        #
        # ``_Lsd`` lives in micrometres (~1e6) for downstream code-clarity,
        # but joint optimisers see a search direction whose step length is
        # set by the *smallest* gradient-magnitude parameter -- ~mm-scale
        # tilt-error gradient -- and that step length is far too small to
        # move ``_Lsd`` meaningfully. We expose a separate
        # ``_Lsd_delta_mm`` parameter (initially zero) that the user can
        # opt into refinement instead of ``_Lsd`` itself, and the effective
        # Lsd used by the forward is ``_Lsd + 1000 * _Lsd_delta_mm``.
        # This is a clean reparameterisation: the optimiser sees an
        # mm-scale variable, the physics still uses um, and there is no
        # need for gradient hooks. ``_Lsd_delta_mm`` defaults to zero so
        # behaviour is unchanged when callers ignore it.
        self._Lsd_delta_mm = nn.Parameter(
            torch.zeros_like(self._Lsd), requires_grad=False
        )
        # Convenience: the same mm-scale reparameterisation idea is *not*
        # applied to ``_y_BC`` or ``_z_BC`` because they are already in
        # pixels (~ 1e3) and at a scale where the optimiser handles them
        # directly without preconditioning. Tilts and grain Eulers are
        # likewise in their natural units.

        # Convenience aliases for single-distance (backward compat / simple access)
        self.Lsd = Lsd_list[0]
        self.y_BC = yBC_list[0]
        self.z_BC = zBC_list[0]
        self.px = geometry.px
        self.omega_start = geometry.omega_start
        self.omega_step = geometry.omega_step
        self.n_frames = geometry.n_frames
        self.n_pixels_y = geometry.n_pixels_y
        self.n_pixels_z = geometry.n_pixels_z
        self.min_eta = geometry.min_eta * self.DEG2RAD  # store in radians
        self.wavelength = geometry.wavelength
        self.flip_y = geometry.flip_y

        # Per-detector tilts (degrees), shape (D, 3) -- one (tx, ty, tz)
        # row per detector. Stored as an nn.Parameter so the user can opt
        # them into gradient-based refinement (auto-calibration).  NF mode
        # (flip_y=False) always applies them via _apply_tilt; FF/pf mode
        # (flip_y=True) skips them by default because the experimental
        # pipeline pre-corrects detector tilts at peak-finding time. Set
        # ``geometry.apply_tilts=True`` to force tilt application in FF
        # mode -- needed to simulate raw multi-panel FF data with no DetCor.
        #
        # Composition: RotMatTilts = Rz(tz) @ Ry(ty) @ Rx(tx)
        # Matches RotationTilts() in NF_HEDM/src/SharedFuncsFit.c:230-266.
        tx_list = geometry._as_list("tx")
        ty_list = geometry._as_list("ty")
        tz_list = geometry._as_list("tz")
        if not (len(tx_list) == len(ty_list) == len(tz_list) == self.n_distances):
            raise ValueError(
                "tx/ty/tz lists must each have length n_distances "
                f"(got {len(tx_list)}, {len(ty_list)}, {len(tz_list)} "
                f"vs n_distances={self.n_distances})"
            )
        tilts_arr = [[float(tx_list[d]), float(ty_list[d]), float(tz_list[d])]
                     for d in range(self.n_distances)]
        self.tilts = nn.Parameter(
            torch.tensor(tilts_arr, dtype=torch.float64, device=device),
            requires_grad=False,
        )
        # Convenience scalar aliases for single-detector backward compat.
        # For multi-detector configurations these expose detector 0 only.
        self.tx = float(tx_list[0])
        self.ty = float(ty_list[0])
        self.tz = float(tz_list[0])
        self._has_tilts = bool(
            torch.any(torch.abs(self.tilts.detach()) > 0.0).item()
        )

        # Multi-detector / multi-panel configuration
        self.apply_tilts = bool(geometry.apply_tilts)

        # Radial detector distortion (canonical midas_distortion v2 model),
        # applied ideal->raw in project_to_detector when apply_distortion=True.
        # OFF by default => identity => indexer/fit-grain output unchanged.
        self.apply_distortion = bool(getattr(geometry, "apply_distortion", False))
        p_dist = getattr(geometry, "p_distortion", None)
        if p_dist is not None:
            self.p_distortion = nn.Parameter(
                torch.as_tensor(p_dist, dtype=torch.float64, device=device),
                requires_grad=False,
            )
            self._has_distortion = bool(
                torch.any(torch.abs(self.p_distortion.detach()) > 0.0).item()
            )
        else:
            self.p_distortion = None
            self._has_distortion = False
        self.rho_d = getattr(geometry, "rho_d", None)

        if geometry.multi_mode not in ("layered", "panel"):
            raise ValueError(
                f"Unknown multi_mode {geometry.multi_mode!r}; "
                "use 'layered' (NF) or 'panel' (FF multi-detector)."
            )
        self.multi_mode = geometry.multi_mode

        # Wedge angle (deg): single global parameter representing the
        # non-orthogonality of the rotation axis and the beam. Stored as
        # nn.Parameter so it can be jointly refined.
        self.wedge = nn.Parameter(
            torch.tensor(float(geometry.wedge), dtype=torch.float64,
                          device=device),
            requires_grad=False,
        )
        self._has_wedge = abs(float(geometry.wedge)) > 0.0

        # Scan config
        self.scan_config = scan_config
        if scan_config is not None:
            self.register_buffer(
                "_beam_positions", scan_config.beam_positions.to(device).float()
            )
            self._beam_size = scan_config.beam_size

        self.epsilon = 1e-7

        # Optional torch.compile of the forward path. We wrap the bound
        # ``forward`` method (not the module) and stash it on the instance;
        # ``__call__`` -> ``self.forward(...)`` then routes through the
        # compiled callable on every invocation. Skipped on CPU/MPS where
        # the inductor backend offers no GPU-style fusion benefit and only
        # adds trace cost.
        #
        # Default mode is ``"default"`` (Inductor without CUDA Graphs).
        # ``"reduce-overhead"`` is materially faster (Track A measured
        # 17-19x speedup at small N vs ~2-3x for default), but its
        # CUDA-Graph backing assumes the caller does NOT alias outputs
        # across calls -- e.g. ``a = model(...); b = model(...)`` then
        # using ``a`` and ``b`` together raises an aliasing error. Iterative
        # optimisers and inversion loops violate that, so the safer
        # ``"default"`` is the right out-of-the-box pick. Users with
        # alias-clean inner loops can opt into ``compile="reduce-overhead"``.
        self._compile_mode: Optional[str] = None
        # Some callers pass device as a string ("cuda", "cpu"); normalise.
        _dev = torch.device(device) if isinstance(device, str) else device
        if compile and _dev.type == "cuda":
            mode = "default" if compile is True else str(compile)
            self._compile_mode = mode
            eager_forward = self.forward
            self.forward = torch.compile(eager_forward, mode=mode, dynamic=False)

    @property
    def _Lsd_eff(self) -> torch.Tensor:
        """Effective Lsd in micrometres, including the mm-scale optimiser
        offset. Use this in the forward path; never compute the projection
        from ``self._Lsd`` directly when refinement is active.
        """
        return self._Lsd + 1000.0 * self._Lsd_delta_mm

    # ------------------------------------------------------------------
    #  euler2mat  (ZXZ convention)
    # ------------------------------------------------------------------

    @staticmethod
    def euler2mat(euler_angles: torch.Tensor) -> torch.Tensor:
        """Convert ZXZ (Bunge) Euler angles to crystal->sample rotation matrices.

        Delegates to ``midas_stress.orientation.euler_to_orient_mat`` -- the
        canonical orientation primitive -- so the convention can never drift
        from the rest of MIDAS. midas_stress's torch backend is differentiable
        and vmap-safe; the result is identical to the former in-line ZXZ build
        (R = Rz(phi1) @ Rx(Phi) @ Rz(phi2)) to ~1e-16.

        Parameters
        ----------
        euler_angles : Tensor (..., 3)
            Euler angles (phi1, Phi, phi2) in radians.

        Returns
        -------
        Tensor (..., 3, 3)
            Rotation matrices (crystal->sample), orthogonalized onto SO(3).
        """
        if not isinstance(euler_angles, torch.Tensor):
            euler_angles = torch.as_tensor(euler_angles)
        R = _ms_euler_to_orient_mat(euler_angles)          # (..., 9), torch
        R = R.reshape(*R.shape[:-1], 3, 3)
        # midas_stress already returns a proper rotation; orthogonalize keeps the
        # historical "exactly on SO(3)" guarantee and is idempotent here.
        return HEDMForwardModel.orthogonalize(R)

    # ------------------------------------------------------------------
    #  orthogonalize  (SO(3) projection via SVD)
    # ------------------------------------------------------------------

    @staticmethod
    def orthogonalize(R: torch.Tensor) -> torch.Tensor:
        """Project a (..., 3, 3) matrix onto SO(3).

        Guarantees ``R^T R = I`` and ``det(R) = +1`` (proper rotation).
        Uses one Newton-Schulz iteration which is differentiable and
        numerically stable (no SVD singularity at repeated singular values).

        For matrices already near SO(3) (like those from ``euler2mat``),
        a single iteration suffices (quadratic convergence).

        Parameters
        ----------
        R : Tensor (..., 3, 3)

        Returns
        -------
        Tensor (..., 3, 3) -- orthogonal with det = +1.
        """
        # Newton-Schulz iteration for polar decomposition:
        #   Q_{k+1} = 0.5 * Q_k * (3*I - Q_k^T * Q_k)
        # Converges quadratically for matrices near SO(3).
        # One iteration is sufficient for matrices from euler2mat
        # (error < 1e-14 after one step for input error < 1e-7).
        I = torch.eye(3, dtype=R.dtype, device=R.device)
        Q = R
        Q = 0.5 * Q @ (3.0 * I - Q.transpose(-1, -2) @ Q)
        # Ensure det = +1 (the iteration preserves det sign for near-SO(3) input,
        # but we guard against pathological cases)
        det = torch.det(Q)
        # Where det < 0, negate (flips to proper rotation)
        sign = torch.where(det < 0, torch.tensor(-1.0, dtype=R.dtype, device=R.device),
                           torch.tensor(1.0, dtype=R.dtype, device=R.device))
        return Q * sign.unsqueeze(-1).unsqueeze(-1)

    # ------------------------------------------------------------------
    #  safe_arccos
    # ------------------------------------------------------------------

    def safe_arccos(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable arccos: clamp to [-1+eps, 1-eps]."""
        return torch.acos(torch.clamp(x, -1.0 + self.epsilon, 1.0 - self.epsilon))

    # ------------------------------------------------------------------
    #  strain_as_voigt  (accept full 3x3 tensor OR plain-Voigt 6-vector)
    # ------------------------------------------------------------------

    @staticmethod
    def strain_as_voigt(strain: torch.Tensor) -> torch.Tensor:
        """Normalize a strain input to PLAIN-Voigt [e11,e12,e13,e22,e23,e33].

        Accepts either a plain-Voigt ``(..., 6)`` tensor (returned unchanged) or
        a full symmetric ``(..., 3, 3)`` strain tensor. The 3x3 path is
        convention-free -- the natural way to hand a strain field straight from
        a tensor source (e.g. a midas_stress strain field) into the forward
        model without picking a Voigt/Mandel packing. The off-diagonals are
        taken as TRUE tensor components (no factor of 2).
        """
        if strain.dim() >= 2 and strain.shape[-1] == 3 and strain.shape[-2] == 3:
            return torch.stack([
                strain[..., 0, 0], strain[..., 0, 1], strain[..., 0, 2],
                strain[..., 1, 1], strain[..., 1, 2], strain[..., 2, 2],
            ], dim=-1)
        return strain

    # ------------------------------------------------------------------
    #  rotate_strain_sample_to_crystal  (port of C RotateStrainSampleToCrystal)
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_strain_sample_to_crystal(
        orientation_matrices: torch.Tensor,
        strain_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate a symmetric infinitesimal strain from sample to crystal frame.

        Matches ``RotateStrainSampleToCrystal`` from
        ``FF_HEDM/src/ForwardSimulationCompressed.c:399-419``:
        eps_crystal = OM^T . eps_sample . OM.

        The rotation is delegated to ``midas_stress.tensor.strain_lab_to_grain``
        (the canonical sample/lab -> crystal/grain strain transform; bit-identical
        to the former in-line ``OM^T S OM``). PLAIN-Voigt pack/unpack is kept here
        on purpose -- midas_stress's Voigt is Mandel (sqrt2 on shears), which must
        not touch the forward model's strain input.

        Parameters
        ----------
        orientation_matrices : Tensor (..., 3, 3)
            Crystal->sample matrices (as returned by :meth:`euler2mat`).
        strain_sample : Tensor (..., 6)
            PLAIN-Voigt [eps_11, eps_12, eps_13, eps_22, eps_23, eps_33].

        Returns
        -------
        strain_crystal : Tensor (..., 6)
            PLAIN-Voigt, same layout.
        """
        e = strain_sample
        S = torch.stack([
            torch.stack([e[..., 0], e[..., 1], e[..., 2]], dim=-1),
            torch.stack([e[..., 1], e[..., 3], e[..., 4]], dim=-1),
            torch.stack([e[..., 2], e[..., 4], e[..., 5]], dim=-1),
        ], dim=-2)
        C = _ms_strain_lab_to_grain(S, orientation_matrices)   # OM^T S OM
        return torch.stack([
            C[..., 0, 0], C[..., 0, 1], C[..., 0, 2],
            C[..., 1, 1], C[..., 1, 2], C[..., 2, 2],
        ], dim=-1)

    # ------------------------------------------------------------------
    #  correct_hkls_latc  (port of C CorrectHKLsLatC)
    # ------------------------------------------------------------------

    def correct_hkls_latc(
        self,
        lattice_params: torch.Tensor,
        strain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute strained reciprocal-space G-vectors and Bragg angles.

        Builds the reciprocal lattice B matrix from lattice parameters
        and transforms integer Miller indices to Cartesian G-vectors.

        Faithfully ports ``CorrectHKLsLatC`` from
        ``FF_HEDM/src/FitPosOrStrainsDoubleDataset.c:214-252``, with the
        optional crystal-frame strain path from ``CorrectHKLsLatCEpsilon``
        in ``FF_HEDM/src/ForwardSimulationCompressed.c:423-475``.

        Parameters
        ----------
        lattice_params : Tensor (..., 6)
            [a, b, c, alpha, beta, gamma] in Angstroms and degrees.
            The ``...`` dimensions allow per-voxel or per-grain parameters.
        strain : Tensor (..., 6) or (..., 3, 3), optional
            Crystal-frame symmetric infinitesimal strain, either PLAIN-Voigt
            [eps_11, eps_12, eps_13, eps_22, eps_23, eps_33] or a full symmetric
            3x3 tensor (normalized via :meth:`strain_as_voigt`; off-diagonals are
            true tensor components, no factor of 2). When supplied, the
            reciprocal lattice is post-multiplied by (I + eps)^{-1}:
            B = (I + eps)^{-1} @ B0. Use :meth:`rotate_strain_sample_to_crystal`
            to convert a sample-frame strain into the crystal frame.

        Returns
        -------
        hkls_cart : Tensor (..., M, 3)
            G-vectors in Cartesian reciprocal space (1/Angstroms).
        thetas : Tensor (..., M)
            Bragg angles in radians.

        Raises
        ------
        RuntimeError
            If ``hkls_int`` was not provided at construction.
        """
        if self.hkls_int is None:
            raise RuntimeError(
                "correct_hkls_latc requires integer Miller indices "
                "(pass hkls_int to the constructor)."
            )

        a = lattice_params[..., 0]
        b = lattice_params[..., 1]
        c = lattice_params[..., 2]
        # Angles in degrees -> radians
        alpha = lattice_params[..., 3] * self.DEG2RAD
        beta = lattice_params[..., 4] * self.DEG2RAD
        gamma = lattice_params[..., 5] * self.DEG2RAD

        sin_a = torch.sin(alpha)
        cos_a = torch.cos(alpha)
        sin_b = torch.sin(beta)
        cos_b = torch.cos(beta)
        sin_g = torch.sin(gamma)
        cos_g = torch.cos(gamma)

        # Reciprocal lattice angles
        # C: GammaPr = acosd((CosA*CosB - CosG) / (SinA*SinB))
        gamma_pr = torch.acos(
            torch.clamp((cos_a * cos_b - cos_g) / (sin_a * sin_b + self.epsilon),
                        -1.0 + self.epsilon, 1.0 - self.epsilon)
        )
        beta_pr = torch.acos(
            torch.clamp((cos_g * cos_a - cos_b) / (sin_g * sin_a + self.epsilon),
                        -1.0 + self.epsilon, 1.0 - self.epsilon)
        )
        sin_beta_pr = torch.sin(beta_pr)

        # Volume and reciprocal lengths
        vol = a * b * c * sin_a * sin_beta_pr * sin_g
        a_pr = b * c * sin_a / (vol + self.epsilon)
        b_pr = c * a * sin_b / (vol + self.epsilon)
        c_pr = a * b * sin_g / (vol + self.epsilon)

        # Build B matrix (..., 3, 3)
        zeros = torch.zeros_like(a)
        # Row 0
        B00 = a_pr
        B01 = b_pr * torch.cos(gamma_pr)
        B02 = c_pr * torch.cos(beta_pr)
        # Row 1
        B10 = zeros
        B11 = b_pr * torch.sin(gamma_pr)
        B12 = -c_pr * sin_beta_pr * cos_a
        # Row 2
        B20 = zeros
        B21 = zeros
        B22 = c_pr * sin_beta_pr * sin_a

        # Stack into (..., 3, 3)
        B = torch.stack([
            torch.stack([B00, B01, B02], dim=-1),
            torch.stack([B10, B11, B12], dim=-1),
            torch.stack([B20, B21, B22], dim=-1),
        ], dim=-2)

        # Optional crystal-frame strain: B = (I + eps)^{-1} @ B0
        # Voigt layout matches C CorrectHKLsLatCEpsilon:
        #   eps = [eps_11, eps_12, eps_13, eps_22, eps_23, eps_33]
        if strain is not None:
            strain = self.strain_as_voigt(strain)   # accept full 3x3 too
            e11 = strain[..., 0]
            e12 = strain[..., 1]
            e13 = strain[..., 2]
            e22 = strain[..., 3]
            e23 = strain[..., 4]
            e33 = strain[..., 5]
            one = torch.ones_like(e11)
            F_mat = torch.stack([
                torch.stack([one + e11, e12,        e13       ], dim=-1),
                torch.stack([e12,       one + e22, e23       ], dim=-1),
                torch.stack([e13,       e23,        one + e33], dim=-1),
            ], dim=-2)
            F_inv = torch.linalg.inv(F_mat)
            B = torch.matmul(F_inv, B)

        # G_cart = B @ hkls_int^T  =>  (..., M, 3)
        # hkls_int is (M, 3), B is (..., 3, 3) -- match dtype
        hkls_cart = torch.einsum("...ij,mj->...mi", B, self.hkls_int.to(B.dtype))

        # d-spacing = 1 / |G_cart|
        g_norm = torch.norm(hkls_cart, dim=-1).clamp(min=self.epsilon)
        d_spacing = 1.0 / g_norm

        # Bragg angle: theta = arcsin(wavelength / (2*d))
        sin_theta = (self.wavelength / (2.0 * d_spacing)).clamp(
            -1.0 + self.epsilon, 1.0 - self.epsilon
        )
        thetas = torch.asin(sin_theta)

        return hkls_cart, thetas

    # ------------------------------------------------------------------
    #  calc_bragg_geometry  (omega quadratic solver + eta)
    # ------------------------------------------------------------------

    def calc_bragg_geometry(
        self,
        orientation_matrices: torch.Tensor,
        hkls_cart: Optional[torch.Tensor] = None,
        thetas: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core Bragg geometry: orientations + G-vectors -> angles.

        Solves the omega quadratic from the diffraction condition and
        computes the eta azimuthal angle.

        Ports the quadratic solver from
        ``NF_HEDM/src/CalcDiffractionSpots.c:87-183``.

        Parameters
        ----------
        orientation_matrices : Tensor (..., N, 3, 3)
            Rotation matrices for each voxel/grain.
        hkls_cart : Tensor (..., M, 3) or None
            G-vectors in Cartesian reciprocal space. If None, uses
            the nominal ``self.hkls``.
        thetas : Tensor (..., M) or None
            Bragg angles in radians. If None, uses ``self.thetas``.

        Returns
        -------
        omega : Tensor (..., 2N, M) -- two solutions (+/-) per position
        eta : Tensor (..., 2N, M)
        two_theta : Tensor (..., 2N, M)
        valid : Tensor (..., 2N, M) float mask (1=valid, 0=invalid)
        """
        dtype = orientation_matrices.dtype
        if hkls_cart is None:
            hkls_cart = self.hkls.to(dtype)     # (M, 3)
        else:
            hkls_cart = hkls_cart.to(dtype)
        if thetas is None:
            thetas = self.thetas.to(dtype)      # (M,)
        else:
            thetas = thetas.to(dtype)

        # G_C = R @ hkls^T  =>  (..., N, M, 3)
        # Two cases supported: (a) hkls_cart shape (M, 3) shared across the
        # batch, (b) per-voxel hkls_cart shape (..., M, 3) for strained
        # rendering. Both flow through the same einsum via leading-dim
        # broadcasting on the second arg.
        G_C = torch.einsum("...nij,...mj->...nmi", orientation_matrices, hkls_cart)

        # v = sin(theta)*|G| -- C precomputes Gs from the UNROTATED G-vector norm
        # (rotation preserves norm in exact arithmetic but not in float64).
        # Match C: use |hkls_cart| (pre-rotation), not |R @ hkls_cart|.
        len_hkl = torch.norm(hkls_cart, dim=-1)  # (M,) or (..., M)
        v_no_wedge = torch.sin(thetas) * len_hkl  # (M,) or (..., M)
        v_no_wedge = v_no_wedge.unsqueeze(-2).expand_as(G_C[..., 0])  # (..., N, M)

        # ---- Wedge: rigorous geometric formulation -------------------
        # The rotation axis tilts from z to n_hat = (sin W, 0, cos W).
        # Full rotation by omega about n_hat:
        #     R_n_hat(omega) = R_y(W) @ R_z(omega) @ R_y(-W)
        # so G_lab = R_y(W) @ R_z(omega) @ G',  where G' = R_y(-W) @ G_sample.
        # Bragg condition -Gx_lab = sin(theta) * |G| reduces to the same
        # quadratic structure as the no-wedge solver, with substitutions:
        #     Gx_eff = cos W * G'_x
        #     Gy_eff = cos W * G'_y
        #     v_eff  = sin theta * |G| + sin W * G'_z
        # At W = 0, G' = G_sample and (Gx_eff, Gy_eff, v_eff) collapse to
        # (Gx, Gy, v), recovering the existing solver bit-identically.
        wedge_rad = self.wedge.to(dtype) * self.DEG2RAD
        cos_W = torch.cos(wedge_rad)
        sin_W = torch.sin(wedge_rad)
        # G' = R_y(-W) @ G_sample
        Gx_p = cos_W * G_C[..., 0] - sin_W * G_C[..., 2]  # (..., N, M)
        Gy_p = G_C[..., 1]
        Gz_p = sin_W * G_C[..., 0] + cos_W * G_C[..., 2]
        Gx = cos_W * Gx_p
        Gy = cos_W * Gy_p
        Gz = G_C[..., 2]  # unused except for reference; kept for clarity
        v = v_no_wedge + sin_W * Gz_p
        # ---------------------------------------------------------------

        # Quadratic solver for omega
        # -Gx*cos(w) + Gy*sin(w) = v   (with the effective Gx, Gy, v above)
        # Rearranged: a*cos^2(w) + b*cos(w) + c = 0
        # C uses almostzero=1e-12 for the Gy≈0 branch (see
        # NF_HEDM/src/CalcDiffractionSpots.c:96 and
        # FF_HEDM/src/ForwardSimulationCompressed.c:168). Match exactly.
        almostzero = 1e-12
        x2 = Gx * Gx
        y2 = Gy * Gy
        a = 1.0 + x2 / (y2 + self.epsilon)
        b_coeff = 2.0 * v * Gx / (y2 + self.epsilon)
        c_coeff = v * v / (y2 + self.epsilon) - 1.0
        discriminant = b_coeff * b_coeff - 4.0 * a * c_coeff

        sqrt_disc = torch.sqrt(torch.abs(discriminant))

        coswp = (-b_coeff + sqrt_disc) / (2.0 * a)
        coswn = (-b_coeff - sqrt_disc) / (2.0 * a)

        wap = self.safe_arccos(coswp)
        wan = self.safe_arccos(coswn)
        wbp = -wap
        wbn = -wan

        # Select correct branch: the one satisfying -Gx*cos(w)+Gy*sin(w)=v
        eqap = -Gx * torch.cos(wap) + Gy * torch.sin(wap)
        eqbp = -Gx * torch.cos(wbp) + Gy * torch.sin(wbp)
        eqan = -Gx * torch.cos(wan) + Gy * torch.sin(wan)
        eqbn = -Gx * torch.cos(wbn) + Gy * torch.sin(wbn)

        Dap = torch.abs(eqap - v)
        Dbp = torch.abs(eqbp - v)
        Dan = torch.abs(eqan - v)
        Dbn = torch.abs(eqbn - v)

        all_wp = torch.where(Dap < Dbp, wap, wbp)
        all_wn = torch.where(Dan < Dbn, wan, wbn)

        # Special case: Gy ~ 0 (C uses almostzero=1e-12)
        # C code (CalcDiffractionSpots.c:97-106):
        #   cosome1 = -v / x;
        #   if (|cosome1| <= 1) { ome = acos(cosome1); solutions: +ome, -ome }
        gy_zero = torch.abs(Gy) < almostzero
        cosome_special = -v / (Gx + self.epsilon)
        cosome_special_valid = (torch.abs(cosome_special) <= 1.0) & gy_zero & (torch.abs(Gx) > self.epsilon)
        special_w = self.safe_arccos(cosome_special)  # positive omega solution
        # Two solutions: +ome and -ome
        special_wp = special_w   # positive
        special_wn = -special_w  # negative

        # When |Gy| < almostzero, use the special case; otherwise use the quadratic
        disc_valid = (discriminant >= 0) & (~gy_zero)
        coswp_valid = (coswp >= -1.0) & (coswp <= 1.0)
        coswn_valid = (coswn >= -1.0) & (coswn <= 1.0)

        omega_p = torch.where(cosome_special_valid, special_wp,
                              torch.where(disc_valid & coswp_valid, all_wp,
                                          torch.zeros_like(all_wp)))
        omega_n = torch.where(cosome_special_valid, special_wn,
                              torch.where(disc_valid & coswn_valid, all_wn,
                                          torch.zeros_like(all_wn)))

        # Concatenate two solutions: (..., 2N, M)
        all_omega = torch.cat([omega_p, omega_n], dim=-2)

        # Rotate G' by omega: m = R_z(omega) @ G' (sample frame, axis-aligned)
        # then G_lab = R_y(W) @ m. R_z(omega) only mixes (x, y), so we apply
        # the rotation in closed form rather than materialising a 3x3 matrix
        # per spot. Double G' along the N dim to match the 2N K-axis of
        # all_omega.
        cos_w = torch.cos(all_omega)
        sin_w = torch.sin(all_omega)
        Gx_p_d = torch.cat([Gx_p, Gx_p], dim=-2)   # (..., 2N, M)
        Gy_p_d = torch.cat([Gy_p, Gy_p], dim=-2)
        Gz_p_d = torch.cat([Gz_p, Gz_p], dim=-2)
        m_x = cos_w * Gx_p_d - sin_w * Gy_p_d
        m_y = sin_w * Gx_p_d + cos_w * Gy_p_d
        m_z = Gz_p_d
        # G_lab = R_y(W) @ m
        Gy_lab = m_y                              # rotation about y leaves y
        Gz_lab = -sin_W * m_x + cos_W * m_z

        # Eta angle from lab-frame (y, z) components.
        r_yz = torch.sqrt(Gy_lab * Gy_lab + Gz_lab * Gz_lab).clamp(min=self.epsilon)
        eta = self.safe_arccos(Gz_lab / r_yz)
        eta = -torch.sign(Gy_lab) * eta

        # 2*theta  (broadcast thetas to match 2N dimension)
        two_theta_single = 2.0 * thetas.unsqueeze(-2)  # (..., 1, M) or (1, M)
        two_theta = two_theta_single.expand_as(all_omega)

        # Validity mask
        valid_p = disc_valid & coswp_valid
        valid_n = disc_valid & coswn_valid
        # For gy_zero special case, valid only if cosome is in [-1, 1]
        valid_p = valid_p | cosome_special_valid
        valid_n = valid_n | cosome_special_valid
        valid = torch.cat([valid_p, valid_n], dim=-2).float()

        # Eta bounds
        eta_ok = (torch.abs(eta) >= self.min_eta) & \
                 ((math.pi - torch.abs(eta)) >= self.min_eta)
        valid = valid * eta_ok.float()

        return all_omega, eta, two_theta, valid

    # ------------------------------------------------------------------
    #  Tilt rotation matrix (RotationTilts in SharedFuncsFit.c:230-266)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rot_tilts(tx_deg: float, ty_deg: float, tz_deg: float,
                         device: torch.device) -> torch.Tensor:
        """Build the 3x3 NF-style tilt rotation matrix Rz(tz) @ Ry(ty) @ Rx(tx).

        Matches RotationTilts() in NF_HEDM/src/SharedFuncsFit.c:230-266.
        """
        d2r = math.pi / 180.0
        tx, ty, tz = tx_deg * d2r, ty_deg * d2r, tz_deg * d2r
        cx, sx = math.cos(tx), math.sin(tx)
        cy, sy = math.cos(ty), math.sin(ty)
        cz, sz = math.cos(tz), math.sin(tz)
        Rx = torch.tensor([[1,  0,  0], [0, cx, -sx], [0, sx,  cx]], dtype=torch.float64)
        Ry = torch.tensor([[cy, 0, sy], [0,  1,  0], [-sy, 0, cy]], dtype=torch.float64)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0,  0,  1]], dtype=torch.float64)
        # Composition matches NF C: Rz @ Ry @ Rx
        return (Rz @ Ry @ Rx).to(device)

    def _build_rot_tilts_from_param(self, dtype, d_idx: int = 0) -> torch.Tensor:
        """Build Rz(tz) @ Ry(ty) @ Rx(tx) from row ``d_idx`` of self.tilts.

        ``self.tilts`` has shape (D, 3); each row is the (tx, ty, tz)
        triple for one detector. The returned matrix shares autograd
        history with self.tilts, so optimising tilts.requires_grad=True
        feeds gradients back to per-detector tilt entries.
        """
        d2r = math.pi / 180.0
        t = self.tilts[d_idx].to(dtype) * d2r
        tx_, ty_, tz_ = t[0], t[1], t[2]
        cx, sx = torch.cos(tx_), torch.sin(tx_)
        cy, sy = torch.cos(ty_), torch.sin(ty_)
        cz, sz = torch.cos(tz_), torch.sin(tz_)
        zero = torch.zeros((), dtype=dtype, device=t.device)
        one  = torch.ones((),  dtype=dtype, device=t.device)
        Rx = torch.stack([
            torch.stack([one,  zero, zero]),
            torch.stack([zero, cx,  -sx ]),
            torch.stack([zero, sx,   cx ]),
        ])
        Ry = torch.stack([
            torch.stack([cy,   zero, sy ]),
            torch.stack([zero, one,  zero]),
            torch.stack([-sy,  zero, cy ]),
        ])
        Rz = torch.stack([
            torch.stack([cz, -sz, zero]),
            torch.stack([sz,  cz, zero]),
            torch.stack([zero, zero, one]),
        ])
        return Rz @ Ry @ Rx

    def _apply_ff_panel_tilt(
        self, ydet: torch.Tensor, zdet: torch.Tensor, Lsd_val, d_idx: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Inverse of peakfit's tilt-aware compute_rt_eta for the FF panel
        convention.

        Forward (peakfit):
            (y_pix, z_pix) → Yc = -(y_pix-yBC)*px, Zc = (z_pix-zBC)*px
                          → ABCPr = TRs @ (0, Yc, Zc)
                          → X     = Lsd + ABCPr[0]
                          → (Y_proj, Z_proj) = (ABCPr[1], ABCPr[2]) * Lsd / X

        Inverse (this function):
            Given (ydet, zdet) = (Y_proj, Z_proj) lab coordinates at the
            Lsd plane, return (Yc, Zc) such that the forward path above
            reproduces them. Reduces to (ydet, zdet) when tilts are zero.

        Used by ``multi_mode='panel'`` with non-zero tilts. The NF
        ``_apply_nf_tilt`` ray-plane intersection diverges from the FF
        convention at large tilt angles (e.g. pinwheel tx ≈ 90°), so
        FF callers route here.
        """
        if not self._has_tilts and not self.tilts.requires_grad:
            return ydet, zdet
        dtype = ydet.dtype
        # Build the FF rotation Rx*Ry*Rz that peakfit's compute_rt_eta uses.
        d2r = math.pi / 180.0
        t = self.tilts[d_idx].to(dtype) * d2r
        cx, sx = torch.cos(t[0]), torch.sin(t[0])
        cy, sy = torch.cos(t[1]), torch.sin(t[1])
        cz, sz = torch.cos(t[2]), torch.sin(t[2])
        zero = torch.zeros((), dtype=dtype, device=t.device)
        one  = torch.ones((),  dtype=dtype, device=t.device)
        Rx = torch.stack([
            torch.stack([one,  zero, zero]),
            torch.stack([zero, cx,  -sx]),
            torch.stack([zero, sx,   cx]),
        ])
        Ry = torch.stack([
            torch.stack([cy,  zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ])
        Rz = torch.stack([
            torch.stack([cz, -sz, zero]),
            torch.stack([sz,  cz, zero]),
            torch.stack([zero, zero, one]),
        ])
        TRs = Rx @ (Ry @ Rz)

        # Per-spot 2×2 linear solve for (Yc, Zc) such that
        #   (TRs[1,1] - ydet*α)*Yc + (TRs[1,2] - ydet*β)*Zc = ydet
        #   (TRs[2,1] - zdet*α)*Yc + (TRs[2,2] - zdet*β)*Zc = zdet
        # with α = TRs[0,1]/Lsd, β = TRs[0,2]/Lsd.
        Lsd_t = Lsd_val if torch.is_tensor(Lsd_val) else torch.tensor(
            Lsd_val, dtype=dtype, device=ydet.device,
        )
        alpha = TRs[0, 1] / Lsd_t
        beta  = TRs[0, 2] / Lsd_t
        M11 = TRs[1, 1] - ydet * alpha
        M12 = TRs[1, 2] - ydet * beta
        M21 = TRs[2, 1] - zdet * alpha
        M22 = TRs[2, 2] - zdet * beta
        det = M11 * M22 - M12 * M21
        safe_det = torch.where(
            torch.abs(det) < self.epsilon,
            torch.full_like(det, self.epsilon), det,
        )
        Yc = (ydet * M22 - zdet * M12) / safe_det
        Zc = (M11 * zdet - M21 * ydet) / safe_det
        return Yc, Zc

    def _apply_nf_tilt(self, ydet: torch.Tensor, zdet: torch.Tensor,
                        Lsd_val, d_idx: int = 0) -> "tuple[torch.Tensor, torch.Tensor]":
        """Apply the per-detector tilt correction to lab-frame (ydet, zdet).

        Ports the ray-plane intersection in
        ``NF_HEDM/src/SharedFuncsFit.c:947-958``: builds
        P0 = RotMatTilts @ [-Lsd, 0, 0], P1 = RotMatTilts @ [0, ydet, zdet],
        and returns the (y, z) coordinates where the line from P0 through P1
        crosses the plane x = 0. Reduces to the identity when tilts are zero.

        ``d_idx`` selects which detector's (tx, ty, tz) row to use from
        ``self.tilts``; the returned tensors share autograd history with
        the tilt parameter so gradient-based refinement of tilts works.
        ``Lsd_val`` can be a Python scalar or a scalar tensor.
        """
        if not self._has_tilts and not self.tilts.requires_grad:
            return ydet, zdet
        dtype = ydet.dtype
        R = self._build_rot_tilts_from_param(dtype, d_idx=d_idx)
        # P0 = -Lsd * R[:, 0]  (3 scalars)
        p0x = -Lsd_val * R[0, 0]
        p0y = -Lsd_val * R[1, 0]
        p0z = -Lsd_val * R[2, 0]
        # P1 = ydet * R[:, 1] + zdet * R[:, 2]  (pointwise on tensors)
        P1x = ydet * R[0, 1] + zdet * R[0, 2]
        P1y = ydet * R[1, 1] + zdet * R[1, 2]
        P1z = ydet * R[2, 1] + zdet * R[2, 2]
        ABCx = P1x - p0x
        ABCy = P1y - p0y
        ABCz = P1z - p0z
        safe_denom = torch.where(
            torch.abs(ABCx) < self.epsilon,
            torch.full_like(ABCx, self.epsilon),
            ABCx,
        )
        out_y = p0y - ABCy * p0x / safe_denom
        out_z = p0z - ABCz * p0x / safe_denom
        return out_y, out_z

    @staticmethod
    def _build_ff_tilt_rot(tx_deg: float, ty_deg: float, tz_deg: float,
                           device: torch.device) -> torch.Tensor:
        """Build the 3x3 FF-style tilt rotation matrix Rx(tx) @ Ry(ty) @ Rz(tz).

        Matches CorrectTiltSpatialDistortion() in
        FF_HEDM/src/ForwardSimulationCompressed.c:593-612.
        Note the composition differs from the NF convention.
        """
        d2r = math.pi / 180.0
        tx, ty, tz = tx_deg * d2r, ty_deg * d2r, tz_deg * d2r
        cx, sx = math.cos(tx), math.sin(tx)
        cy, sy = math.cos(ty), math.sin(ty)
        cz, sz = math.cos(tz), math.sin(tz)
        Rx = torch.tensor([[1,  0,  0], [0, cx, -sx], [0, sx,  cx]], dtype=torch.float64)
        Ry = torch.tensor([[cy, 0, sy], [0,  1,  0], [-sy, 0, cy]], dtype=torch.float64)
        Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0,  0,  1]], dtype=torch.float64)
        # Composition matches FF C: Rx @ Ry @ Rz
        return (Rx @ Ry @ Rz).to(device)

    # ------------------------------------------------------------------
    #  project_to_detector
    # ------------------------------------------------------------------

    def project_to_detector(
        self,
        omega: torch.Tensor,
        eta: torch.Tensor,
        two_theta: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> SpotDescriptors:
        """Position-dependent detector projection for one or more distances.

        Implements the geometry from ``SharedFuncsFit.c:DisplacementSpots``
        (lines 269-292).  For multi-distance NF-HEDM, projects to each
        distance and requires spots to be on-detector at ALL distances
        (the ``AllDistsFound`` logic from ``CalcFracOverlap``, line 638).

        Parameters
        ----------
        omega : Tensor (..., 2N, M)
        eta : Tensor (..., 2N, M)
        two_theta : Tensor (..., 2N, M)
        positions : Tensor (N, 3) or (..., N, 3)
            Real-space positions [x, y, z] in micrometers.
        valid : Tensor (..., 2N, M)

        Returns
        -------
        SpotDescriptors
        """
        N = positions.shape[-2]

        # Omega-rotated position: rotate (x,y,z) by omega around z-axis
        pos_doubled = torch.cat([positions, positions], dim=-2)  # (..., 2N, 3)

        cos_w = torch.cos(omega)  # (..., 2N, M)
        sin_w = torch.sin(omega)

        px = pos_doubled[..., 0].unsqueeze(-1)  # (..., 2N, 1)
        py = pos_doubled[..., 1].unsqueeze(-1)
        pz = pos_doubled[..., 2].unsqueeze(-1)

        x_grain = px * cos_w - py * sin_w  # (..., 2N, M)
        y_grain = px * sin_w + py * cos_w
        z_grain = pz.expand_as(x_grain)

        tan_2th = torch.tan(two_theta)
        sin_eta = torch.sin(eta)
        cos_eta = torch.cos(eta)

        # Frame number (same at all distances -- omega doesn't change)
        frame_nr = (omega / self.DEG2RAD - self.omega_start) / self.omega_step
        frame_ok = (frame_nr >= 0) & (frame_nr < self.n_frames)

        # Project to each detector distance
        D = self.n_distances
        dtype = omega.dtype
        # _Lsd, _y_BC, _z_BC are (D,) tensors
        # Reshape to (D, 1..., 1, 1) for broadcasting against (..., 2N, M)
        extra_dims = omega.ndim  # number of dims in (..., 2N, M)
        Lsd_d = self._Lsd_eff.to(dtype).reshape(D, *([1] * extra_dims))
        yBC_d = self._y_BC.to(dtype).reshape(D, *([1] * extra_dims))
        zBC_d = self._z_BC.to(dtype).reshape(D, *([1] * extra_dims))

        # ydet, zdet, y_pixel, z_pixel all get shape (D, ..., 2N, M)
        dist_d = Lsd_d - x_grain.unsqueeze(0)  # (D, ..., 2N, M)
        ydet_d = y_grain.unsqueeze(0) - dist_d * tan_2th.unsqueeze(0) * sin_eta.unsqueeze(0)
        zdet_d = z_grain.unsqueeze(0) + dist_d * tan_2th.unsqueeze(0) * cos_eta.unsqueeze(0)

        # Apply detector tilt.
        #
        # Design note: FF and pf-HEDM *experimental* workflows apply a
        # DetCor correction at peak-finding time, so the per-spot centroids
        # in SpotMatrix.csv are already tilt- and distortion-corrected. A
        # differentiable forward model targeting that data therefore must
        # NOT apply tilts -- doing so would double-correct. This is why
        # FF mode (``flip_y=True``) skips tilts by default.
        #
        # NF-HEDM works at pixel level against raw detector images with no
        # DetCor step, so the forward model MUST include tilts. Multi-panel
        # FF synthetic data (this file's panel mode) is in the same boat:
        # there is no DetCor pre-correction. Set ``apply_tilts=True`` on
        # the geometry to force tilt application in FF mode.
        #
        # Ports the ray-plane intersection in NF_HEDM/src/SharedFuncsFit.c
        # (composition Rz @ Ry @ Rx, P0 = R @ [-Lsd, 0, 0]) per detector,
        # using row d of self.tilts.
        tilts_active = self._has_tilts or self.tilts.requires_grad
        apply = tilts_active and ((not self.flip_y) or self.apply_tilts)
        if apply:
            Lsd_list = self._Lsd_eff.to(dtype)
            out_y = []
            out_z = []
            # FF multi-panel mode uses the inverse of peakfit's
            # tilt-aware compute_rt_eta (small-tilt agreement with the
            # NF ray-plane intersection, but the latter diverges for
            # pinwheel-scale tilts). NF mode keeps the ray-plane path.
            ff_panel = self.flip_y and self.multi_mode == "panel" and self.apply_tilts
            for d in range(self.n_distances):
                if ff_panel:
                    yd, zd = self._apply_ff_panel_tilt(
                        ydet_d[d], zdet_d[d], Lsd_list[d], d_idx=d
                    )
                else:
                    yd, zd = self._apply_nf_tilt(
                        ydet_d[d], zdet_d[d], Lsd_list[d], d_idx=d
                    )
                out_y.append(yd)
                out_z.append(zd)
            ydet_d = torch.stack(out_y, dim=0)
            zdet_d = torch.stack(out_z, dim=0)

        # Ideal->raw radial distortion (canonical midas_distortion v2 model),
        # applied on the BC-relative detector-plane coords (um) before pixel
        # conversion -- mirrors midas_calibrate_v2.forward.geometry. Gated OFF
        # by default so the indexer/fit-grain (ideal frame) are byte-unchanged;
        # raw-patch consumers (pf_odf, grain_odf) opt in. R is BC-relative, so
        # the distortion is frame-flip invariant; the eta convention (and any
        # phase offset between the calibration frame and this frame) is the one
        # thing to validate empirically (see implementation_plan_distortion_layer).
        if self.apply_distortion and self._has_distortion:
            from midas_distortion import apply_distortion as _apply_dist, \
                v2_term_layout as _v2_terms, resolve_rho_d_um as _resolve_rho_d
            eps = torch.tensor(1e-9, dtype=ydet_d.dtype, device=ydet_d.device)
            R = torch.sqrt(ydet_d * ydet_d + zdet_d * zdet_d).clamp(min=eps)
            # eta convention matches calibrate_v2 forward: atan2(-y, z), degrees.
            eta_deg_d = self.RAD2DEG * torch.atan2(-ydet_d, zdet_d)
            # resolve_rho_d_um passes a supplied rho_d through, or computes the
            # max BC-relative corner distance (um) when None.
            rho_d_val, _rho_how = _resolve_rho_d(
                self.rho_d,
                NrPixelsY=self.n_pixels_y, NrPixelsZ=self.n_pixels_z,
                BC_y=float(self._y_BC.reshape(-1)[0]),
                BC_z=float(self._z_BC.reshape(-1)[0]),
                pxY=self.px,
            )
            rho_d_t = torch.as_tensor(float(rho_d_val), dtype=R.dtype, device=R.device)
            p_v2 = self.p_distortion.to(R.dtype)
            R_corr = _apply_dist(R, eta_deg_d, p_v2, rho_d_t, terms=_v2_terms())
            scale = R_corr / R
            ydet_d = ydet_d * scale
            zdet_d = zdet_d * scale

        # FF/PF: y-axis on detector flipped (yBC - ydet/px), validated against C
        # NF:    not flipped (yBC + ydet/px), validated against C
        y_sign = -1.0 if self.flip_y else 1.0
        y_pixel_d = yBC_d + y_sign * ydet_d / self.px  # (D, ..., 2N, M)
        z_pixel_d = zBC_d + zdet_d / self.px

        # Per-detector / per-distance bounds check
        layer_bounds_ok = (
            (y_pixel_d >= 0) & (y_pixel_d < self.n_pixels_y) &
            (z_pixel_d >= 0) & (z_pixel_d < self.n_pixels_z)
        )  # (D, ..., 2N, M)

        # Per-detector validity = angular valid & frame ok & on-detector
        layer_valid = (
            valid.unsqueeze(0) * frame_ok.unsqueeze(0).float()
            * layer_bounds_ok.float()
        )

        if self.multi_mode == "layered":
            # NF semantics: spot must be on every layer (AllDistsFound).
            overall_valid = layer_valid.prod(dim=0)  # (..., 2N, M)
            if D == 1:
                y_pixel_out = y_pixel_d.squeeze(0)
                z_pixel_out = z_pixel_d.squeeze(0)
                layer_valid_out = None
            else:
                y_pixel_out = y_pixel_d
                z_pixel_out = z_pixel_d
                layer_valid_out = layer_valid
            det_id_out = None
        elif self.multi_mode == "panel":
            # FF multi-panel semantics: spot is valid if it lands on at
            # least one panel; record which panel it landed on.
            #
            # We pick the first panel where layer_valid > 0.5. With
            # physically separated panels (typical multi-panel setups)
            # there is no overlap, so "first" == "the" panel. If panels
            # do overlap, this returns the lowest-index hit -- a
            # deterministic but arbitrary tiebreak.
            valid_bool = layer_valid > 0.5  # (D, ..., 2N, M) bool
            any_panel = valid_bool.any(dim=0)  # (..., 2N, M)
            # argmax of bool along D returns the first True (or 0 if none).
            # We mask out the "or 0 if none" case via any_panel below.
            det_id_out = valid_bool.float().argmax(dim=0).to(torch.int64)
            # Gather y_pixel / z_pixel along D using det_id.
            #   y_pixel_d: (D, ..., 2N, M); det_id: (..., 2N, M)
            gather_idx = det_id_out.unsqueeze(0)  # (1, ..., 2N, M)
            y_pixel_out = torch.gather(y_pixel_d, 0, gather_idx).squeeze(0)
            z_pixel_out = torch.gather(z_pixel_d, 0, gather_idx).squeeze(0)
            overall_valid = any_panel.float()
            layer_valid_out = layer_valid
        else:
            raise RuntimeError(f"Internal: unknown multi_mode {self.multi_mode!r}")

        return SpotDescriptors(
            omega=omega,
            eta=eta,
            two_theta=two_theta,
            y_pixel=y_pixel_out,
            z_pixel=z_pixel_out,
            frame_nr=frame_nr,
            valid=overall_valid,
            layer_valid=layer_valid_out,
            det_id=det_id_out,
        )

    # ------------------------------------------------------------------
    #  forward  (orchestrator)
    # ------------------------------------------------------------------

    def forward(
        self,
        euler_angles: torch.Tensor,
        positions: torch.Tensor,
        lattice_params: Optional[torch.Tensor] = None,
        strain: Optional[torch.Tensor] = None,
    ) -> SpotDescriptors:
        """Full forward simulation pipeline.

        Parameters
        ----------
        euler_angles : Tensor (..., N, 3)
            Euler angles (phi1, Phi, phi2) in radians at each position.
        positions : Tensor (N, 3) or (N, 2) or (..., N, 3)
            Real-space positions in micrometers. If (N,2), z is padded to 0.
        lattice_params : Tensor (..., 6) or (..., N, 6), optional
            Strained lattice parameters [a,b,c,alpha,beta,gamma] in
            Angstroms/degrees. None = use nominal hkls/thetas (no strain).
        strain : Tensor (..., 6), (..., N, 6), or (..., 3, 3), optional
            Crystal-frame symmetric infinitesimal strain, either PLAIN-Voigt
            [eps_11, eps_12, eps_13, eps_22, eps_23, eps_33] or a full symmetric
            3x3 tensor (see :meth:`strain_as_voigt`). Applied as
            B = (I + eps)^{-1} @ B0 in addition to any lattice-parameter
            strain expressed through ``lattice_params``. Requires
            ``lattice_params`` to be supplied.

        Returns
        -------
        SpotDescriptors
        """
        # Backward compat: pad (N,2) -> (N,3)
        if positions.shape[-1] == 2:
            positions = F.pad(positions, (0, 1), value=0.0)

        # 1. Orientation matrices
        orientation_matrices = self.euler2mat(euler_angles)

        # 2. Optionally compute strained G-vectors / thetas
        hkls_cart = None
        thetas = None
        if lattice_params is not None:
            hkls_cart, thetas = self.correct_hkls_latc(lattice_params, strain=strain)
        elif strain is not None:
            raise ValueError(
                "strain was supplied but lattice_params is None; strain "
                "requires a reference lattice to apply (I + eps)^{-1} @ B0."
            )

        # 3. Bragg geometry
        omega, eta, two_theta, valid = self.calc_bragg_geometry(
            orientation_matrices, hkls_cart, thetas
        )

        # 4. Detector projection
        spots = self.project_to_detector(omega, eta, two_theta, positions, valid)

        # 5. Scan filter (multi-scan only)
        if self.scan_config is not None:
            spots = self.filter_by_scan(spots, positions)

        return spots

    # ------------------------------------------------------------------
    #  filter_by_scan  (beam proximity for pf-HEDM)
    # ------------------------------------------------------------------

    def filter_by_scan(
        self, spots: SpotDescriptors, positions: torch.Tensor
    ) -> SpotDescriptors:
        """Apply beam illumination filter for multi-scan geometry.

        For each spot, checks whether the omega-rotated y-position of the
        source voxel falls within the beam at each scan position.

        Ports ``FitOrStrainsScanningOMP.c:1050-1058``:
            yRot = posX * sin(omega) + posY * cos(omega)
            |yRot - beam_y[scan]| < beam_size / 2

        Parameters
        ----------
        spots : SpotDescriptors
        positions : Tensor (..., N, 3)

        Returns
        -------
        SpotDescriptors with ``scan_mask`` populated.
        """
        if self.scan_config is None:
            return spots

        N = positions.shape[-2]
        pos_doubled = torch.cat([positions, positions], dim=-2)  # (..., 2N, 3)

        cos_w = torch.cos(spots.omega)  # (..., 2N, M)
        sin_w = torch.sin(spots.omega)

        px = pos_doubled[..., 0].unsqueeze(-1)  # (..., 2N, 1)
        py = pos_doubled[..., 1].unsqueeze(-1)

        # Omega-rotated y position
        y_rot = px * sin_w + py * cos_w  # (..., 2N, M)

        # beam_positions: (S,)
        beam_y = self._beam_positions
        half_beam = self._beam_size / 2.0

        # |yRot - beam_y[s]| < half_beam
        # y_rot: (..., 2N, M), beam_y: (S,)
        # Expand for broadcasting: (..., 1, 2N, M) vs (S, 1, 1)
        y_rot_exp = y_rot.unsqueeze(-3)  # (..., 1, 2N, M)
        beam_y_exp = beam_y.reshape(-1, 1, 1)  # (S, 1, 1)

        scan_mask = (torch.abs(y_rot_exp - beam_y_exp) < half_beam).float()
        # Combine with overall validity
        scan_mask = scan_mask * spots.valid.unsqueeze(-3)

        spots = SpotDescriptors(
            omega=spots.omega,
            eta=spots.eta,
            two_theta=spots.two_theta,
            y_pixel=spots.y_pixel,
            z_pixel=spots.z_pixel,
            frame_nr=spots.frame_nr,
            valid=spots.valid,
            scan_mask=scan_mask,
        )
        return spots

    # ------------------------------------------------------------------
    #  predict_images  (NF output mode: Gaussian splatting)
    # ------------------------------------------------------------------

    @staticmethod
    def predict_images(
        spots: SpotDescriptors,
        n_frames: int,
        n_pixels_y: int,
        n_pixels_z: int,
        sigma: float = 1.0,
        radius: int = 3,
    ) -> torch.Tensor:
        """Gaussian-splat predicted spots onto detector grid.

        This is the NF-HEDM output mode. Each valid spot is represented
        as a Gaussian blob on the (frame, y, z) detector volume.

        Parameters
        ----------
        spots : SpotDescriptors
            Output from ``forward()``.
        n_frames, n_pixels_y, n_pixels_z : int
            Grid dimensions.
        sigma : float
            Gaussian kernel sigma in pixels.
        radius : int
            Kernel radius in pixels.

        Returns
        -------
        Tensor (..., n_frames, n_pixels_y, n_pixels_z)
        """
        # Extract coordinates and mask
        frame_nr = spots.frame_nr  # (..., K, M)
        y_pix = spots.y_pixel
        z_pix = spots.z_pixel
        valid = spots.valid

        # Flatten batch dims for processing
        orig_shape = frame_nr.shape  # (..., K, M)
        batch_shape = orig_shape[:-2]
        n_batch = 1
        for s in batch_shape:
            n_batch *= s
        KM = orig_shape[-2] * orig_shape[-1]

        # Reshape to (B, KM, 3) where 3 = (frame, y, z)
        coords = torch.stack([frame_nr, y_pix, z_pix], dim=-1)  # (..., K, M, 3)
        coords = coords.reshape(n_batch, KM, 3)
        mask = valid.reshape(n_batch, KM)

        # Zero out invalid spots
        coords = coords * mask.unsqueeze(-1)

        device = coords.device
        dtype = coords.dtype

        grids = torch.zeros(n_batch, n_frames, n_pixels_y, n_pixels_z,
                            dtype=dtype, device=device)
        gaussian_factor = -0.5 / (sigma ** 2)

        # Filter non-zero coordinates
        batch_ids = torch.arange(n_batch, device=device).unsqueeze(1).expand(-1, KM)
        non_zero_mask = mask > 0.5
        non_zero_coords = coords[non_zero_mask]
        non_zero_batch = batch_ids[non_zero_mask]

        if non_zero_coords.shape[0] == 0:
            return grids.reshape(*batch_shape, n_frames, n_pixels_y, n_pixels_z)

        rounded = non_zero_coords.round().long()

        # Neighborhood offsets
        offsets = torch.arange(-radius, radius + 1, device=device)
        oz, ox, oy = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
        local_offsets = torch.stack([oz, ox, oy], dim=-1).reshape(-1, 3)
        n_local = local_offsets.shape[0]

        expanded_centers = rounded.unsqueeze(1)  # (P, 1, 3)
        neighbors = expanded_centers + local_offsets.unsqueeze(0)  # (P, L, 3)

        f_neigh = neighbors[..., 0].clamp(0, n_frames - 1)
        y_neigh = neighbors[..., 1].clamp(0, n_pixels_y - 1)
        z_neigh = neighbors[..., 2].clamp(0, n_pixels_z - 1)

        distances = torch.sum(
            (neighbors.float() - non_zero_coords.unsqueeze(1)) ** 2, dim=-1
        )
        weights = torch.exp(distances * gaussian_factor)

        # Flatten for scatter_add
        f_flat = f_neigh.flatten()
        y_flat = y_neigh.flatten()
        z_flat = z_neigh.flatten()
        w_flat = weights.flatten()

        batch_flat = non_zero_batch.unsqueeze(1).expand(-1, n_local).flatten()

        flat_idx = (batch_flat * (n_frames * n_pixels_y * n_pixels_z)
                    + f_flat * (n_pixels_y * n_pixels_z)
                    + y_flat * n_pixels_z
                    + z_flat)

        grids.view(-1).scatter_add_(0, flat_idx, w_flat)

        return grids.reshape(*batch_shape, n_frames, n_pixels_y, n_pixels_z)

    # ------------------------------------------------------------------
    #  predict_spot_coords  (FF/pf output mode)
    # ------------------------------------------------------------------

    @staticmethod
    def predict_spot_coords(
        spots: SpotDescriptors,
        space: str = "angular",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract spot coordinates for COM matching (FF/pf mode).

        Parameters
        ----------
        spots : SpotDescriptors
        space : str
            ``"angular"``: return (2theta, eta, omega) in radians.
            ``"detector"``: return (y_pixel, z_pixel, frame_nr).

        Returns
        -------
        coords : Tensor (..., K, M, 3)
        valid : Tensor (..., K, M)
        """
        if space == "angular":
            coords = torch.stack(
                [spots.two_theta, spots.eta, spots.omega], dim=-1
            )
        elif space == "detector":
            coords = torch.stack(
                [spots.y_pixel, spots.z_pixel, spots.frame_nr], dim=-1
            )
        else:
            raise ValueError(f"Unknown space: {space!r}. Use 'angular' or 'detector'.")
        return coords, spots.valid

    # ------------------------------------------------------------------
    #  Triangular voxel support  (NF-HEDM)
    # ------------------------------------------------------------------

    @staticmethod
    def tri_vertices(
        centers: torch.Tensor,
        edge_lengths: torch.Tensor,
        ud: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 3 triangle vertices from voxel centres.

        Matches ``simulateNF.c`` lines 556-572.

        Parameters
        ----------
        centers : Tensor (N, 2) or (N, 3)
            Voxel centres [x, y] or [x, y, z] in micrometers.
        edge_lengths : Tensor (N,)
            Edge length per voxel in micrometers.
        ud : Tensor (N,)
            Up/down flag (+1 or -1).

        Returns
        -------
        Tensor (N, 3, 3)
            Vertices ``[V0, V1, V2]`` each of shape ``(3,)`` = ``[x, y, z]``.
            If input centres are 2D, z is set to 0.
        """
        if centers.shape[-1] == 2:
            centers = F.pad(centers, (0, 1), value=0.0)

        xs = centers[:, 0]
        ys = centers[:, 1]
        zs = centers[:, 2]

        gs = edge_lengths / 2.0
        sqrt3 = math.sqrt(3.0)
        dy1 = edge_lengths / sqrt3
        dy2 = -edge_lengths / (2.0 * sqrt3)
        # flip if ud < 0
        sign = torch.sign(ud)
        dy1 = dy1 * sign
        dy2 = dy2 * sign

        # V0 = (xs, ys+dy1, zs), V1 = (xs-gs, ys+dy2, zs), V2 = (xs+gs, ys+dy2, zs)
        V0 = torch.stack([xs,      ys + dy1, zs], dim=-1)
        V1 = torch.stack([xs - gs, ys + dy2, zs], dim=-1)
        V2 = torch.stack([xs + gs, ys + dy2, zs], dim=-1)

        return torch.stack([V0, V1, V2], dim=1)  # (N, 3, 3)

    @staticmethod
    def _c_round(x: float) -> int:
        """C-compatible round(): half away from zero.

        Python's ``round()`` uses banker's rounding (half to even),
        which differs from C's ``round()`` at ``.5`` boundaries.
        C's ``(int)round(x)`` = ``floor(x + 0.5)`` for x >= 0,
        ``ceil(x - 0.5)`` for x < 0.
        """
        return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))

    @staticmethod
    def rasterize_triangle(v0y, v0z, v1y, v1z, v2y, v2z):
        """Rasterize a single triangle on an integer pixel grid.

        Matches ``CalcPixels2`` in ``SharedFuncsFit.c`` lines 308-370.
        Uses the edge-function rasterizer with a distance-based border
        (``distSq < 0.9801``) for edges, exactly as the C code does.

        Parameters
        ----------
        v0y, v0z, v1y, v1z, v2y, v2z : int
            Rounded integer pixel coordinates of the 3 vertices.

        Returns
        -------
        list of (int, int)
            List of (y, z) integer pixel coordinates inside the triangle.
        """
        min_y = min(v0y, v1y, v2y)
        max_y = max(v0y, v1y, v2y)
        min_z = min(v0z, v1z, v2z)
        max_z = max(v0z, v1z, v2z)

        # Edge function coefficients (matching C variable names)
        A01 = v0z - v1z;  B01 = v1y - v0y
        A12 = v1z - v2z;  B12 = v2y - v1y
        A20 = v2z - v0z;  B20 = v0y - v2y

        def orient2d(ax, ay, bx, by, cx, cy):
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        def dist_sq_to_edge(ax, ay, bx, by, px, py):
            num = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            den_sq = (ay - by) ** 2 + (bx - ax) ** 2
            if den_sq == 0:
                return 1e30
            return (num * num) / den_sq

        pixels = []
        w0_row = orient2d(v1y, v1z, v2y, v2z, min_y, min_z)
        w1_row = orient2d(v2y, v2z, v0y, v0z, min_y, min_z)
        w2_row = orient2d(v0y, v0z, v1y, v1z, min_y, min_z)

        for pz in range(min_z, max_z + 1):
            w0 = w0_row
            w1 = w1_row
            w2 = w2_row
            for py in range(min_y, max_y + 1):
                inside = (w0 >= 0 and w1 >= 0 and w2 >= 0)
                if not inside:
                    # Check distance to each edge (0.9801 = 0.99^2)
                    inside = (
                        dist_sq_to_edge(v1y, v1z, v2y, v2z, py, pz) < 0.9801 or
                        dist_sq_to_edge(v2y, v2z, v0y, v0z, py, pz) < 0.9801 or
                        dist_sq_to_edge(v0y, v0z, v1y, v1z, py, pz) < 0.9801
                    )
                if inside:
                    pixels.append((py, pz))
                w0 += A12
                w1 += A20
                w2 += A01
            w0_row += B12
            w1_row += B20
            w2_row += B01

        return pixels

    def forward_nf_triangles(
        self,
        euler_angles: torch.Tensor,
        centers: torch.Tensor,
        tri_config: TriVoxelConfig,
        lattice_params: Optional[torch.Tensor] = None,
        strain: Optional[torch.Tensor] = None,
    ) -> SpotDescriptors:
        """NF-HEDM forward simulation with triangular voxel rasterization.

        Matches the full ``simulateNF`` / ``CalcFracOverlap`` pipeline:
        compute Bragg geometry once per voxel, project 3 triangle vertices,
        rasterize the detector-space triangle, check all distances.

        Parameters
        ----------
        euler_angles : Tensor (N, 3) in radians
        centers : Tensor (N, 2) or (N, 3) in micrometers
        tri_config : TriVoxelConfig
        lattice_params : optional strain

        Returns
        -------
        SpotDescriptors with per-pixel y_pixel/z_pixel (not per-vertex).
            y_pixel, z_pixel: lists of per-voxel, per-spot rasterized pixel coords.
            For bit-level comparison, use ``predict_spotsinfo_bits`` instead.
        """
        if centers.shape[-1] == 2:
            centers = F.pad(centers, (0, 1), value=0.0)

        N = centers.shape[0]
        vertices = self.tri_vertices(
            centers, tri_config.edge_lengths, tri_config.ud
        )  # (N, 3, 3)

        # 1. Compute orientation matrices.
        # C: reads radians from .mic, multiplies by rad2deg, then
        # Euler2OrientMat uses cosd()=cos(deg2rad*x). With both C and Python
        # now using M_PI/180, the roundtrip is lossless and we can use
        # radians directly.
        euler_deg_c = euler_angles * self.RAD2DEG
        euler_rad_c = euler_deg_c * self.DEG2RAD
        orientation_matrices = self.euler2mat(euler_rad_c)

        # 2. Optionally strained HKLs
        hkls_cart = thetas = None
        if lattice_params is not None:
            hkls_cart, thetas = self.correct_hkls_latc(lattice_params, strain=strain)
        elif strain is not None:
            raise ValueError(
                "strain was supplied but lattice_params is None; strain "
                "requires a reference lattice to apply (I + eps)^{-1} @ B0."
            )

        # 3. Bragg geometry (once per voxel, from center orientation)
        omega, eta, two_theta, valid = self.calc_bragg_geometry(
            orientation_matrices, hkls_cart, thetas
        )
        # omega, eta, two_theta, valid: (2N, M)

        # Recompute G_C for eta recomputation below
        dtype = omega.dtype
        use_hkls = hkls_cart if hkls_cart is not None else self.hkls.to(dtype)
        G_C = torch.einsum("...nij,mj->...nmi", orientation_matrices, use_hkls)

        # 4. Frame number
        frame_nr = (omega / self.DEG2RAD - self.omega_start) / self.omega_step
        frame_ok = (frame_nr >= 0) & (frame_nr < self.n_frames)
        valid = valid * frame_ok.float()

        # 5. Project each vertex through DisplacementSpots for each distance
        D = self.n_distances
        dtype = omega.dtype
        Lsd_0 = self._Lsd[0].to(dtype)

        # ---------------------------------------------------------------
        # Replicate C's exact CalcOmega→CalcSpotPosition chain to avoid
        # float-precision differences at pixel boundaries.
        #
        # C CalcOmega stores omega in DEGREES: omega_deg = acos(...)*rad2deg
        # C then recomputes eta by RotateAroundZ(G, omega_DEG):
        #   internally: omega_rad2 = omega_deg * deg2rad   (roundtrip!)
        #   gw = Rz(omega_rad2) @ G
        #   eta_deg = CalcEtaAngle(gw[1], gw[2])
        # C CalcSpotPosition: eta_rad = eta_deg * deg2rad  (another roundtrip!)
        #   yl = -sin(eta_rad) * RingRadius
        #   zl = cos(eta_rad) * RingRadius
        # C RingRadius = Lsd * tan(2 * deg2rad * Theta_deg)
        #
        # We replicate every degree conversion.
        # ---------------------------------------------------------------
        omega_deg = omega * self.RAD2DEG
        omega_rad_c = omega_deg * self.DEG2RAD  # C's deg2rad * omega_deg

        cos_w = torch.cos(omega_rad_c)
        sin_w = torch.sin(omega_rad_c)

        # Recompute eta via C's chain: rotate G by omega_deg, then CalcEtaAngle
        G_C_doubled = torch.cat([G_C, G_C], dim=-3)  # (2N, M, 3)
        # gw = Rz(omega_rad_c) @ G
        gw_y = G_C_doubled[..., 0] * sin_w + G_C_doubled[..., 1] * cos_w
        gw_z = G_C_doubled[..., 2]
        r_yz = torch.sqrt(gw_y * gw_y + gw_z * gw_z).clamp(min=self.epsilon)
        eta_c_rad = torch.acos(torch.clamp(gw_z / r_yz,
                                           -1.0 + self.epsilon, 1.0 - self.epsilon))
        eta_c_rad = -torch.sign(gw_y) * eta_c_rad  # C: if (y > 0) alpha = -alpha
        # Convert to degrees then back (C's CalcSpotPosition chain)
        eta_c_deg = eta_c_rad * self.RAD2DEG
        eta_c_rad2 = eta_c_deg * self.DEG2RAD

        theta_deg = (two_theta / 2.0) * self.RAD2DEG
        ring_radius = Lsd_0 * torch.tan(2.0 * self.DEG2RAD * theta_deg)

        sin_eta_c = torch.sin(eta_c_rad2)
        cos_eta_c = torch.cos(eta_c_rad2)
        ythis = -sin_eta_c * ring_radius
        zthis = cos_eta_c * ring_radius

        # Project center reference. For non-zero tilts, apply the NF
        # ray-plane intersection (SharedFuncsFit.c:947-958).
        # YZSpotsTemp = outxyz/px + bc
        ybc_0 = self._y_BC[0].to(dtype)
        zbc_0 = self._z_BC[0].to(dtype)
        Lsd_0_scalar = self._Lsd[0].to(dtype)
        y_center_lab, z_center_lab = self._apply_nf_tilt(ythis, zthis, Lsd_0_scalar)
        y_center = y_center_lab / self.px + ybc_0  # (2N, M)
        z_center = z_center_lab / self.px + zbc_0

        # Project each vertex: DisplacementSpots for each of 3 vertices
        # vertices: (N, 3, 3) -> double to (2N, 3, 3)
        verts_doubled = torch.cat([vertices, vertices], dim=0)  # (2N, 3, 3)

        # For each vertex v, compute Displ_Y, Displ_Z:
        # xa = vx*cos(w) - vy*sin(w), ya = vx*sin(w) + vy*cos(w)
        # Displ_Y = ya + ythis*(1-xa/Lsd), Displ_Z = (1-xa/Lsd)*zthis
        vert_y_pixel = []
        vert_z_pixel = []
        for vi in range(3):
            vx = verts_doubled[:, vi, 0].unsqueeze(-1)  # (2N, 1)
            vy = verts_doubled[:, vi, 1].unsqueeze(-1)

            xa = vx * cos_w - vy * sin_w  # (2N, M)
            ya = vx * sin_w + vy * cos_w

            t = 1.0 - xa / Lsd_0
            displ_y = ya + ythis * t
            displ_z = t * zthis

            # Apply NF tilt (no-op when tilts are zero)
            displ_y_tilt, displ_z_tilt = self._apply_nf_tilt(
                displ_y, displ_z, Lsd_0_scalar
            )
            yp = displ_y_tilt / self.px + ybc_0
            zp = displ_z_tilt / self.px + zbc_0
            vert_y_pixel.append(yp)
            vert_z_pixel.append(zp)

        # 6. Compute relative offsets from center (matching C lines 584-586)
        # YZSpots[k] = YZSpotsT[k] - YZSpotsTemp
        rel_y = [vyp - y_center for vyp in vert_y_pixel]  # 3 x (2N, M)
        rel_z = [vzp - z_center for vzp in vert_z_pixel]

        # 7. Rasterize and collect hits
        # This is the per-spot loop (not vectorizable due to variable triangle sizes)
        K = 2 * N
        M = omega.shape[-1]

        all_hits = []  # list of (vox_nr, dist_nr, frame, y_px, z_px, omega_deg)

        for k in range(K):
            for m in range(M):
                if valid[k, m] < 0.5:
                    continue

                # Relative triangle vertices in pixel coords
                ry = [rel_y[vi][k, m].item() for vi in range(3)]
                rz = [rel_z[vi][k, m].item() for vi in range(3)]

                gs_um = tri_config.edge_lengths[k % N].item()
                cr = self._c_round
                if gs_um > self.px:
                    # Rasterize triangle (CalcPixels2 rounds vertices, line 312)
                    v0y, v0z = cr(ry[0]), cr(rz[0])
                    v1y, v1z = cr(ry[1]), cr(rz[1])
                    v2y, v2z = cr(ry[2]), cr(rz[2])
                    pixels = self.rasterize_triangle(v0y, v0z, v1y, v1z, v2y, v2z)
                else:
                    # Single center pixel (C line 596-599: (int)round(...))
                    cy = cr((ry[0] + ry[1] + ry[2]) / 3.0)
                    cz = cr((rz[0] + rz[1] + rz[2]) / 3.0)
                    pixels = [(cy, cz)]

                # Absolute pixel = center + offset, check all distances
                yc = y_center[k, m].item()
                zc = z_center[k, m].item()
                frame = int(frame_nr[k, m].item())  # C uses (int) truncation
                ome_deg = omega[k, m].item() * self.RAD2DEG

                for (py_off, pz_off) in pixels:
                    all_dists_ok = True
                    layer_pixels = []
                    for d in range(D):
                        Lsd_d = self._Lsd[d].item()
                        ybc_d = self._y_BC[d].item()
                        zbc_d = self._z_BC[d].item()
                        # Scale to this distance (matching C lines 605-613)
                        my = int(math.floor(
                            ((yc - ybc_0) * self.px * (Lsd_d / Lsd_0)) / self.px
                            + ybc_d
                        )) + py_off
                        mz = int(math.floor(
                            ((zc - zbc_0) * self.px * (Lsd_d / Lsd_0)) / self.px
                            + zbc_d
                        )) + pz_off
                        if my < 0 or my >= self.n_pixels_y or mz < 0 or mz >= self.n_pixels_z:
                            all_dists_ok = False
                            break
                        layer_pixels.append((d, frame, my, mz))

                    if all_dists_ok:
                        vox_nr = k % N
                        for (d, fr, my, mz) in layer_pixels:
                            all_hits.append((vox_nr, d, fr, my, mz, ome_deg))

        return all_hits
