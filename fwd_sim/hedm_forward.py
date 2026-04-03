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
    y_BC: "float | list[float]"    # Beam center y (pixels), per distance
    z_BC: "float | list[float]"    # Beam center z (pixels), per distance
    px: float                      # Pixel size (um) -- shared across distances
    omega_start: float             # Omega start (degrees)
    omega_step: float              # Omega step (degrees, may be negative)
    n_frames: int                  # Frames per distance (NrFilesPerDistance)
    n_pixels_y: int                # Detector pixels in y
    n_pixels_z: int                # Detector pixels in z
    min_eta: float                 # Minimum eta angle (degrees)
    wavelength: float = 0.0        # X-ray wavelength (Angstroms)
    flip_y: bool = True            # FF/PF: True (DetHor = yBC - ydet/px).
                                   # NF:    False (pixel = yBC + ydet/px).
                                   # Validated against C code conventions.

    @property
    def n_distances(self) -> int:
        return len(self.Lsd) if isinstance(self.Lsd, list) else 1

    def _as_list(self, attr):
        v = getattr(self, attr)
        return v if isinstance(v, list) else [v]


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

    For multi-distance NF-HEDM, ``y_pixel``, ``z_pixel``, and
    ``layer_valid`` have an extra leading ``D`` (n_distances) dimension.
    The ``valid`` mask combines the angular validity with ALL-distances-found.
    """
    omega: torch.Tensor                      # (..., K, M) radians
    eta: torch.Tensor                        # (..., K, M) radians
    two_theta: torch.Tensor                  # (..., K, M) radians
    y_pixel: torch.Tensor                    # (D, ..., K, M) or (..., K, M) fractional pixel
    z_pixel: torch.Tensor                    # (D, ..., K, M) or (..., K, M) fractional pixel
    frame_nr: torch.Tensor                   # (..., K, M) fractional frame (same at all distances)
    valid: torch.Tensor                      # (..., K, M) float mask (1=valid at ALL distances)
    layer_valid: Optional[torch.Tensor] = None  # (D, ..., K, M) per-distance validity
    scan_mask: Optional[torch.Tensor] = None    # (..., S, K, M) per-beam-position validity (pf)


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

    # Use the EXACT same hardcoded constants as the C code (SharedFuncsFit.c,
    # GetMisorientation.c, etc.) rather than M_PI/180.  The C code's constant
    # has 13 significant digits and is ~3.5e-18 larger than M_PI/180.
    # This difference causes ~2.2e-16 error in cos/sin at 60°/30°, which
    # can flip pixel assignments at integer boundaries.
    DEG2RAD = 0.0174532925199433
    RAD2DEG = 57.2957795130823

    def __init__(
        self,
        hkls: torch.Tensor,
        thetas: torch.Tensor,
        geometry: HEDMGeometry,
        hkls_int: Optional[torch.Tensor] = None,
        scan_config: Optional[ScanConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
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
        self.register_buffer("_Lsd", torch.tensor(Lsd_list, dtype=torch.float32, device=device))
        self.register_buffer("_y_BC", torch.tensor(yBC_list, dtype=torch.float32, device=device))
        self.register_buffer("_z_BC", torch.tensor(zBC_list, dtype=torch.float32, device=device))
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

        # Scan config
        self.scan_config = scan_config
        if scan_config is not None:
            self.register_buffer(
                "_beam_positions", scan_config.beam_positions.to(device).float()
            )
            self._beam_size = scan_config.beam_size

        self.epsilon = 1e-7

    # ------------------------------------------------------------------
    #  euler2mat  (ZXZ convention)
    # ------------------------------------------------------------------

    @staticmethod
    def euler2mat(euler_angles: torch.Tensor) -> torch.Tensor:
        """Convert ZXZ Euler angles to rotation matrices.

        Parameters
        ----------
        euler_angles : Tensor (..., 3)
            Euler angles (phi1, Phi, phi2) in radians.

        Returns
        -------
        Tensor (..., 3, 3)
            Rotation matrices.
        """
        c = torch.cos(euler_angles)
        s = torch.sin(euler_angles)

        c0, c1, c2 = c[..., 0], c[..., 1], c[..., 2]
        s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]

        # ZXZ rotation matrix: R = Rz(phi1) @ Rx(Phi) @ Rz(phi2)
        # Verified element-by-element against nfhedm.py lines 114-120
        R = torch.zeros(*euler_angles.shape[:-1], 3, 3,
                        dtype=euler_angles.dtype, device=euler_angles.device)
        R[..., 0, 0] =  c0 * c2 - s0 * c1 * s2
        R[..., 0, 1] = -s0 * c1 * c2 - c0 * s2
        R[..., 0, 2] =  s0 * s1
        R[..., 1, 0] =  s0 * c2 + c0 * c1 * s2
        R[..., 1, 1] =  c0 * c1 * c2 - s0 * s2
        R[..., 1, 2] = -c0 * s1
        R[..., 2, 0] =  s1 * s2
        R[..., 2, 1] =  s1 * c2
        R[..., 2, 2] =  c1

        return R

    # ------------------------------------------------------------------
    #  safe_arccos
    # ------------------------------------------------------------------

    def safe_arccos(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable arccos: clamp to [-1+eps, 1-eps]."""
        return torch.acos(torch.clamp(x, -1.0 + self.epsilon, 1.0 - self.epsilon))

    # ------------------------------------------------------------------
    #  correct_hkls_latc  (port of C CorrectHKLsLatC)
    # ------------------------------------------------------------------

    def correct_hkls_latc(
        self, lattice_params: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute strained reciprocal-space G-vectors and Bragg angles.

        Builds the reciprocal lattice B matrix from lattice parameters
        and transforms integer Miller indices to Cartesian G-vectors.

        Faithfully ports ``CorrectHKLsLatC`` from
        ``FF_HEDM/src/FitPosOrStrainsDoubleDataset.c:214-252``.

        Parameters
        ----------
        lattice_params : Tensor (..., 6)
            [a, b, c, alpha, beta, gamma] in Angstroms and degrees.
            The ``...`` dimensions allow per-voxel or per-grain parameters.

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

        # G_cart = B @ hkls_int^T  =>  (..., M, 3)
        # hkls_int is (M, 3), B is (..., 3, 3)
        hkls_cart = torch.einsum("...ij,mj->...mi", B, self.hkls_int)

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
        G_C = torch.einsum("...nij,mj->...nmi", orientation_matrices, hkls_cart)

        # v = sin(theta)*|G| -- C precomputes Gs from the UNROTATED G-vector norm
        # (rotation preserves norm in exact arithmetic but not in float64).
        # Match C: use |hkls_cart| (pre-rotation), not |R @ hkls_cart|.
        len_hkl = torch.norm(hkls_cart, dim=-1)  # (M,) or (..., M)
        v = torch.sin(thetas) * len_hkl  # (M,) or (..., M)
        v = v.unsqueeze(-2).expand_as(G_C[..., 0])  # (..., N, M)

        # Extract components
        Gx = G_C[..., 0]  # (..., N, M)
        Gy = G_C[..., 1]
        Gz = G_C[..., 2]

        # Quadratic solver for omega
        # -Gx*cos(w) + Gy*sin(w) = v
        # Rearranged: a*cos^2(w) + b*cos(w) + c = 0
        # C uses almostzero=1e-4 for the Gy≈0 branch. Must match exactly.
        almostzero = 1e-4
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

        # Special case: Gy ~ 0 (C uses almostzero=1e-4)
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

        # Build omega rotation matrix for each spot
        cos_w = torch.cos(all_omega)
        sin_w = torch.sin(all_omega)

        # Construct Rz(omega): rotation around z-axis
        # [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
        Omega_mat = torch.zeros(*all_omega.shape, 3, 3,
                                dtype=all_omega.dtype, device=all_omega.device)
        Omega_mat[..., 0, 0] = cos_w
        Omega_mat[..., 0, 1] = -sin_w
        Omega_mat[..., 1, 0] = sin_w
        Omega_mat[..., 1, 1] = cos_w
        Omega_mat[..., 2, 2] = 1.0

        # Rotate G_C by omega: nrot = Omega @ G_C
        # G_C is (..., N, M, 3); double along N dim (dim=-3)
        G_C_doubled = torch.cat([G_C, G_C], dim=-3)  # (..., 2N, M, 3)
        nrot = torch.einsum("...kmij,...kmj->...kmi", Omega_mat, G_C_doubled)
        nrot_y = nrot[..., 1]  # (..., 2N, M)
        nrot_z = nrot[..., 2]

        # Eta angle
        r_yz = torch.sqrt(nrot_y * nrot_y + nrot_z * nrot_z).clamp(min=self.epsilon)
        eta = self.safe_arccos(nrot_z / r_yz)
        eta = -torch.sign(nrot_y) * eta

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
        Lsd_d = self._Lsd.to(dtype).reshape(D, *([1] * extra_dims))
        yBC_d = self._y_BC.to(dtype).reshape(D, *([1] * extra_dims))
        zBC_d = self._z_BC.to(dtype).reshape(D, *([1] * extra_dims))

        # ydet, zdet, y_pixel, z_pixel all get shape (D, ..., 2N, M)
        dist_d = Lsd_d - x_grain.unsqueeze(0)  # (D, ..., 2N, M)
        ydet_d = y_grain.unsqueeze(0) - dist_d * tan_2th.unsqueeze(0) * sin_eta.unsqueeze(0)
        zdet_d = z_grain.unsqueeze(0) + dist_d * tan_2th.unsqueeze(0) * cos_eta.unsqueeze(0)

        # FF/PF: y-axis on detector flipped (yBC - ydet/px), validated against C
        # NF:    not flipped (yBC + ydet/px), validated against C
        y_sign = -1.0 if self.flip_y else 1.0
        y_pixel_d = yBC_d + y_sign * ydet_d / self.px  # (D, ..., 2N, M)
        z_pixel_d = zBC_d + zdet_d / self.px

        # Per-distance detector bounds
        layer_bounds_ok = (
            (y_pixel_d >= 0) & (y_pixel_d < self.n_pixels_y) &
            (z_pixel_d >= 0) & (z_pixel_d < self.n_pixels_z)
        )  # (D, ..., 2N, M)

        # Per-distance validity = angular valid & frame ok & detector bounds
        layer_valid = valid.unsqueeze(0) * frame_ok.unsqueeze(0).float() * layer_bounds_ok.float()

        # Overall valid = valid at ALL distances (AllDistsFound)
        all_dists_valid = layer_valid.prod(dim=0)  # (..., 2N, M)

        # For single-distance, squeeze out the D dimension for convenience
        if D == 1:
            y_pixel_out = y_pixel_d.squeeze(0)
            z_pixel_out = z_pixel_d.squeeze(0)
            layer_valid_out = None
        else:
            y_pixel_out = y_pixel_d
            z_pixel_out = z_pixel_d
            layer_valid_out = layer_valid

        return SpotDescriptors(
            omega=omega,
            eta=eta,
            two_theta=two_theta,
            y_pixel=y_pixel_out,
            z_pixel=z_pixel_out,
            frame_nr=frame_nr,
            valid=all_dists_valid,
            layer_valid=layer_valid_out,
        )

    # ------------------------------------------------------------------
    #  forward  (orchestrator)
    # ------------------------------------------------------------------

    def forward(
        self,
        euler_angles: torch.Tensor,
        positions: torch.Tensor,
        lattice_params: Optional[torch.Tensor] = None,
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
            hkls_cart, thetas = self.correct_hkls_latc(lattice_params)

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

        # 1. Compute orientation matrices using C's exact chain.
        # simulateNF reads radians from .mic, multiplies by hardcoded rad2deg
        # (57.2957795130823), then Euler2OrientMat uses cosd()=cos(deg2rad*x)
        # where deg2rad=0.0174532925199433. Since deg2rad*rad2deg != 1.0
        # (product = 0.999999999999999889), the roundtrip causes the matrix
        # to be slightly non-orthogonal (det ≈ 0.99999999999999978).
        # We must replicate this to get bit-identical Gc values.
        euler_deg_c = euler_angles * self.RAD2DEG
        euler_rad_c = euler_deg_c * self.DEG2RAD
        orientation_matrices = self.euler2mat(euler_rad_c)

        # 2. Optionally strained HKLs
        hkls_cart = thetas = None
        if lattice_params is not None:
            hkls_cart, thetas = self.correct_hkls_latc(lattice_params)

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

        # Project center reference (with tx=ty=tz=0, no RotMatTilts):
        # outxyz[1] = ythis, outxyz[2] = zthis (identity tilt)
        # YZSpotsTemp = outxyz/px + bc
        ybc_0 = self._y_BC[0].to(dtype)
        zbc_0 = self._z_BC[0].to(dtype)
        y_center = ythis / self.px + ybc_0  # (2N, M)
        z_center = zthis / self.px + zbc_0

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

            # To pixel (with tx=ty=tz=0, no tilt): pixel = displ/px + bc
            yp = displ_y / self.px + ybc_0
            zp = displ_z / self.px + zbc_0
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
