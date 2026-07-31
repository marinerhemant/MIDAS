"""Detector / sample / beam geometry for midas_defect.

Owns:
  * `Geometry` — small dataclass that captures the subset of MIDAS paramstest
    fields needed for sparse-voxel reciprocal-space work.
  * `Geometry.from_paramstest(path)` — parser for FF paramstest /
    parameters_final.txt files.
  * `demk_default_geometry()` — factory pre-populated with the Demk Sep-2025
    calibration (Lsd=652.7 mm, BC=(698.42, 813.68), λ=0.172979 Å, px=172 µm).
  * `Geometry.as_hedm_geometry()` — convert to `midas_diffract.HEDMGeometry`
    for the q→pixel forward direction.
  * Differentiable forward transforms (built on the canonical MIDAS conventions):
      `pixel_to_qlab(rows, cols, geom, *, device, dtype)`     — pixel → lab q
      `qlab_to_qsample(qlab, omega_rad)`                       — R_z(-ω)
      `qsample_to_qlab(qsample, omega_rad)`                    — R_z(+ω)

Units:
  * Lsd, px  — micrometers (matches MIDAS C and `midas_diffract.HEDMGeometry`)
  * |q|      — 1/Å
  * ω        — radians inside the math, degrees on the dataclass for human-readability

Differentiability:
  * All transforms are torch ops on tensors; (Lsd, BCy, BCz, px, λ) are kept
    as torch scalars so geometry can be jointly refined with downstream
    parameters by P6/P7.
  * Tilt + wedge are torch-native too; we keep them off the default path
    (Demk calibration has tx=0, tz~0.5°, ty~-0.2°, Wedge=0) but the same
    code path supports nonzero tilts.

Convention (matches MIDAS FF, ImTransOpt=0):
  * Lab frame: X = beam direction (+downstream), Y = horizontal, Z = vertical (up).
  * Detector pixel (row=Z-pixel, col=Y-pixel). BCy is the horizontal beam
    centre (column), BCz the vertical (row).
  * Lab coordinates of a pixel (no tilt):
      x_lab = Lsd
      y_lab = -(col - BCy) * px
      z_lab =  (BCz - row) * px
  * ω-rotation axis = lab +Z (vertical rotation stage); sample frame is the
    crystal-rest frame. q_sample = R_z(-ω) @ q_lab (matches MIDAS `_spot_to_gv`:
    z-component invariant, x/y mixed). The detector model (tilt + radial
    distortion) is reused from `midas_transforms.apply_tilt_distortion`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from midas_transforms.device import resolve_device, resolve_dtype


__all__ = [
    "Geometry",
    "demk_default_geometry",
    "pixel_to_qlab",
    "qlab_to_qsample",
    "qsample_to_qlab",
]


# ---------------------------------------------------------------------------
# Geometry dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """Subset of FF paramstest fields needed by midas_defect.

    All distances in micrometers, all angles in degrees. Convert to torch
    tensors on demand via the transform helpers below.
    """
    lsd_um: float                     # sample-detector distance, micrometers
    bcy_px: float                     # horizontal beam centre, pixels (column)
    bcz_px: float                     # vertical beam centre, pixels (row)
    px_um: float                      # pixel pitch, micrometers
    wavelength_A: float               # X-ray wavelength, Å
    n_pix_y: int                      # detector horizontal extent, pixels
    n_pix_z: int                      # detector vertical extent, pixels
    omega_first_deg: float            # ω of frame index 0, degrees
    omega_step_deg: float             # ω step between frames, degrees
    n_frames: int                     # frames per sweep
    tx_deg: float = 0.0               # detector tilt about lab X (deg)
    ty_deg: float = 0.0               # detector tilt about lab Y (deg)
    tz_deg: float = 0.0               # detector tilt about lab Z (deg)
    wedge_deg: float = 0.0            # ω-axis non-orthogonality (deg)
    # MIDAS lens-distortion model, p0..p14 (the full 15-coefficient set consumed
    # by `midas_transforms.apply_tilt_distortion` → `pixel_to_REta_torch`), and
    # the normalization radius RhoD (µm). All-zero p_coeffs ⇒ no distortion.
    # Older calibrations populate only p0..p3 and leave p4..p14 zero.
    p_coeffs: tuple = (0.0,) * 15
    rho_d_um: float = 200000.0
    label: str = ""                   # free-form label for logging

    # ------------------------- factories -----------------------------------
    @classmethod
    def from_paramstest(cls, path: Union[str, Path]) -> "Geometry":
        """Parse a MIDAS-format paramstest.txt or parameters_final.txt.

        Recognized keys (case-sensitive, MIDAS convention):
            Lsd, BC, ty, tz, tx, Wedge, Wavelength, px, NrPixelsY, NrPixelsZ,
            numPxY, numPxZ, OmegaFirstFile, OmegaStart, OmegaStep, NrFilesPerSweep,
            EndNr, StartNr, p0..p14, RhoD.
        Unknown keys are silently ignored. Missing required keys raise.

        The distortion model is read in full (p0..p14). Reading only p0..p3 —
        as this parser did before 2026-07-30 — silently drops the higher-order
        terms of any modern calibration, which shifts predicted spot positions
        without any error being raised.
        """
        path = Path(path)
        fields_: dict[str, list[str]] = {}
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            key, vals = parts[0], parts[1:]
            fields_[key] = vals

        def _get(*names: str) -> Optional[list[str]]:
            for n in names:
                if n in fields_:
                    return fields_[n]
            return None

        def _f(name: str) -> float:
            vals = _get(name)
            if vals is None:
                raise ValueError(f"required key '{name}' missing in {path}")
            return float(vals[0])

        def _opt_f(name: str, default: float) -> float:
            vals = _get(name)
            return float(vals[0]) if vals else default

        bc = _get("BC")
        if bc is None or len(bc) < 2:
            raise ValueError(f"required 'BC <y> <z>' missing or malformed in {path}")
        bcy_px = float(bc[0])
        bcz_px = float(bc[1])

        # n-pixels: accept either MIDAS-old (numPxY/numPxZ) or NF (NrPixelsY/Z)
        npy_vals = _get("NrPixelsY", "numPxY")
        npz_vals = _get("NrPixelsZ", "numPxZ")
        if npy_vals is None or npz_vals is None:
            raise ValueError(
                f"detector pixel dims (NrPixelsY/Z or numPxY/Z) missing in {path}"
            )
        n_pix_y = int(npy_vals[0])
        n_pix_z = int(npz_vals[0])

        # ω fields are sometimes absent in pure-calibration files; default sensibly.
        om_first = _opt_f("OmegaFirstFile", _opt_f("OmegaStart", -180.0))
        om_step  = _opt_f("OmegaStep",  0.25)
        nfr_vals = _get("NrFilesPerSweep", "EndNr")
        n_frames = int(nfr_vals[0]) if nfr_vals else int(round(360.0 / abs(om_step)))

        # Lens distortion: the full MIDAS p0..p14 set (+ RhoD); absent ⇒ no
        # distortion. Older paramstest files carry only p0..p3, which parse to
        # the same tuple with p4..p14 = 0, so this is backwards-compatible.
        p_coeffs = tuple(_opt_f(f"p{i}", 0.0) for i in range(15))
        rho_d_um = _opt_f("RhoD", 200000.0)

        return cls(
            lsd_um=_f("Lsd"),
            bcy_px=bcy_px,
            bcz_px=bcz_px,
            px_um=_f("px"),
            wavelength_A=_f("Wavelength"),
            n_pix_y=n_pix_y,
            n_pix_z=n_pix_z,
            omega_first_deg=om_first,
            omega_step_deg=om_step,
            n_frames=n_frames,
            tx_deg=_opt_f("tx", 0.0),
            ty_deg=_opt_f("ty", 0.0),
            tz_deg=_opt_f("tz", 0.0),
            wedge_deg=_opt_f("Wedge", 0.0),
            p_coeffs=p_coeffs,
            rho_d_um=rho_d_um,
            label=str(path),
        )

    # ------------------------- interop -------------------------------------
    def as_hedm_geometry(self):
        """Return a `midas_diffract.HEDMGeometry` carrying the same fields.

        Use this when handing the geometry to the upstream HEDM forward
        model (e.g., for predicted-spot computations in P2/P6).
        """
        from midas_diffract.forward import HEDMGeometry
        return HEDMGeometry(
            Lsd=self.lsd_um,
            y_BC=self.bcy_px,
            z_BC=self.bcz_px,
            px=self.px_um,
            omega_start=self.omega_first_deg,
            omega_step=self.omega_step_deg,
            n_frames=self.n_frames,
            n_pixels_y=self.n_pix_y,
            n_pixels_z=self.n_pix_z,
            min_eta=6.0,                 # MIDAS default (tunable later)
            wavelength=self.wavelength_A,
            tx=self.tx_deg,
            ty=self.ty_deg,
            tz=self.tz_deg,
            wedge=self.wedge_deg,
        )

    def omega_of_frame(self, frame_idx: "int | torch.Tensor"):
        """Map a 0-based frame index to ω in degrees."""
        return self.omega_first_deg + frame_idx * self.omega_step_deg


def demk_default_geometry() -> Geometry:
    """Hard-coded Demk Sep-2025 calibration (from `parameters_final.txt`).

    Use this when the on-disk paramstest is not available (e.g., on the dev
    laptop). On copland prefer ``Geometry.from_paramstest(...)``.
    """
    return Geometry(
        lsd_um=652665.632540605729,
        bcy_px=698.420227947936,
        bcz_px=813.680034653341,
        px_um=172.0,
        wavelength_A=0.172979,
        n_pix_y=1475,
        n_pix_z=1679,
        # validated ω map: ω = 180 − 0.25·frame (start +180, step −0.25 CW)
        omega_first_deg=180.0,
        omega_step_deg=-0.25,
        n_frames=1440,
        tx_deg=0.095682,
        ty_deg=-0.196484201531,
        tz_deg=0.534276032234,
        wedge_deg=0.0,
        p_coeffs=(0.000230, 0.001234, 0.000211, 32.904494) + (0.0,) * 11,
        rho_d_um=219964.42,
        label="demk_default",
    )


# ---------------------------------------------------------------------------
# torch transforms
# ---------------------------------------------------------------------------

def _g_to_tensors(geom: Geometry, *, dtype: torch.dtype, device: torch.device
                  ) -> dict[str, torch.Tensor]:
    """Promote the differentiable geometry scalars to torch tensors."""
    kw = dict(dtype=dtype, device=device)
    return dict(
        lsd  = torch.as_tensor(geom.lsd_um,        **kw),
        bcy  = torch.as_tensor(geom.bcy_px,        **kw),
        bcz  = torch.as_tensor(geom.bcz_px,        **kw),
        px   = torch.as_tensor(geom.px_um,         **kw),
        lamb = torch.as_tensor(geom.wavelength_A,  **kw),
        tx   = torch.as_tensor(geom.tx_deg,        **kw),
        ty   = torch.as_tensor(geom.ty_deg,        **kw),
        tz   = torch.as_tensor(geom.tz_deg,        **kw),
    )


def _rot_x(angle_rad: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    zero = torch.zeros_like(angle_rad)
    one = torch.ones_like(angle_rad)
    return torch.stack([
        torch.stack([one,  zero, zero]),
        torch.stack([zero, c,   -s  ]),
        torch.stack([zero, s,    c  ]),
    ])


def _rot_y(angle_rad: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    zero = torch.zeros_like(angle_rad)
    one = torch.ones_like(angle_rad)
    return torch.stack([
        torch.stack([ c,  zero, s  ]),
        torch.stack([zero, one, zero]),
        torch.stack([-s,  zero, c  ]),
    ])


def _rot_z(angle_rad: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    zero = torch.zeros_like(angle_rad)
    one = torch.ones_like(angle_rad)
    return torch.stack([
        torch.stack([c,  -s,   zero]),
        torch.stack([s,   c,   zero]),
        torch.stack([zero, zero, one]),
    ])


def pixel_to_qlab(
    rows: "torch.Tensor | np.ndarray | Sequence[float]",
    cols: "torch.Tensor | np.ndarray | Sequence[float]",
    geom: Geometry,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.Tensor:
    """Detector pixel (row, col) → lab-frame q-vector (1/Å).

    Returns
    -------
    Tensor of shape `(..., 3)` containing `(qlx, qly, qlz)`.

    Differentiable through every Geometry scalar field (Lsd, BCy, BCz, px,
    λ, tilts). Inputs `rows`/`cols` are NOT differentiable (they're pixel
    indices).
    """
    from midas_transforms.fit_setup.transform import apply_tilt_distortion

    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)
    rows_t = torch.as_tensor(rows, dtype=dtype_, device=device_)   # Z_pix
    cols_t = torch.as_tensor(cols, dtype=dtype_, device=device_)   # Y_pix

    g = _g_to_tensors(geom, dtype=dtype_, device=device_)
    p_coeffs = torch.as_tensor(geom.p_coeffs, dtype=dtype_, device=device_)
    rho_d = torch.as_tensor(geom.rho_d_um, dtype=dtype_, device=device_)

    # MIDAS-canonical detector model: tilt (R_z R_y R_x) + radial distortion,
    # reused from midas_transforms (no re-ported geometry). Returns lab (Y, Z)
    # in microns at the nominal sample-detector distance Lsd.
    Yl, Zl = apply_tilt_distortion(
        cols_t, rows_t, Lsd=g["lsd"], BC_y=g["bcy"], BC_z=g["bcz"],
        tx=g["tx"], ty=g["ty"], tz=g["tz"], p_coeffs=p_coeffs, px=g["px"],
        rho_d=rho_d,
    )
    x_um = g["lsd"].expand_as(Yl) if Yl.ndim > 0 else g["lsd"]
    p_lab = torch.stack([x_um, Yl, Zl], dim=-1)

    # q = (2π/λ) (k_f - k_i), with k_i = +X
    norm = torch.linalg.vector_norm(p_lab, dim=-1, keepdim=True)
    k_f = p_lab / norm
    k_i = torch.zeros_like(k_f)
    k_i[..., 0] = 1.0
    k0 = 2.0 * math.pi / g["lamb"]
    return k0 * (k_f - k_i)


def qlab_to_qsample(
    qlab: torch.Tensor,
    omega_rad: "torch.Tensor | float",
) -> torch.Tensor:
    """Rotate `qlab` (..., 3) into the sample frame.

    Convention (matches MIDAS `_spot_to_gv`): the ω-rotation axis is the lab +Z
    (vertical) — the rotation-stage axis — so `q_sample = R_z(-ω) @ q_lab`. This
    leaves the vertical (z) component invariant and mixes the beam (x) and
    horizontal (y) components, exactly as the FF indexer does. Differentiable
    through ω.
    """
    omega_t = torch.as_tensor(omega_rad, dtype=qlab.dtype, device=qlab.device)
    c = torch.cos(-omega_t)
    s = torch.sin(-omega_t)
    qx = c * qlab[..., 0] - s * qlab[..., 1]
    qy = s * qlab[..., 0] + c * qlab[..., 1]
    qz = qlab[..., 2]
    return torch.stack([qx, qy, qz], dim=-1)


def qsample_to_qlab(
    qsample: torch.Tensor,
    omega_rad: "torch.Tensor | float",
) -> torch.Tensor:
    """Inverse of `qlab_to_qsample`: q_lab = R_z(+ω) @ q_sample."""
    omega_t = torch.as_tensor(omega_rad, dtype=qsample.dtype, device=qsample.device)
    c = torch.cos(omega_t)
    s = torch.sin(omega_t)
    qx = c * qsample[..., 0] - s * qsample[..., 1]
    qy = s * qsample[..., 0] + c * qsample[..., 1]
    qz = qsample[..., 2]
    return torch.stack([qx, qy, qz], dim=-1)


# ---------------------------------------------------------------------------
# Ewald-sphere crossing predictor (pure NumPy; convention matches the transforms
# above -- beam k_i = +X, rotation axis lab +Z, q_lab = R_z(+ω) @ q_sample).
# ---------------------------------------------------------------------------

def ewald_crossing_omegas(
    q_sample: "np.ndarray | Sequence[float]",
    wavelength_A: float,
) -> "np.ndarray":
    """The two ω (radians) at which a sample-fixed reflection crosses the Ewald sphere.

    A reciprocal-lattice vector fixed to the crystal (``q_sample``, 1/Å) is carried
    through the lab by the rotation ``q_lab(ω) = R_z(+ω) @ q_sample`` (see
    :func:`qsample_to_qlab`). It diffracts when the elastic condition holds, which
    with the beam along +X and ``k0 = 2π/λ`` reduces to

        q_lab_x(ω) = -|q|^2 / (2 k0).

    Writing ``q_lab_x = qx cos ω - qy sin ω`` this is ``A cos ω + B sin ω = C`` with
    ``A = qx``, ``B = -qy``, ``C = -|q|^2/(2 k0)`` -- generally **two** solutions ω,
    the reflection's two Ewald crossings per 360° turn. Both carry the *same* ``q``
    (hence the same ``|F|^2`` and, provably, the same Lorentz-polarization factor),
    so a kinematic intensity is equal at the two crossings.

    Returns a length-2 array of ω in radians in ``(-π, π]``, sorted ascending. If
    the reflection never satisfies the condition (``|C| > sqrt(A^2+B^2)`` -- it lies
    in the blind region for this |q|/λ), returns an empty array. The two values are
    identical (tangency) when ``|C| = sqrt(A^2+B^2)``.
    """
    q = np.asarray(q_sample, dtype=np.float64).reshape(3)
    qmag = float(np.linalg.norm(q))
    if qmag == 0.0:
        return np.empty(0, dtype=np.float64)
    k0 = 2.0 * math.pi / wavelength_A
    A = q[0]
    B = -q[1]
    C = -(qmag * qmag) / (2.0 * k0)
    R = math.hypot(A, B)
    if R == 0.0 or abs(C) > R * (1.0 + 1e-12):
        return np.empty(0, dtype=np.float64)
    phi = math.atan2(B, A)
    d = math.acos(max(-1.0, min(1.0, C / R)))
    sols = np.array([phi - d, phi + d], dtype=np.float64)
    sols = (sols + math.pi) % (2.0 * math.pi) - math.pi   # wrap to (-π, π]
    return np.sort(sols)


def ewald_crossings(
    q_sample: "np.ndarray | Sequence[float]",
    wavelength_A: float,
) -> list[dict]:
    """Both Ewald crossings of ``q_sample`` with their lab-frame q-vectors.

    Returns one dict per crossing with keys ``omega_rad``, ``omega_deg``, and
    ``q_lab`` (the ``(3,)`` diffracting lab vector ``R_z(+ω) @ q_sample``). Empty
    list if the reflection never diffracts. Useful for matching predicted crossing
    ω / detector side against observed satellite pairs (the two-crossing model that
    underlies the unified satellite indexing).
    """
    q = np.asarray(q_sample, dtype=np.float64).reshape(3)
    out = []
    for w in ewald_crossing_omegas(q, wavelength_A):
        c, s = math.cos(w), math.sin(w)
        q_lab = np.array([c * q[0] - s * q[1], s * q[0] + c * q[1], q[2]])
        out.append({
            "omega_rad": float(w),
            "omega_deg": float(math.degrees(w)),
            "q_lab": q_lab,
        })
    return out
