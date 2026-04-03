"""MLEM / OS-EM sinogram reconstruction for pf-HEDM.

Replaces FBP (gridrec) for sparse sinograms where data exists only
at discrete omega angles, not a continuous 0-360 sweep.

MLEM naturally handles:
  - Missing projections (only uses sinogram rows that contain data)
  - Non-uniform angular sampling
  - Positivity constraint (inherently produces >= 0 reconstructions)
  - Poisson noise model (better than FBP's implicit Gaussian assumption)

Usage:
    from mlem_recon import mlem, osem, forward_project, back_project

    recon = mlem(sinogram, angles_deg, n_iter=50)
    # or with ordered subsets for speed:
    recon = osem(sinogram, angles_deg, n_iter=10, n_subsets=4)
"""

import numpy as np
from typing import Optional


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def forward_project(
    image: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """Radon transform: image -> sinogram.

    Projects a 2D image along each angle using bilinear interpolation.

    Parameters
    ----------
    image : ndarray (N, N)
        2D image to project.
    angles_deg : ndarray (nThetas,)
        Projection angles in degrees.

    Returns
    -------
    ndarray (nThetas, N) -- sinogram.
    """
    N = image.shape[0]
    assert image.shape[1] == N, "Image must be square"
    nThetas = len(angles_deg)
    sino = np.zeros((nThetas, N), dtype=np.float64)

    # Center coordinates
    center = (N - 1) / 2.0
    x = np.arange(N) - center  # detector pixel positions

    for i, angle in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # For each detector pixel t, integrate along the ray
        for j, t in enumerate(x):
            # Ray: (t*cos_a - s*sin_a, t*sin_a + s*cos_a) for s in [-N/2, N/2]
            # Sample along the ray
            s_vals = x  # use same sampling as pixel grid
            ray_x = t * cos_a - s_vals * sin_a + center
            ray_y = t * sin_a + s_vals * cos_a + center

            # Bilinear interpolation
            ix = np.floor(ray_x).astype(int)
            iy = np.floor(ray_y).astype(int)
            fx = ray_x - ix
            fy = ray_y - iy

            # Bounds check
            valid = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
            ix = np.clip(ix, 0, N - 2)
            iy = np.clip(iy, 0, N - 2)

            vals = (
                image[iy, ix] * (1 - fx) * (1 - fy) +
                image[iy, ix + 1] * fx * (1 - fy) +
                image[iy + 1, ix] * (1 - fx) * fy +
                image[iy + 1, ix + 1] * fx * fy
            )
            sino[i, j] = np.sum(vals * valid)

    return sino


def back_project(
    sinogram: np.ndarray,
    angles_deg: np.ndarray,
    N: int,
) -> np.ndarray:
    """Adjoint Radon transform: sinogram -> image (unfiltered back-projection).

    Parameters
    ----------
    sinogram : ndarray (nThetas, M)
        Sinogram data.
    angles_deg : ndarray (nThetas,)
        Projection angles in degrees.
    N : int
        Output image size (N x N).

    Returns
    -------
    ndarray (N, N) -- back-projected image.
    """
    M = sinogram.shape[1]
    image = np.zeros((N, N), dtype=np.float64)

    center_img = (N - 1) / 2.0
    center_det = (M - 1) / 2.0

    # Pixel coordinates
    yy, xx = np.mgrid[0:N, 0:N]
    xx = xx.astype(np.float64) - center_img
    yy = yy.astype(np.float64) - center_img

    for i, angle in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Project pixel position onto detector
        t = xx * cos_a + yy * sin_a + center_det

        # Linear interpolation into sinogram row
        it = np.floor(t).astype(int)
        ft = t - it

        valid = (it >= 0) & (it < M - 1)
        it_safe = np.clip(it, 0, M - 2)

        vals = sinogram[i, it_safe] * (1 - ft) + sinogram[i, it_safe + 1] * ft
        image += vals * valid

    return image


def mlem(
    sinogram: np.ndarray,
    angles_deg: np.ndarray,
    n_iter: int = 50,
    init: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Maximum Likelihood Expectation Maximization reconstruction.

    Iteratively refines an estimate by:
        1. Forward project current estimate
        2. Compute ratio: measured / projected
        3. Back-project the ratio
        4. Multiply estimate by (back-projected ratio / sensitivity)

    Only uses sinogram rows with non-zero data (handles missing angles).

    Parameters
    ----------
    sinogram : ndarray (nThetas, M)
        Measured sinogram. Rows with all zeros are treated as missing.
    angles_deg : ndarray (nThetas,)
        Projection angles in degrees.
    n_iter : int
        Number of iterations.
    init : ndarray (M, M), optional
        Initial estimate. Default: uniform positive image.
    mask : ndarray (nThetas, M) bool, optional
        Which sinogram entries to use. Default: non-zero entries.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    ndarray (M, M) -- reconstructed image.
    """
    nThetas, M = sinogram.shape
    N = M  # square reconstruction

    # Filter to non-zero rows (valid projections)
    if mask is None:
        row_has_data = np.any(sinogram > 0, axis=1)
    else:
        row_has_data = np.any(mask, axis=1)

    valid_idx = np.where(row_has_data)[0]
    sino_valid = sinogram[valid_idx]
    angles_valid = angles_deg[valid_idx]

    if len(valid_idx) == 0:
        return np.zeros((N, N))

    # Sensitivity image: back-projection of ones (normalization factor)
    ones_sino = np.ones_like(sino_valid)
    sensitivity = back_project(ones_sino, angles_valid, N)
    sensitivity = np.maximum(sensitivity, eps)

    # Initial estimate
    if init is not None:
        estimate = init.copy()
    else:
        estimate = np.ones((N, N), dtype=np.float64)

    for iteration in range(n_iter):
        # Forward project current estimate
        proj = forward_project(estimate, angles_valid)
        proj = np.maximum(proj, eps)

        # Ratio
        ratio = sino_valid / proj

        # Back-project ratio
        correction = back_project(ratio, angles_valid, N)

        # Update: multiplicative (preserves non-negativity)
        estimate *= correction / sensitivity

    return estimate


def osem(
    sinogram: np.ndarray,
    angles_deg: np.ndarray,
    n_iter: int = 10,
    n_subsets: int = 4,
    init: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Ordered Subsets Expectation Maximization (accelerated MLEM).

    Divides projections into subsets and updates after each subset,
    converging ~n_subsets times faster than standard MLEM.

    Parameters
    ----------
    sinogram : ndarray (nThetas, M)
    angles_deg : ndarray (nThetas,)
    n_iter : int
        Number of full iterations (each processes all subsets).
    n_subsets : int
        Number of ordered subsets.
    init : ndarray (M, M), optional
    eps : float

    Returns
    -------
    ndarray (M, M)
    """
    nThetas, M = sinogram.shape
    N = M

    row_has_data = np.any(sinogram > 0, axis=1)
    valid_idx = np.where(row_has_data)[0]

    if len(valid_idx) == 0:
        return np.zeros((N, N))

    # Divide valid indices into subsets (interleaved for even angular coverage)
    subsets = [valid_idx[i::n_subsets] for i in range(n_subsets)]

    if init is not None:
        estimate = init.copy()
    else:
        estimate = np.ones((N, N), dtype=np.float64)

    for iteration in range(n_iter):
        for subset_idx in subsets:
            sino_sub = sinogram[subset_idx]
            angles_sub = angles_deg[subset_idx]

            sensitivity = back_project(np.ones_like(sino_sub), angles_sub, N)
            sensitivity = np.maximum(sensitivity, eps)

            proj = forward_project(estimate, angles_sub)
            proj = np.maximum(proj, eps)

            ratio = sino_sub / proj
            correction = back_project(ratio, angles_sub, N)

            estimate *= correction / sensitivity

    return estimate
