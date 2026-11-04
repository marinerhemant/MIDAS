"""End-to-end diffraction-spot prediction pipeline.

Mirrors the per-orientation loop in ``MakeDiffrSpots.c::CalcDiffrSpots_Furnace``
(L214-L269) and the orchestration in ``main`` (L539-L630), vectorized over all
orientations and HKLs at once.

The pipeline is differentiable in the input quaternions (and downstream in
the orientation matrices and lattice parameters that produced ``hkls``). The
hard ``OmegaRange``/``BoxSize``/``ExcludePoleAngle`` filters return a
``valid`` mask, mirroring the ``midas_diffract`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import logging

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from .binary_io import write_all, write_key_bin, write_orient_mat_bin
from .geometry import bragg_omega_eta, calc_spot_position
from .hkls import read_hkls_csv, read_seed_orientations
from .orientations import quat_to_orient_matrix
from .params import DiffrSpotsParams


@dataclass
class DiffrSpotsResult:
    """Output bundle of the diffr_spots pipeline.

    Layout:
      - ``omegas, etas, yls, zls`` are ``(N_orient, M_hkls, 2)`` -- the trailing
        2-axis is the two omega solutions per HKL.
      - ``valid`` is a bool tensor of the same shape combining all the filters.
      - ``orient_mats`` is ``(N_orient, 3, 3)``.
      - ``counts`` is ``(N_orient,)`` int with the per-orientation valid count.

    Convenience views:
      - ``flat_spots()``  -> ``(TotalSpots, 3)`` float tensor of valid (yl, zl, omega)
        in the order MakeDiffrSpots writes to ``DiffractionSpots.bin``.
      - ``offsets()``     -> ``(N_orient,)`` int tensor of cumulative offsets.
    """

    omegas: torch.Tensor
    etas: torch.Tensor
    yls: torch.Tensor
    zls: torch.Tensor
    valid: torch.Tensor
    orient_mats: torch.Tensor

    @property
    def n_orientations(self) -> int:
        return int(self.orient_mats.shape[0])

    @property
    def counts(self) -> torch.Tensor:
        """Per-orientation valid spot counts."""
        return self.valid.reshape(self.n_orientations, -1).sum(dim=1).to(torch.int64)

    def offsets(self) -> torch.Tensor:
        c = self.counts
        out = torch.zeros_like(c)
        if c.numel() > 1:
            out[1:] = torch.cumsum(c[:-1], dim=0)
        return out

    def flat_spots(self) -> torch.Tensor:
        """Return ``(TotalSpots, 3)`` of valid (yl, zl, omega) rows.

        Output ordering matches MakeDiffrSpots: orientation-major, with HKLs
        within an orientation appearing in the order they are listed in the
        input HKL table, two solutions per HKL.
        """
        # One boolean mask over the (N, 2M) view: row-major selection is orientation-major with the HKLs in
        # input order, exactly the order of the former per-orientation loop, but with no per-orientation Python
        # loop and no second copy of the data (the loop + torch.cat needed ~3x the output size on the device:
        # 972,439 orientations x 442 hkls ran out of a 47 GB card here).
        N = self.n_orientations
        if N == 0:
            return torch.empty((0, 3), dtype=self.yls.dtype, device=self.yls.device)
        mask = self.valid.reshape(N, -1)
        return torch.stack(
            [self.yls.reshape(N, -1)[mask], self.zls.reshape(N, -1)[mask], self.omegas.reshape(N, -1)[mask]],
            dim=1,
        )


LOGGER = logging.getLogger(__name__)


def _predict_spots_chunk(
    quaternions: torch.Tensor,
    hkls: torch.Tensor,
    thetas_deg: torch.Tensor,
    distance: Union[float, torch.Tensor],
    *,
    omega_ranges: Optional[Sequence[Sequence[float]]] = None,
    box_sizes: Optional[Sequence[Sequence[float]]] = None,
    exclude_pole_angle: float = 0.0,
    wedge_deg: float = 0.0,
) -> DiffrSpotsResult:
    """Forward-simulate diffraction spots for a set of orientations.

    Bragg geometry (omega/eta solver) is delegated to
    ``midas_diffract.HEDMForwardModel.calc_bragg_geometry`` to keep one
    canonical implementation. This package adds the lab-frame ``(yl, zl)``
    projection, the OmegaRange/BoxSize/ExcludePoleAngle filters, and the
    binary I/O that MakeDiffrSpots needs.

    All inputs/outputs are PyTorch tensors. The returned tensors are
    autograd-tracked through ``quaternions``, ``hkls``, ``thetas_deg``, and
    ``distance``; the validity mask is detached.

    Parameters
    ----------
    quaternions : Tensor of shape ``(N, 4)`` -- ``(w, x, y, z)``.
    hkls        : Tensor of shape ``(M, 3)`` -- reciprocal-space G-vectors.
    thetas_deg  : Tensor of shape ``(M,)`` -- Bragg angles in degrees.
    distance    : Detector distance in micrometers (Lsd[0]).
    omega_ranges, box_sizes : paired omega/box filters; a spot passes if it
                  is inside *any* pair (matches MakeDiffrSpots.c L248-L259).
    exclude_pole_angle : skip spots whose ``|eta|`` is within this many degrees
                  of 0 or 180 (matches L245).
    wedge_deg   : the Parameters-file ``Wedge`` (degrees), passed through to
                  midas_diffract (``midas_diffract.forward`` "Wedge
                  convention": rotation axis ``(-sin W, 0, cos W)``,
                  orientations in the rotation-stage frame, the same frame
                  as FF Grains.csv). ``DiffrSpotsPipeline`` reads it from the
                  parameter file. 0 matches the original C MakeDiffrSpots.

    Returns
    -------
    DiffrSpotsResult.
    """
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) quaternions, got shape {tuple(quaternions.shape)}"
        )
    if hkls.ndim != 2 or hkls.shape[1] != 3:
        raise ValueError(
            f"Expected (M, 3) hkls, got shape {tuple(hkls.shape)}"
        )
    if thetas_deg.shape != (hkls.shape[0],):
        raise ValueError(
            f"thetas_deg shape {tuple(thetas_deg.shape)} must equal (M,)={(hkls.shape[0],)}"
        )

    device = quaternions.device
    dtype = quaternions.dtype

    # Build orientation matrices: (N, 3, 3). Quaternion->matrix is the one
    # primitive midas_diffract does NOT expose (it has euler->matrix), so we
    # keep our own implementation.
    orient_mats = quat_to_orient_matrix(quaternions)

    # Bragg quadratic + eta: delegated to midas_diffract.
    omegas_n2m, etas_n2m, valid_n2m = bragg_omega_eta(
        orient_mats,
        hkls.to(device=device, dtype=dtype),
        thetas_deg.to(device=device, dtype=dtype),
        distance_um=float(distance) if not isinstance(distance, torch.Tensor) else float(distance.item()),
        wedge_deg=wedge_deg,
        device=device,
    )
    # midas_diffract returns (N, 2, M); the rest of this package uses (N, M, 2)
    # so the trailing 2 names "the two omega solutions per HKL" -- matches the
    # natural per-HKL grouping in MakeDiffrSpots.
    omegas = omegas_n2m.transpose(-2, -1).contiguous()
    etas = etas_n2m.transpose(-2, -1).contiguous()
    geom_valid = valid_n2m.transpose(-2, -1).contiguous()

    # Spot positions per (N, M, 2)
    thetas_b = thetas_deg.to(device=device, dtype=dtype).view(1, -1, 1).expand_as(omegas)
    distance_t = (
        distance
        if isinstance(distance, torch.Tensor)
        else torch.tensor(float(distance), device=device, dtype=dtype)
    )
    ring_radius_b = distance_t * torch.tan(2.0 * thetas_b * (torch.pi / 180.0))
    yls, zls = calc_spot_position(ring_radius_b, etas)

    # ExcludePoleAngle: |eta| < EPA  or  (180 - |eta|) < EPA
    eta_abs = etas.abs()
    pole_ok = (eta_abs >= exclude_pole_angle) & ((180.0 - eta_abs) >= exclude_pole_angle)

    # OmegaRange + BoxSize: spot is valid if ANY (range_i, box_i) pair accepts.
    if omega_ranges and box_sizes:
        if len(omega_ranges) != len(box_sizes):
            raise ValueError(
                f"omega_ranges (len {len(omega_ranges)}) must match box_sizes "
                f"(len {len(box_sizes)})"
            )
        any_ok = torch.zeros_like(geom_valid)
        for (om_lo, om_hi), (yl_lo, yl_hi, zl_lo, zl_hi) in zip(omega_ranges, box_sizes):
            ok = (
                (omegas > om_lo)
                & (omegas < om_hi)
                & (yls > yl_lo)
                & (yls < yl_hi)
                & (zls > zl_lo)
                & (zls < zl_hi)
            )
            any_ok = any_ok | ok
        range_ok = any_ok
    else:
        range_ok = torch.ones_like(geom_valid)

    valid = geom_valid & pole_ok & range_ok

    return DiffrSpotsResult(
        omegas=omegas,
        etas=etas,
        yls=yls,
        zls=zls,
        valid=valid,
        orient_mats=orient_mats,
    )



def _auto_batch_orientations(n_or: int, n_hkl: int, device, dtype) -> int:
    """Orientations per chunk that keep the intermediates inside a memory budget.

    predict_spots materialises several (N, M, 2) tensors -- omegas, etas, yls,
    zls, valid and the Bragg-solver temporaries. At 486,755 orientations x 610
    reflections that is 4.75 GB EACH in float64, which is what made a DHCP phase
    fail outright on a 47 GB card:

        torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.21 GiB

    Chunking costs nothing (the results are concatenated) and removes the
    failure mode entirely, so the size of the phase no longer decides whether
    the stage runs.
    """
    per_or = max(1, n_hkl * 2)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    n_live = 10                      # concurrent (N, M, 2) tensors, incl. temps
    budget = 2 << 30                 # 2 GiB on CPU
    if torch.device(device).type == "cuda":
        try:
            free, _total = torch.cuda.mem_get_info(device)
            budget = int(free * 0.25)
        except Exception:            # noqa: BLE001
            budget = 4 << 30
    batch = max(1, budget // (per_or * itemsize * n_live))
    return min(n_or, batch)


def predict_spots(
    quaternions: torch.Tensor,
    hkls: torch.Tensor,
    thetas_deg: torch.Tensor,
    *,
    distance,
    omega_ranges=None,
    box_sizes=None,
    exclude_pole_angle: float = 0.0,
    wedge_deg: float = 0.0,
    batch_size: Optional[int] = None,
) -> "DiffrSpotsResult":
    """Predict spots for every orientation, batching to bound peak memory.

    Identical output to computing all orientations at once -- the orientation
    axis is independent, so chunking and concatenating is exact, not an
    approximation. ``batch_size=None`` sizes the chunk from free device memory;
    pass an integer to pin it.
    """
    n_or = int(quaternions.shape[0])
    if batch_size is None:
        batch_size = _auto_batch_orientations(
            n_or, int(hkls.shape[0]), quaternions.device, quaternions.dtype)
    if batch_size >= n_or:
        return _predict_spots_chunk(
            quaternions, hkls, thetas_deg, distance=distance,
            omega_ranges=omega_ranges, box_sizes=box_sizes,
            exclude_pole_angle=exclude_pole_angle, wedge_deg=wedge_deg)

    n_chunks = -(-n_or // batch_size)
    LOGGER.info("diffraction spots: %d orientations in %d chunk(s) of %d "
                "(bounding peak memory)", n_or, n_chunks, batch_size)
    parts = []
    for ci in range(n_chunks):
        lo, hi = ci * batch_size, min((ci + 1) * batch_size, n_or)
        parts.append(_predict_spots_chunk(
            quaternions[lo:hi], hkls, thetas_deg, distance=distance,
            omega_ranges=omega_ranges, box_sizes=box_sizes,
            exclude_pole_angle=exclude_pole_angle, wedge_deg=wedge_deg))
        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            LOGGER.info("  diffraction spots: chunk %d/%d", ci + 1, n_chunks)
    return DiffrSpotsResult(
        omegas=torch.cat([q.omegas for q in parts], dim=0),
        etas=torch.cat([q.etas for q in parts], dim=0),
        yls=torch.cat([q.yls for q in parts], dim=0),
        zls=torch.cat([q.zls for q in parts], dim=0),
        valid=torch.cat([q.valid for q in parts], dim=0),
        orient_mats=torch.cat([q.orient_mats for q in parts], dim=0),
    )


# -----------------------------------------------------------------------------
# High-level pipeline class
# -----------------------------------------------------------------------------


def resolve_hkls_csv(
    params: DiffrSpotsParams,
    *,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Locate ``hkls.csv`` the way the rest of the NF pipeline does.

    Every other NF stage writes and reads ``hkls.csv`` in the run/result
    folder (``OutputDirectory``, falling back to ``DataDirectory`` --
    ``midas_nf_pipeline.workflows`` sets ``resultFolder`` that way, and
    ``midas_nf_fitorientation.io.read_hkls`` reads ``<out_dir>/hkls.csv``).
    Search order, first existing file wins:

    1. ``output_dir`` (the ``--output-dir`` override), if given;
    2. ``params.output_directory`` (``OutputDirectory``);
    3. ``params.data_directory`` (``DataDirectory``) -- last resort only,
       because the raw-data folder often holds a stale ``hkls.csv``.

    Raises ``FileNotFoundError`` naming every path searched if none exists.
    """
    candidates: list[Path] = []
    for d in (output_dir, params.output_directory, params.data_directory):
        if d:
            c = Path(d) / "hkls.csv"
            if c not in candidates:
                candidates.append(c)
    for i, c in enumerate(candidates):
        if c.is_file():
            if i > 0:
                LOGGER.warning(
                    "diffr_spots: hkls.csv not found in %s; falling back to %s",
                    ", ".join(str(x.parent) for x in candidates[:i]), c,
                )
            return c
    raise FileNotFoundError(
        "hkls.csv not found; searched: "
        + ", ".join(str(c) for c in candidates)
        + ". Generate it in the output directory or pass --hkls-csv."
    )


class DiffrSpotsPipeline:
    """Orchestrator for the full ``MakeDiffrSpots`` workflow.

    Parameters
    ----------
    params : DiffrSpotsParams
    device, dtype : standard torch construction kwargs.

    Optional ``hkls_csv`` / ``seed_orientations_csv`` overrides bypass the
    paths in ``params``. Without ``hkls_csv`` the file is located by
    :func:`resolve_hkls_csv` (output dir first, ``DataDirectory`` last);
    ``output_dir`` is the same override later given to :meth:`run`.
    """

    def __init__(
        self,
        params: DiffrSpotsParams,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        *,
        hkls_csv: Optional[Union[str, Path]] = None,
        seed_orientations_csv: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.params = params
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)

        # Load HKLs (using rings_to_use filter from params, if any)
        self._hkls_csv = (
            Path(hkls_csv)
            if hkls_csv is not None
            else resolve_hkls_csv(params, output_dir=output_dir)
        )
        LOGGER.info("diffr_spots: reading hkls.csv from %s", self._hkls_csv)
        self._seeds_csv = (
            Path(seed_orientations_csv)
            if seed_orientations_csv is not None
            else Path(params.seed_orientations)
        )

        self.hkls, self.thetas_deg, self.ring_nrs = read_hkls_csv(
            self._hkls_csv, rings_to_use=params.rings_to_use or None
        )
        self.hkls = self.hkls.to(device=self.device, dtype=self.dtype)
        self.thetas_deg = self.thetas_deg.to(device=self.device, dtype=self.dtype)
        self.quaternions = read_seed_orientations(
            self._seeds_csv,
            nr_orientations=params.nr_orientations or None,
            dtype=self.dtype,
        ).to(device=self.device)

    @property
    def hkls_csv_path(self) -> Path:
        """The ``hkls.csv`` actually read."""
        return self._hkls_csv

    @property
    def n_hkls(self) -> int:
        return int(self.hkls.shape[0])

    @property
    def n_orientations(self) -> int:
        return int(self.quaternions.shape[0])

    def predict(self) -> DiffrSpotsResult:
        return predict_spots(
            self.quaternions,
            self.hkls,
            self.thetas_deg,
            distance=self.params.primary_distance,
            omega_ranges=self.params.omega_ranges,
            box_sizes=self.params.box_sizes,
            exclude_pole_angle=self.params.exclude_pole_angle,
            # The spot list must be predicted with the same rotation axis the
            # NF fit / screen use (they read the same Wedge); this was
            # silently 0 -- the parameter was never parsed here.
            wedge_deg=self.params.wedge,
        )

    def run_to_disk(
        self, *, output_dir: Optional[Union[str, Path]] = None, batch_size: Optional[int] = None
    ) -> tuple[int, dict]:
        """Like :meth:`run` (same three files; Key.bin and OrientMat.bin byte-identical, DiffractionSpots.bin equal to
        floating-point rounding, max |diff| 3e-14 um measured, because batching changes the vectorised kernels' rounding) but with device memory bounded by ``batch_size``
        orientations: each chunk is predicted, its valid spots are appended to ``DiffractionSpots.bin`` as float64 and
        freed. :meth:`run` keeps every (N, M, 2) tensor of the whole problem on the device and so cannot do a large
        seed set (972,439 orientations x 442 hkls, a trigonal cache for a long-c cell). Returns ``(total_spots, paths)``."""
        out_dir = Path(output_dir or self.params.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        n_or = self.n_orientations
        if batch_size is None:
            batch_size = _auto_batch_orientations(n_or, self.n_hkls, self.device, self.dtype)
        batch_size = max(1, min(int(batch_size), max(n_or, 1)))
        counts, mats, total = [], [], 0
        spots_path = out_dir / "DiffractionSpots.bin"
        with torch.no_grad(), open(spots_path, "wb") as fh:
            for lo in range(0, n_or, batch_size):
                hi = min(lo + batch_size, n_or)
                r = _predict_spots_chunk(
                    self.quaternions[lo:hi], self.hkls, self.thetas_deg, distance=self.params.primary_distance,
                    omega_ranges=self.params.omega_ranges, box_sizes=self.params.box_sizes,
                    exclude_pole_angle=self.params.exclude_pole_angle, wedge_deg=self.params.wedge)
                flat = r.flat_spots()
                flat.detach().cpu().numpy().astype(np.float64).tofile(fh)
                total += int(flat.shape[0])
                counts.append(r.counts.cpu()); mats.append(r.orient_mats.detach().cpu())
                del r, flat
        write_key_bin(torch.cat(counts) if counts else torch.zeros(0, dtype=torch.int64), out_dir / "Key.bin")
        write_orient_mat_bin(torch.cat(mats) if mats else torch.zeros((0, 3, 3), dtype=self.dtype), out_dir / "OrientMat.bin")
        return total, {"Key.bin": out_dir / "Key.bin", "DiffractionSpots.bin": spots_path, "OrientMat.bin": out_dir / "OrientMat.bin"}

    def run(
        self, *, output_dir: Optional[Union[str, Path]] = None
    ) -> tuple[DiffrSpotsResult, dict]:
        """Run the prediction and write the three binary files.

        Returns ``(result, written_paths)``.
        """
        result = self.predict()
        out_dir = Path(output_dir or self.params.output_directory)
        paths = write_all(
            out_dir, result.counts, result.flat_spots(), result.orient_mats
        )
        return result, paths
