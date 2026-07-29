"""I/O readers and patch-cropping helpers for midas-grain-odf.

The package consumes three artifacts:

  - **geometry**: a small JSON file describing detector + scan geometry plus
    the reference crystal (lattice + space group rule for HKL enumeration).
  - **grains table**: a CSV with one row per grain (orientation + centroid +
    optional strained lattice).
  - **spot observations**: a CSV with one row per measured spot (grain_id,
    hkl_int, omega_branch, y, z, frame).

Frames are loaded as a 3-D ``(n_frames, n_y, n_z)`` array — accepted in
NPY, NPZ, or HDF5. Zarr support is one ``zarr.open_array`` call away when
the dependency is desired; we keep zarr optional.

Output ODF results are written to an HDF5 file with one group per grain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from midas_grain_odf.odf import axis_angle_to_matrix


# ---------------------------------------------------------------------------
#  Geometry / crystal config
# ---------------------------------------------------------------------------


@dataclass
class GrainOdfConfig:
    """Top-level config bundle loaded from a geometry JSON file."""

    # detector / scan
    Lsd: float
    y_BC: float
    z_BC: float
    px: float
    omega_start: float
    omega_step: float
    n_frames: int
    n_pixels_y: int
    n_pixels_z: int
    min_eta: float
    wavelength: float
    # crystal
    lattice: Tuple[float, float, float, float, float, float]
    crystal_system: str = "cubic_fcc"   # for HKL enumeration; extend as needed.
    # HKL enumeration cutoffs
    d_min_A: float = 0.6
    h_max: int = 4
    # torch.compile of the forward path. CUDA-only; ignored on CPU/MPS.
    # Pass True for "reduce-overhead" (default) or a string mode name.
    compile: "bool | str" = False

    @classmethod
    def from_json(cls, path: str | Path) -> "GrainOdfConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# ---------------------------------------------------------------------------
#  HKL enumeration
# ---------------------------------------------------------------------------


def enumerate_hkls(cfg: GrainOdfConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (hkls_cart, thetas, hkls_int) for the configured crystal.

    Currently supports ``crystal_system="cubic_fcc"``; the lattice is read
    from ``cfg.lattice[0]`` (a, in Angstroms). FCC selection rule applies.
    Extend with new branches as required.
    """
    if cfg.crystal_system == "cubic_fcc":
        a = cfg.lattice[0]
        return _enumerate_fcc(a, cfg.wavelength, cfg.d_min_A, cfg.h_max)
    raise NotImplementedError(
        f"HKL enumeration for crystal_system={cfg.crystal_system!r} "
        f"not yet implemented; subclass enumerate_hkls or extend this function."
    )


def _enumerate_fcc(a: float, wavelength_A: float, d_min_A: float, h_max: int):
    hkls = []
    ints = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                parity = (h % 2, k % 2, l % 2)
                if parity not in {(0, 0, 0), (1, 1, 1)}:
                    continue
                G = np.array([h, k, l], dtype=np.float64) / a
                Gn = float(np.linalg.norm(G))
                d = 1.0 / Gn
                if d < d_min_A:
                    continue
                arg = wavelength_A * Gn / 2.0
                if abs(arg) >= 1.0:
                    continue
                hkls.append(G)
                ints.append([h, k, l])
    G = torch.tensor(np.array(hkls), dtype=torch.float64)
    th = torch.tensor(
        [np.arcsin(wavelength_A * float(np.linalg.norm(g)) / 2.0)
         for g in hkls], dtype=torch.float64,
    )
    intt = torch.tensor(np.array(ints), dtype=torch.float64)
    return G, th, intt


# ---------------------------------------------------------------------------
#  Grain table
# ---------------------------------------------------------------------------


@dataclass
class GrainRow:
    grain_id: int
    R_avg: torch.Tensor          # (3, 3)
    position: torch.Tensor       # (3,) in micrometers
    lattice: torch.Tensor        # (6,) [a,b,c,alpha,beta,gamma], None if unstrained


def load_grains_csv(path: str | Path) -> List[GrainRow]:
    """Load a grains table from CSV.

    Required columns:
        grain_id
        R00 R01 R02 R10 R11 R12 R20 R21 R22   (orientation matrix entries)
        pos_x pos_y pos_z                      (in micrometers)

    Optional columns:
        a b c alpha beta gamma                  (strained lattice)
    """
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        R = torch.tensor([
            [row["R00"], row["R01"], row["R02"]],
            [row["R10"], row["R11"], row["R12"]],
            [row["R20"], row["R21"], row["R22"]],
        ], dtype=torch.float64)
        pos = torch.tensor(
            [row["pos_x"], row["pos_y"], row["pos_z"]], dtype=torch.float64,
        )
        if all(c in df.columns for c in ("a", "b", "c", "alpha", "beta", "gamma")):
            lattice = torch.tensor(
                [row["a"], row["b"], row["c"],
                 row["alpha"], row["beta"], row["gamma"]],
                dtype=torch.float64,
            )
        else:
            lattice = None
        out.append(GrainRow(
            grain_id=int(row["grain_id"]),
            R_avg=R, position=pos, lattice=lattice,
        ))
    return out


def write_grains_csv(grains: List[GrainRow], path: str | Path) -> None:
    rows = []
    for g in grains:
        d = {"grain_id": int(g.grain_id)}
        for i in range(3):
            for j in range(3):
                d[f"R{i}{j}"] = float(g.R_avg[i, j])
        d["pos_x"] = float(g.position[0])
        d["pos_y"] = float(g.position[1])
        d["pos_z"] = float(g.position[2])
        if g.lattice is not None:
            for k, name in enumerate(["a", "b", "c", "alpha", "beta", "gamma"]):
                d[name] = float(g.lattice[k])
        rows.append(d)
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
#  Spot observations
# ---------------------------------------------------------------------------


@dataclass
class SpotObservation:
    """One measured spot, indexed against the predicted (hkl, omega_branch)."""
    grain_id: int
    hkl_h: int
    hkl_k: int
    hkl_l: int
    omega_branch: int     # 0 = +ω solution, 1 = -ω solution
    y_pixel: float
    z_pixel: float
    frame: float


def load_spots_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["grain_id", "hkl_h", "hkl_k", "hkl_l", "omega_branch",
                "y_pixel", "z_pixel", "frame"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"spots CSV missing columns: {missing}")
    return df


def write_spots_csv(spots: List[SpotObservation], path: str | Path) -> None:
    rows = [s.__dict__ for s in spots]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
#  Frame loader
# ---------------------------------------------------------------------------


def load_frames(path: str | Path, dataset_key: str = "frames") -> np.ndarray:
    """Load a 3-D frame stack from .npy / .npz / .h5 / .hdf5.

    Always returns an in-memory ``np.ndarray`` of shape (n_frames, ny, nz).
    For very large stacks, callers should slice on disk before calling
    (see ``crop_patches_from_frames`` for the per-spot slab pattern).
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        with np.load(path) as z:
            return z[dataset_key]
    if path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            return f[dataset_key][...]
    raise ValueError(f"unsupported frames suffix: {path.suffix!r}")


# ---------------------------------------------------------------------------
#  Patch cropping
# ---------------------------------------------------------------------------


def crop_patches_from_frames(
    frames: np.ndarray,
    anchor_y: torch.Tensor,
    anchor_z: torch.Tensor,
    anchor_f: torch.Tensor,
    patch_F: int,
    patch_P: int,
) -> torch.Tensor:
    """Cut (F, P, P) patches around per-spot anchors.

    Anchors with non-integer values are rounded; sub-pixel residuals are
    absorbed into the per-spot Delta during inversion.

    Out-of-bounds regions are zero-padded (this matches the splatter's
    in_bounds masking).

    Returns
    -------
    patches : Tensor (S, F, P, P) float64
    """
    n_frames, n_y, n_z = frames.shape
    S = anchor_y.shape[0]
    half_F = patch_F // 2
    half_P = patch_P // 2

    cy = anchor_y.detach().round().long().cpu().numpy()
    cz = anchor_z.detach().round().long().cpu().numpy()
    cf = anchor_f.detach().round().long().cpu().numpy()

    out = np.zeros((S, patch_F, patch_P, patch_P), dtype=np.float64)
    for s in range(S):
        f0 = int(cf[s]) - half_F
        y0 = int(cy[s]) - half_P
        z0 = int(cz[s]) - half_P
        # Clip to frames bounds.
        f_src_lo, f_src_hi = max(0, f0), min(n_frames, f0 + patch_F)
        y_src_lo, y_src_hi = max(0, y0), min(n_y, y0 + patch_P)
        z_src_lo, z_src_hi = max(0, z0), min(n_z, z0 + patch_P)
        f_dst_lo, f_dst_hi = f_src_lo - f0, f_src_hi - f0
        y_dst_lo, y_dst_hi = y_src_lo - y0, y_src_hi - y0
        z_dst_lo, z_dst_hi = z_src_lo - z0, z_src_hi - z0
        if (f_src_hi <= f_src_lo) or (y_src_hi <= y_src_lo) or (z_src_hi <= z_src_lo):
            continue
        patch = frames[f_src_lo:f_src_hi, y_src_lo:y_src_hi, z_src_lo:z_src_hi]
        out[s, f_dst_lo:f_dst_hi, y_dst_lo:y_dst_hi, z_dst_lo:z_dst_hi] = patch
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
#  Spot indexer (matched-spot bookkeeping)
# ---------------------------------------------------------------------------


def build_spot_indexer(
    grain_rows: List[GrainRow],
    spots_df: pd.DataFrame,
    hkls_int: torch.Tensor,
) -> Dict[int, Dict[str, Any]]:
    """For each grain, build the indexer mapping observation -> (om_branch, hkl_idx).

    Returns a dict ``{grain_id: {"spot_indexer": LongTensor (S,),
                                  "meas_y", "meas_z", "meas_f"}}``.

    The flat spot index uses the convention from
    ``HEDMForwardModel.forward(...)`` after reshape (..., 2, M) -> (..., 2*M):
        flat = om_branch * M + hkl_idx
    """
    M = int(hkls_int.shape[0])
    hkls_int_np = hkls_int.detach().cpu().numpy().astype(int)

    # Build a (h,k,l) -> hkl_index lookup.
    hkl_lookup: Dict[Tuple[int, int, int], int] = {}
    for i, row in enumerate(hkls_int_np):
        hkl_lookup[(int(row[0]), int(row[1]), int(row[2]))] = i

    out: Dict[int, Dict[str, Any]] = {}
    for grain in grain_rows:
        sub = spots_df[spots_df["grain_id"] == grain.grain_id]
        if len(sub) == 0:
            continue

        flat_idx = []
        meas_y, meas_z, meas_f = [], [], []
        for _, srow in sub.iterrows():
            key = (int(srow["hkl_h"]), int(srow["hkl_k"]), int(srow["hkl_l"]))
            if key not in hkl_lookup:
                continue   # silently skip; calibrate by reporting count later
            hkl_idx = hkl_lookup[key]
            om_branch = int(srow["omega_branch"])
            flat_idx.append(om_branch * M + hkl_idx)
            meas_y.append(float(srow["y_pixel"]))
            meas_z.append(float(srow["z_pixel"]))
            meas_f.append(float(srow["frame"]))

        if not flat_idx:
            continue

        out[grain.grain_id] = {
            "spot_indexer": torch.tensor(flat_idx, dtype=torch.long),
            "meas_y": torch.tensor(meas_y, dtype=torch.float64),
            "meas_z": torch.tensor(meas_z, dtype=torch.float64),
            "meas_f": torch.tensor(meas_f, dtype=torch.float64),
        }
    return out


# ---------------------------------------------------------------------------
#  Result writer
# ---------------------------------------------------------------------------


def write_odf_results_h5(
    results: Dict[int, Dict[str, Any]],
    path: str | Path,
) -> None:
    """Write per-grain ODF inversion results to an HDF5 file.

    Layout:
        /grain_<id>/R_samples         (K, 3, 3)
        /grain_<id>/weights           (K,)
        /grain_<id>/delta_y           (S,)
        /grain_<id>/delta_z           (S,)
        /grain_<id>/delta_f           (S,)
        /grain_<id>/keep              (S,) bool
        /grain_<id>/loss_history      (n_steps,)
        /grain_<id>.attrs[odf_type]   str
        /grain_<id>.attrs[converged]  bool
    """
    import h5py
    with h5py.File(path, "w") as h5:
        for gid, payload in results.items():
            grp = h5.create_group(f"grain_{int(gid)}")
            for k, v in payload.items():
                if isinstance(v, torch.Tensor):
                    grp.create_dataset(k, data=v.detach().cpu().numpy())
                elif isinstance(v, (list, np.ndarray)):
                    grp.create_dataset(k, data=np.asarray(v))
                elif isinstance(v, (int, float, bool, str)):
                    grp.attrs[k] = v
