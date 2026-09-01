"""Shared test scaffolding for midas_defect.

Implements the fixtures described in implementation_plan.md Section 7.1:
  * _device_param  — parametrize cpu / cuda / mps with skipif-not-available
  * _dtype_param   — parametrize float32 / float64
  * demk_layer1_voxels — lazy loader for the canonical Demk-layer-1 sparse NPZ
  * synthetic_rod_cube         — deterministic 3-rod q-space synthetic
  * synthetic_asterism_patch   — deterministic single anisotropic Gaussian patch
  * synthetic_ht_pattern       — closed-form Hendricks–Teller sample at fixed α

All synthetic fixtures use fixed seeds so the tests are reproducible without
any on-disk reference data. Real-data fixtures are gated by the env var
MIDAS_DEFORM_REAL_DATA=1 and the file existing on the local mount.
"""

from __future__ import annotations

import os

# midas_transforms.apply_tilt_distortion (used by geometry.pixel_to_qlab) links a
# second OpenMP runtime alongside torch's; allow the duplicate on dev machines.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# fast Monte-Carlo Mackenzie reference (test speedup)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _fast_mackenzie_mc():
    """Reduce the cached MC sample count of ``mackenzie_pdf`` for the test session.

    ``mackenzie_pdf`` builds its histogram at ``n_mc_samples=200_000`` for production
    accuracy, and every distribution/regression test uses that default -- so the
    ~one-shot ~57 s-per-phase Monte-Carlo dominates the suite (FCC + BCC + HCP ~ 170 s).
    The tests only assert shape / normalisation / phase-equality, which a coarser MC
    satisfies. Patching the keyword default (one function object, shared by all imports)
    means every call site shares the cheaper cache, so each phase is computed once at
    the reduced count. Restored after the session.
    """
    from midas_defect.distributions import mackenzie as _mk
    kd = _mk.mackenzie_pdf.__kwdefaults__
    orig = kd.get("n_mc_samples")
    kd["n_mc_samples"] = 50_000
    try:
        yield
    finally:
        kd["n_mc_samples"] = orig


# ---------------------------------------------------------------------------
# device / dtype parametrization
# ---------------------------------------------------------------------------

_DEVICES: list[str] = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _DEVICES.append("mps")


@pytest.fixture(params=_DEVICES, ids=lambda d: f"device={d}")
def _device_param(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64],
                ids=["dtype=float32", "dtype=float64"])
def _dtype_param(request) -> torch.dtype:
    return request.param


# ---------------------------------------------------------------------------
# real-data fixture (Demk layer 1)
# ---------------------------------------------------------------------------

DEMK_REAL_DATA_ENV = "MIDAS_DEFORM_REAL_DATA"

# Canonical mount on copland; locally we expect the user to scp it into
# tests/fixtures/demk_layer1_voxels.npz before running real-data tests.
DEMK_REMOTE_PATH = (
    "/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/"
    "demk_qspace_scope/voxels_layer2346_bin4.npz"
)
DEMK_LOCAL_PATH = (
    Path(__file__).parent / "fixtures" / "demk_layer1_voxels.npz"
)

# Recognised analysis paths for the binned voxels + L1 grains.
# Order = first-match-wins; covers laptop, copland hsharma home, and gdata.
_DEMK_BIN4_CANDIDATES = [
    Path("/Users/hsharma/Desktop/analysis/demk/qscope/exp/layers/voxels_layer2346_bin4.npz"),
    Path("/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/demk_qspace_scope/voxels_layer2346_bin4.npz"),
    Path(os.path.expanduser("~/data/voxels_layer2346_bin4.npz")),
]
_DEMK_GRAINS_L1_CANDIDATES = [
    Path("/Users/hsharma/Desktop/analysis/demk/demk_ff_Cu_results/LayerNr_1/Grains.csv"),
    Path("/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/demk_ff_Cu/LayerNr_1/Grains.csv"),
]


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def demk_bin4_paths():
    """(voxels_npz, grains_csv) for the binned demk L1 — env-gated, no repo copy.

    Skipped unless MIDAS_DEFECT_REAL_DATA=1 (or the legacy MIDAS_DEFORM_REAL_DATA)
    and both files exist locally / on the gdata mount.
    """
    if (os.environ.get("MIDAS_DEFECT_REAL_DATA", "0") != "1"
            and os.environ.get(DEMK_REAL_DATA_ENV, "0") != "1"):
        pytest.skip("set MIDAS_DEFECT_REAL_DATA=1 to run real-data regression")
    bin4 = _first_existing(_DEMK_BIN4_CANDIDATES)
    grains = _first_existing(_DEMK_GRAINS_L1_CANDIDATES)
    if bin4 is None or grains is None:
        pytest.skip(
            f"demk bin4 voxels / grains not found at any of: "
            f"{_DEMK_BIN4_CANDIDATES} / {_DEMK_GRAINS_L1_CANDIDATES}"
        )
    return str(bin4), str(grains)


@pytest.fixture
def demk_layer1_voxels():
    """Lazy loader for the canonical Demk-layer-1 sparse-voxel NPZ.

    Skipped unless MIDAS_DEFORM_REAL_DATA=1 *and* the file exists at one of
    the recognized locations.
    """
    if os.environ.get(DEMK_REAL_DATA_ENV, "0") != "1":
        pytest.skip(f"set {DEMK_REAL_DATA_ENV}=1 to run real-data tests")
    for candidate in (DEMK_LOCAL_PATH, Path(DEMK_REMOTE_PATH)):
        if candidate.exists():
            return np.load(candidate, allow_pickle=False)
    pytest.skip(
        f"Demk layer-1 voxels not found at {DEMK_LOCAL_PATH} or "
        f"{DEMK_REMOTE_PATH}; scp the file from copland to enable."
    )


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_rod_cube() -> dict:
    """Deterministic 3-rod q-space voxel cube.

    Three rods at known directions + 50 random Bragg-like clusters embedded in
    a unit cube. Used by P1, P6, P7 tests.
    """
    rng = np.random.default_rng(0)
    rod_dirs = np.array([
        [1.0, 0.0, 0.0],          # along q_x
        [0.0, 1.0, 0.0],          # along q_y
        [1.0, 1.0, 1.0] / np.sqrt(3.0),  # along [111]
    ])
    pivots = np.zeros_like(rod_dirs)

    xs, ys, zs, vs = [], [], [], []
    # rod points — uniform along t, with a few "Bragg bumps" at known
    # shell crossings to mimic the real Demk pattern (rods include bright
    # cores wherever they cross a reciprocal lattice point)
    bragg_ts = (-0.3, -0.1, 0.1, 0.3)
    for d, p in zip(rod_dirs, pivots):
        ts = np.linspace(-0.4, 0.4, 200)
        pts = p[None, :] + ts[:, None] * d[None, :]
        pts += rng.normal(scale=0.005, size=pts.shape)
        # baseline intensities along the rod
        ints = rng.uniform(80, 200, size=ts.shape)
        # bump intensity at the "Bragg bump" positions (rods cross shells)
        for t_b in bragg_ts:
            close = np.abs(ts - t_b) < 0.03
            ints[close] = rng.uniform(300, 500, size=close.sum())
        xs.append(pts[:, 0]); ys.append(pts[:, 1]); zs.append(pts[:, 2])
        vs.append(ints)
    # random Bragg-like clusters (away from the rods)
    for _ in range(50):
        c = rng.uniform(-0.45, 0.45, size=3)
        n = rng.integers(20, 80)
        pts = c[None, :] + rng.normal(scale=0.01, size=(n, 3))
        xs.append(pts[:, 0]); ys.append(pts[:, 1]); zs.append(pts[:, 2])
        vs.append(rng.uniform(80, 500, size=n))

    return dict(
        qx=np.concatenate(xs).astype(np.float64),
        qy=np.concatenate(ys).astype(np.float64),
        qz=np.concatenate(zs).astype(np.float64),
        intensity=np.concatenate(vs).astype(np.float64),
        rod_dirs=rod_dirs,
        rod_pivots=pivots,
    )


@pytest.fixture
def synthetic_asterism_patch() -> dict:
    """Deterministic single anisotropic 3-D Gaussian asterism patch.

    Returned as a (Nz, Ny, Nω) intensity volume with known Σ, center, and amplitude.
    Used by P2, P3 tests.
    """
    rng = np.random.default_rng(0)
    Nz, Ny, Nw = 21, 21, 41
    center = np.array([10.0, 10.0, 20.0])
    # Anisotropic Σ: stretched along ω, slightly along z
    sigma_diag = np.diag([1.5 ** 2, 0.8 ** 2, 6.0 ** 2])
    R = _small_rotation(rng, max_deg=15.0)
    Sigma = R @ sigma_diag @ R.T
    Sigma_inv = np.linalg.inv(Sigma)
    amplitude = 1000.0

    zz, yy, ww = np.meshgrid(
        np.arange(Nz, dtype=np.float64),
        np.arange(Ny, dtype=np.float64),
        np.arange(Nw, dtype=np.float64),
        indexing="ij",
    )
    pts = np.stack([zz, yy, ww], axis=-1) - center
    quad = np.einsum("...i,ij,...j->...", pts, Sigma_inv, pts)
    img = amplitude * np.exp(-0.5 * quad)
    img += rng.normal(scale=2.0, size=img.shape)

    return dict(
        volume=img.astype(np.float64),
        center=center,
        sigma=Sigma,
        amplitude=amplitude,
    )


@pytest.fixture
def synthetic_ht_pattern() -> dict:
    """Closed-form Hendricks–Teller pattern at fixed α along a known rod direction.

    Returns a sparse cloud of (q, intensity) sampled densely along the rod
    direction and falling off with cos²(π α t) modulation, plus a few off-rod
    Bragg-like blobs. Used by P6, P7 tests.
    """
    rng = np.random.default_rng(0)
    alpha = 0.2
    rod_dir = np.array([0.0, 0.0, 1.0])      # along q_z
    rod_pivot = np.array([1.5, 0.0, 0.0])    # offset, so rod passes through (1.5, 0, 0..2)
    ts = np.linspace(0.0, 2.0, 400)
    pts = rod_pivot[None, :] + ts[:, None] * rod_dir[None, :]
    # Hendricks–Teller-like: I(q_∥) = (1-α²) / (1 - 2α cos(2π q_∥ d_layer) + α²) for
    # a unit-spacing layer; here we set d_layer = 1 for simplicity.
    d_layer = 1.0
    cos_term = np.cos(2.0 * np.pi * ts * d_layer)
    I_rod = (1.0 - alpha * alpha) / (1.0 - 2.0 * alpha * cos_term + alpha * alpha)
    I_rod *= 100.0
    pts += rng.normal(scale=0.003, size=pts.shape)

    # off-rod random clusters (noise)
    n_noise = 30
    noise_pts = rng.uniform(-0.5, 2.5, size=(n_noise, 3))
    noise_I = rng.uniform(5.0, 30.0, size=n_noise)

    qx = np.concatenate([pts[:, 0], noise_pts[:, 0]])
    qy = np.concatenate([pts[:, 1], noise_pts[:, 1]])
    qz = np.concatenate([pts[:, 2], noise_pts[:, 2]])
    intensity = np.concatenate([I_rod, noise_I])
    return dict(
        qx=qx.astype(np.float64),
        qy=qy.astype(np.float64),
        qz=qz.astype(np.float64),
        intensity=intensity.astype(np.float64),
        alpha=alpha,
        d_layer=d_layer,
        rod_dir=rod_dir,
        rod_pivot=rod_pivot,
    )


# ---------------------------------------------------------------------------
# synthetic Cu-Al-like end-to-end fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_cu_al_dataset() -> dict:
    """Module-scoped synthetic stand-in for a Cu-Al matrix-twin population.

    A deterministic two-variant population designed to exercise every
    Phase 0-3 module. Built once per test module and reused; downstream
    tests get a fresh dict view via ``request`` if they need to mutate.

    Returns
    -------
    dict with keys
        OM          (n_grains, 3, 3)  orientation matrices (sample frame)
        pos         (n_grains, 3)     positions in um
        radii       (n_grains,)       grain radii in um
        true_variant (n_grains,)      0 = matrix, 1 = twin
        eps_sample  (n_grains, 3, 3)  elastic strain in sample frame
        qs          (n_voxels, 3)     voxel q-positions in 1/A
        vals        (n_voxels,)       voxel intensities
        grain_of_voxel (n_voxels,)    grain index per voxel
        G_arr       (n_hkls, 3)       crystal-frame g-vectors (1/A)
        hkls        (n_hkls, 3)       Miller indices
        burgers     float             Burgers magnitude (m)
        lattice_a   float             lattice param (A) used to set |G|
        loading_axis (3,)             sample-frame loading direction
    """
    import midas_stress.orientation as o

    rng = np.random.default_rng(0)
    n_matrix = 30
    n_twin = 30
    n_grains = n_matrix + n_twin

    # Orientations -- matrix near identity with small jitter; twin = matrix * Sigma3
    sigma3 = np.asarray(
        o.axis_angle_to_orient_mat(np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0), 60.0)
    )
    jitter_deg = 3.0
    OM_matrix = np.stack(
        [
            np.asarray(
                o.axis_angle_to_orient_mat(
                    _unit(rng.normal(size=3)), rng.uniform(0.0, jitter_deg)
                )
            )
            for _ in range(n_matrix)
        ],
        axis=0,
    )
    OM_twin = np.stack(
        [
            OM_matrix[i % n_matrix]
            @ sigma3
            @ np.asarray(
                o.axis_angle_to_orient_mat(
                    _unit(rng.normal(size=3)), rng.uniform(0.0, jitter_deg)
                )
            )
            for i in range(n_twin)
        ],
        axis=0,
    )
    OM = np.concatenate([OM_matrix, OM_twin], axis=0)
    true_variant = np.concatenate(
        [np.zeros(n_matrix, dtype=int), np.ones(n_twin, dtype=int)]
    )

    # Positions: matrix-twin pairs share a base position so they end up as
    # spatial neighbours -- mimicking lamellar morphology.
    base_pos = rng.uniform(0.0, 200.0, size=(n_matrix, 3))  # um
    matrix_pos = base_pos
    twin_pos = base_pos + rng.normal(scale=2.0, size=(n_matrix, 3))
    pos = np.concatenate([matrix_pos, twin_pos], axis=0)

    # Grain radii: log-normal around 15 um
    radii = rng.lognormal(np.log(15.0), 0.3, size=n_grains)

    # Elastic strains: small deviatoric, slightly larger for matrix variant
    eps_sample = np.zeros((n_grains, 3, 3))
    for g in range(n_grains):
        scale = 1.2e-3 if true_variant[g] == 0 else 0.8e-3
        e = rng.normal(scale=scale, size=(3, 3))
        eps_sample[g] = 0.5 * (e + e.T)

    # Reciprocal lattice (Cu-like, a = 3.615 A)
    a = 3.615
    hkls = np.array(
        [
            [1, 1, 1],
            [2, 0, 0],
            [2, 2, 0],
            [3, 1, 1],
            [2, 2, 2],
            [4, 0, 0],
            [3, 3, 1],
            [4, 2, 0],
        ]
    )
    G_arr = hkls.astype(float) * (2.0 * np.pi / a)  # 1/A
    G_mag = np.linalg.norm(G_arr, axis=1)

    # Voxel cloud: per (grain, hkl), plant a small Gaussian cloud around the
    # predicted Bragg position. FWHM scales with |G| (strain-broadening
    # signature) and is larger for the twin variant -> higher rho_WH.
    qs_list = []
    vals_list = []
    g_of_v_list = []
    for g in range(n_grains):
        sigma_scale = 0.015 if true_variant[g] == 0 else 0.030
        for hi in range(len(hkls)):
            target = OM[g] @ G_arr[hi]
            sigma = sigma_scale * G_mag[hi] / G_mag[0]
            n_per = 80
            cloud = rng.normal(scale=sigma, size=(n_per, 3)) + target
            qs_list.append(cloud)
            vals_list.append(np.ones(n_per) * (1.0 + 0.5 * rng.standard_normal()))
            g_of_v_list.append(np.full(n_per, g, dtype=int))
    qs = np.concatenate(qs_list, axis=0)
    vals = np.concatenate(vals_list, axis=0)
    grain_of_voxel = np.concatenate(g_of_v_list, axis=0)

    return dict(
        OM=OM,
        pos=pos,
        radii=radii,
        true_variant=true_variant,
        eps_sample=eps_sample,
        qs=qs,
        vals=vals,
        grain_of_voxel=grain_of_voxel,
        G_arr=G_arr,
        hkls=hkls,
        burgers=2.57e-10,  # Cu
        lattice_a=a,
        loading_axis=np.array([0.0, 0.0, 1.0]),
    )


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Real-data fixture: demk FCC L1 (Cu-rich solid solution polycrystal)
# ---------------------------------------------------------------------------

_DEMK_FCC_ROOT_CANDIDATES = [
    Path("/Users/hsharma/Desktop/analysis/demk/fcc_reanalysis"),     # laptop
    Path(os.path.expanduser("~/data/demk_fcc_reanalysis_local")),     # copland hsharma home
    Path("/gdata/dm/MPE/OrthrosJr/analysis/sharma_work/demk_fcc_reanalysis_local"),
]


def _demk_fcc_root() -> Path | None:
    for p in _DEMK_FCC_ROOT_CANDIDATES:
        if p.exists():
            return p
    return None


DEMK_FCC_ROOT = _demk_fcc_root() or _DEMK_FCC_ROOT_CANDIDATES[0]


#: The exact legacy layout this fixture's ground-truth tables were computed
#: from. Grains.csv has been widened repeatedly (19 -> 21 -> 23 -> 47 -> 53)
#: and columns 13..18 changed MEANING at 47: on a 21-column file they are the
#: Voigt strain E11 E22 E33 E12 E13 E23, on 47/53 they are the lattice
#: parameters a b c alpha beta gamma. A `len(parts) < 21` guard passes a
#: 53-column file happily and would start reading a lattice as a strain --
#: silently, with no exception and plausible-looking magnitudes.
_DEMK_VOIGT_NAMES = ("E11", "E22", "E33", "E12", "E13", "E23")


def _read_grains_by_name(path):
    """Read a Grains.csv resolving every column BY NAME.

    Returns ``(OM (n,3,3), pos (n,3), eps_voigt (n,6) or None,
    radius (n,), confidence (n,), n_columns)``.

    Prefers the canonical reader in ``midas_process_grains.io`` -- the single
    place in the tree that knows every width and both ID spellings -- and
    falls back to an inline name-driven parse when that package is not
    installed (it is not a declared dependency of midas_defect).

    ``eps_voigt`` is None whenever the file does NOT name E11..E23. It is
    never filled from positions 13:19, because on a modern file those are the
    lattice parameters.
    """
    try:
        from midas_process_grains.io import read_grains_csv as _canon
    except ImportError:
        _canon = None

    if _canon is not None:
        t = _canon(path)
        return (t.orient_mat, t.positions, t.strain_voigt,
                t.grain_radius, t.confidence, t.n_columns)

    lines = Path(str(path)).read_text().splitlines()
    hidx = next((i for i, ln in enumerate(lines)
                 if ln.startswith("%") and "O11" in ln.lstrip("%").split()), None)
    if hidx is None:
        raise ValueError(f"{path}: no '%' header line naming O11..O33")
    cols = lines[hidx].lstrip("%").split()
    idx = {c: i for i, c in enumerate(cols)}
    rows = []
    for raw in lines[hidx + 1:]:
        # rstrip() first: the C writer terminates data rows with a tab.
        toks = raw.rstrip().split()
        if not toks or raw.lstrip().startswith("%") or len(toks) < len(cols):
            continue
        rows.append([float(v) for v in toks[:len(cols)]])
    arr = np.asarray(rows, dtype=float)

    def block(names):
        if not all(n in idx for n in names):
            return None
        return arr[:, [idx[n] for n in names]]

    om = block([f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)])
    if om is None or arr.size == 0:
        raise ValueError(
            f"{path}: header names {cols!r} but no complete O11..O33 data rows")
    return (om.reshape(-1, 3, 3),
            block(("X", "Y", "Z")),
            block(_DEMK_VOIGT_NAMES),
            arr[:, idx["GrainRadius"]] if "GrainRadius" in idx else None,
            arr[:, idx["Confidence"]] if "Confidence" in idx else None,
            len(cols))


@pytest.fixture(scope="module")
def demk_fcc_l1() -> dict:
    """Module-scoped fixture for the demk FCC L1 canonical re-analysis.

    Loads ``Grains_L1_local.csv`` (248 grains, FCC Cu, a=3.6356 A) and the
    canonical ground-truth result tables (``uq_table.csv``,
    ``mechanism_deep.csv``, ``comprehensive_polish.csv``,
    ``mecking_kocks_energy_balance.json``, ``cp_prediction.json``,
    ``strain_partition_final.csv``).

    Skipped unless the local mount exists -- raw voxels remain on copland, so
    voxel-level analyses are not tested here.
    """
    if not DEMK_FCC_ROOT.exists():
        pytest.skip(f"demk FCC L1 data not mounted at {DEMK_FCC_ROOT}")

    import json
    import csv

    grains_csv = DEMK_FCC_ROOT / "Grains_L1_local.csv"
    if not grains_csv.exists():
        pytest.skip(f"missing {grains_csv}")

    OM, pos, eps_voigt, radii, confs, n_cols = _read_grains_by_name(grains_csv)
    if eps_voigt is None:
        # A regenerated (47/53-column) file. Its strain is eFab/eKen in
        # MICROSTRAIN and in a different convention from the legacy Voigt
        # block, so the ground-truth tables below no longer apply -- and cols
        # 13..18 are now a b c alpha beta gamma, which the old
        # `len(parts) < 21` guard would have read as a strain without
        # complaint. Fail loudly instead of comparing apples to 1e6 oranges.
        pytest.fail(
            f"{grains_csv} has {n_cols} columns and no E11..E23 Voigt strain "
            f"block. This fixture's ground-truth tables were computed from "
            f"the 21-column legacy layout; a regenerated file carries "
            f"eFab*/eKen* in microstrain under a different convention. "
            f"Regenerate the ground truth (or read strain via "
            f"midas_process_grains.io.read_grains_csv(...).strain_ken) before "
            f"re-enabling this comparison."
        )
    if radii is None or confs is None:
        pytest.fail(f"{grains_csv}: missing GrainRadius/Confidence columns")
    radii_list = list(radii)
    conf_list = list(confs)
    # Expand to (n, 3, 3) symmetric strain tensors
    eps_tensor = np.zeros((OM.shape[0], 3, 3))
    eps_tensor[:, 0, 0] = eps_voigt[:, 0]
    eps_tensor[:, 1, 1] = eps_voigt[:, 1]
    eps_tensor[:, 2, 2] = eps_voigt[:, 2]
    eps_tensor[:, 0, 1] = eps_tensor[:, 1, 0] = eps_voigt[:, 3]
    eps_tensor[:, 0, 2] = eps_tensor[:, 2, 0] = eps_voigt[:, 4]
    eps_tensor[:, 1, 2] = eps_tensor[:, 2, 1] = eps_voigt[:, 5]

    # Ground-truth tables
    def _read_csv(name: str) -> dict[str, dict]:
        path = DEMK_FCC_ROOT / name
        if not path.exists():
            return {}
        out: dict[str, dict] = {}
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = (row["category"], row["name"])
                out[key] = {
                    "value": float(row["value"]) if row["value"] not in ("", "nan") else float("nan"),
                    "lower": float(row["lower"]) if row["lower"] not in ("", "nan") else float("nan"),
                    "upper": float(row["upper"]) if row["upper"] not in ("", "nan") else float("nan"),
                    "units": row["units"],
                    "n_boot": int(row["n_boot"]) if row["n_boot"] else 0,
                    "boot_unit": row["boot_unit"],
                }
        return out

    uq_table = _read_csv("uq_table.csv")
    mechanism_deep = _read_csv("mechanism_deep.csv")
    comprehensive_polish = _read_csv("comprehensive_polish.csv")
    strain_partition_final = _read_csv("strain_partition_final.csv")

    mk_path = DEMK_FCC_ROOT / "mecking_kocks_energy_balance.json"
    mk_energy = json.loads(mk_path.read_text()) if mk_path.exists() else {}
    cp_path = DEMK_FCC_ROOT / "cp_prediction.json"
    cp_pred = json.loads(cp_path.read_text()) if cp_path.exists() else {}

    return dict(
        OM=OM,
        pos=pos,
        eps_voigt=eps_voigt,
        eps_tensor=eps_tensor,
        radii=np.asarray(radii_list, dtype=float),
        confidence=np.asarray(conf_list, dtype=float),
        n_grains=OM.shape[0],
        lattice_a=3.6356,           # A
        burgers=2.571e-10,           # m   (Cu, a/sqrt(2) at a=3.6356 A)
        space_group=225,
        # Ground truth tables
        uq_table=uq_table,
        mechanism_deep=mechanism_deep,
        comprehensive_polish=comprehensive_polish,
        strain_partition_final=strain_partition_final,
        mk_energy=mk_energy,
        cp_pred=cp_pred,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    """Random rotation matrix of magnitude up to max_deg about a random axis."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(0.0, max_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
