"""PF seed-outer parity — the invariant the seed-outer restructure promises.

The PF spot-driven path reuses one seed's forward model across every voxel that
seed's beam gate serves (orientation outer, voxel inner) instead of recomputing
it per voxel. That is a pure bookkeeping change: it must not move a single byte
of ``Output/*.bin``. These tests assert exactly that, and they are the only
tests in this package that exercise the restructure.

Why they are shaped this way:

* ``MIDAS_PF_NO_SEED_OUTER=1`` forces the legacy voxel-outer loop, so ONE binary
  produces both arms. No golden file is needed and no cross-platform floating
  point enters — both arms run on the same machine with the same libm.
* The bin index (``Data.bin`` / ``nData.bin``) is gitignored — ~1 GB on this
  fixture's bin geometry — so every pre-existing C-parity test in this package
  SKIPS on a fresh clone. That is how the seed-outer change nearly shipped with
  no coverage at all. Here the bins are regenerated from the tracked
  ``Spots.bin``; the regeneration is bit-exact (verified: the rebuilt
  ``Data.bin``/``nData.bin`` md5s match the originals produced by the C
  ``SaveBinDataScanning`` step).
* Thread-count invariance is tested separately because it is the one thing the
  restructure could plausibly break. Parallelising over seeds means several
  threads emit solutions for one voxel, where the legacy loop had one thread own
  a voxel and append in ascending seed order. The implementation buffers per
  thread and re-sorts on ``(voxel, seed row)``; if that sort were wrong, a
  single-thread run would still look fine.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from midas_index import backend_c

FIXTURE = Path(__file__).parent / "data" / "scanning_5grain_golden"
OUT_FILES = (
    "IndexBest_all.bin",
    "IndexKey_all.bin",
    "IndexBest_IDs_all.bin",
    "IndexBest_weights_all.bin",
)

pytestmark = pytest.mark.skipif(
    not backend_c.available(),
    reason=f"C indexer not built at {backend_c.binary_path()}",
)


def _regen_bins(dst: Path) -> None:
    """Rebuild Data.bin / nData.bin from the tracked Spots.bin.

    Mirrors the tail of ``midas_transforms.bin_data.voxel_binner.bin_data_scanning``
    (the block after the spot sort): per-ring bin assignment, then the PF
    (spotRow, scannrobs) pair layout. Per-ring processing is bit-identical to
    the all-at-once path because Data.bin is ring-major.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("midas_transforms")
    from midas_transforms.bin_data.core import _bin_assignment, _build_ring_radii
    from midas_transforms.bin_data.voxel_binner import _bin_to_data_ndata_scanning
    from midas_transforms.io import binary as bio
    from midas_transforms.params import read_paramstest

    p = read_paramstest(dst / "paramstest.txt")
    spots = np.fromfile(dst / "Spots.bin", dtype=np.float64).reshape(-1, 10)
    dev, dt = torch.device("cpu"), torch.float64
    spots_t = torch.tensor(spots[:, :8], device=dev, dtype=dt)
    scan_t = torch.tensor(spots[:, 9], device=dev).long()

    radii = _build_ring_radii(p).to(device=dev, dtype=dt)
    n_ring = p.highest_ring_no
    n_eta = math.ceil(360.0 / p.EtaBinSize)
    n_ome = math.ceil(360.0 / p.OmeBinSize)
    counts = torch.zeros(n_ring * n_eta * n_ome, dtype=torch.int64, device=dev)
    parts = []
    for r in [i for i in range(radii.shape[0]) if float(radii[i].item()) > 0]:
        one = torch.zeros_like(radii)
        one[r] = radii[r]
        arrays = list(_bin_assignment(
            spots_t, one, margin_ome=p.MarginOme, margin_eta=p.MarginEta,
            eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize,
            step_size_orient=p.StepSizeOrient))
        arrays.append(scan_t[arrays[0]])
        d_r, nd_r = _bin_to_data_ndata_scanning(
            arrays, n_ring_bins=n_ring, n_eta_bins=n_eta, n_ome_bins=n_ome)
        counts += nd_r[:, 0]
        if d_r.shape[0]:
            parts.append(d_r)
    data = torch.cat(parts) if parts else torch.zeros((0, 2), dtype=torch.int64)
    offs = torch.zeros_like(counts)
    offs[1:] = torch.cumsum(counts[:-1], dim=0)
    bio.write_data_ndata_bin_scanning(
        dst / "Data.bin", dst / "nData.bin",
        data.numpy().astype(np.uint64),
        torch.stack([counts, offs], dim=1).numpy().astype(np.uint64))


@pytest.fixture(scope="session")
def workdir(tmp_path_factory) -> Path:
    """A self-contained copy of the fixture, with the bin index present."""
    if not (FIXTURE / "Spots.bin").exists():
        pytest.skip(f"fixture inputs missing at {FIXTURE}")
    d = tmp_path_factory.mktemp("pf_seed_outer")
    for f in ("Spots.bin", "hkls.csv", "paramstest.txt", "positions.csv"):
        shutil.copy(FIXTURE / f, d / f)
    for f in ("Data.bin", "nData.bin"):
        if (FIXTURE / f).exists():
            shutil.copy(FIXTURE / f, d / f)
    if not (d / "Data.bin").exists():
        _regen_bins(d)
    return d


SEED_OUTER_MARKER = "PF seed-outer:"


def _run(workdir: Path, tag: str, *, seed_outer: bool, procs: int,
         block_nr: int = 0, n_blocks: int = 1) -> tuple[dict[str, bytes], str]:
    """Run the indexer into its own output dir; return the four blobs + stdout.

    stdout is returned because it is the only way to tell that the two arms
    actually took different code paths. On a binary that predates the
    restructure, ``MIDAS_PF_NO_SEED_OUTER`` is simply an unread environment
    variable and BOTH arms run the legacy loop — every parity assertion below
    would then pass while testing nothing.
    """
    run = workdir / tag
    (run / "Output").mkdir(parents=True, exist_ok=True)
    for f in ("Data.bin", "nData.bin", "Spots.bin", "hkls.csv", "positions.csv"):
        tgt = run / f
        if not tgt.exists():
            tgt.symlink_to(workdir / f)
    lines = [ln for ln in (workdir / "paramstest.txt").read_text().splitlines()
             if not ln.startswith(("OutputFolder", "ResultFolder"))]
    lines += [f"OutputFolder {run}/Output", f"ResultFolder {run}/Results"]
    pt = run / "paramstest.txt"
    pt.write_text("\n".join(lines) + "\n")

    n_scans = sum(1 for ln in (workdir / "positions.csv").read_text().splitlines()
                  if ln.strip())
    env = {} if seed_outer else {"MIDAS_PF_NO_SEED_OUTER": "1"}
    proc = backend_c.run_indexer(pt, block_nr=block_nr, n_blocks=n_blocks,
                                 n_work=n_scans, num_procs=procs,
                                 extra_env=env, cwd=run)
    stdout = (proc.stdout or b"").decode("utf-8", "replace")
    out = {}
    for f in OUT_FILES:
        path = run / "Output" / f
        assert path.exists(), f"{tag}: {f} was not written"
        out[f] = path.read_bytes()
    return out, stdout


@pytest.fixture(scope="session")
def _raw_arms(workdir):
    return {
        "legacy": _run(workdir, "legacy", seed_outer=False, procs=4),
        "seed_outer": _run(workdir, "seed_outer", seed_outer=True, procs=4),
        "seed_outer_1t": _run(workdir, "seed_outer_1t", seed_outer=True, procs=1),
        "legacy_shard0": _run(workdir, "legacy_shard0", seed_outer=False,
                              procs=4, block_nr=0, n_blocks=2),
        "seed_outer_shard0": _run(workdir, "seed_outer_shard0", seed_outer=True,
                                  procs=4, block_nr=0, n_blocks=2),
    }


@pytest.fixture(scope="session")
def arms(_raw_arms) -> dict[str, dict[str, bytes]]:
    """Every configuration the tests compare, run once each.

    Skips the whole module if the binary has no seed-outer path — otherwise
    every assertion here would pass against an unpatched build that ignores
    the environment variable and runs the legacy loop twice.
    """
    if SEED_OUTER_MARKER not in _raw_arms["seed_outer"][1]:
        pytest.skip("this midas_indexer build has no seed-outer path "
                    "(MIDAS_PF_NO_SEED_OUTER is unread) — nothing to compare")
    return {k: v[0] for k, v in _raw_arms.items()}


def test_the_two_arms_really_took_different_paths(_raw_arms):
    """The guard the rest of the module depends on.

    Without this, an unpatched binary makes every parity test below green while
    exercising one code path twice.
    """
    if SEED_OUTER_MARKER not in _raw_arms["seed_outer"][1]:
        pytest.skip("build predates the seed-outer path")
    assert SEED_OUTER_MARKER in _raw_arms["seed_outer"][1], (
        "seed-outer arm did not report taking the seed-outer path")
    assert SEED_OUTER_MARKER not in _raw_arms["legacy"][1], (
        "MIDAS_PF_NO_SEED_OUTER=1 did not force the legacy path")


def test_output_is_not_trivially_empty(arms):
    """Guard: two empty results compare equal.

    A fixture that yields no solutions makes every parity assertion below pass
    vacuously. This has actually happened during development, so it is asserted
    rather than assumed. The consolidated header is 4 + 12*nVox bytes; anything
    larger means real solutions were written.
    """
    blob = arms["seed_outer"]["IndexBest_all.bin"]
    n_vox = int(np.frombuffer(blob[:4], dtype=np.int32)[0])
    n_sol = np.frombuffer(blob[4:4 + 4 * n_vox], dtype=np.int32)
    assert n_vox > 0
    assert int(n_sol.sum()) > 0, "fixture produced no solutions — tests would be vacuous"
    assert len(blob) > 4 + 12 * n_vox


@pytest.mark.parametrize("name", OUT_FILES)
def test_seed_outer_matches_legacy(arms, name):
    """The restructure must not change a byte of any output file."""
    assert arms["seed_outer"][name] == arms["legacy"][name], (
        f"{name} differs between the seed-outer and legacy paths")


@pytest.mark.parametrize("name", OUT_FILES)
def test_seed_outer_is_thread_count_invariant(arms, name):
    """Solutions must be committed in (voxel, seed) order regardless of threads.

    Seed-outer parallelism has several threads producing solutions for the same
    voxel; the per-thread buffers are merged by a stable sort. A broken sort
    shows up here and nowhere else.
    """
    assert arms["seed_outer"][name] == arms["seed_outer_1t"][name], (
        f"{name} depends on thread count — the (voxel, seed) merge is wrong")


@pytest.mark.parametrize("name", OUT_FILES)
def test_seed_outer_matches_legacy_when_sharded(arms, name):
    """Sharding clips the voxel range, and the served-set enumeration must clip
    with it. This is the one place the seed-outer gate could disagree with the
    legacy per-voxel gate."""
    assert arms["seed_outer_shard0"][name] == arms["legacy_shard0"][name], (
        f"{name} differs under sharding — served-set clipping is wrong")


def test_matches_golden_structurally(arms):
    """Compare against the stored golden, tolerantly and for a stated reason.

    The golden was generated on macOS (its paramstest.txt still points at a
    /Users/... path). Apple's libm and glibc differ in the last ULP of ``acos``,
    which propagates into the orientation matrices — measured at ~3e-15, with
    the internal angle at ~3e-8 absolute. Every DISCRETE decision is unaffected:
    solution counts, offsets, SpotID, position and match counts are bit-equal.

    So this asserts exact equality on everything discrete and a tolerance on the
    transcendental-derived columns. Asserting byte equality here would fail on
    Linux for a reason that has nothing to do with the indexer.
    """
    gold_path = FIXTURE / "golden" / "IndexBest_all.bin"
    if not gold_path.exists():
        pytest.skip("golden IndexBest_all.bin not present")
    gold, ours = gold_path.read_bytes(), arms["seed_outer"]["IndexBest_all.bin"]

    def unpack(b):
        n = int(np.frombuffer(b[:4], dtype=np.int32)[0])
        n_sol = np.frombuffer(b[4:4 + 4 * n], dtype=np.int32)
        offs = np.frombuffer(b[4 + 4 * n:4 + 12 * n], dtype=np.int64)
        vals = np.frombuffer(b[4 + 12 * n:], dtype=np.float64).reshape(-1, 16)
        return n, n_sol, offs, vals

    ng, sg, og, vg = unpack(gold)
    no, so, oo, vo = unpack(ours)
    assert ng == no
    np.testing.assert_array_equal(sg, so, "per-voxel solution counts differ")
    np.testing.assert_array_equal(og, oo, "record offsets differ")
    assert vg.shape == vo.shape

    # Discrete columns: SpotID(0), x,y,z(11-13), nTspots(14), nMatches(15).
    for c in (0, 11, 12, 13, 14, 15):
        np.testing.assert_array_equal(vg[:, c], vo[:, c],
                                      f"discrete column {c} differs")
    # Orientation matrix (2-10): transcendental-derived, last-ULP platform drift.
    np.testing.assert_allclose(vg[:, 2:11], vo[:, 2:11], rtol=0, atol=1e-12)
    # Internal angle (1): a ratio of the above, so a looser absolute bound.
    np.testing.assert_allclose(vg[:, 1], vo[:, 1], rtol=0, atol=1e-6)
