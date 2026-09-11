"""The calling contract in manuals/defect/phase-1-ingest.md must EXECUTE.

Drift is the failure mode this guards. The prose in that doc set was correct
while its signatures were wrong, and six of them were hit in one afternoon by
the person who had just written the rest of it. A doc you cannot run is a doc
that rots silently, so this test extracts the fenced block from the manual and
runs it verbatim against a synthetic scene. Change a signature in
`midas_defect.ingest` and the MANUAL fails CI, which is the point.

2026-09-10: the contract now starts from a PONI and raw frames -- `live_frames`,
`poni_file_to_row_col`, `detector_angle_maps`, and omega mapped from the LIVE
stack back to raw frame numbers. The scene therefore carries a dead first frame,
a real PONI file, and a geometry whose rings clear `detect_powder_rings`' 2 deg
floor, and the test checks the omega mapping, not just that the block ran.
"""
from __future__ import annotations
import re
import sys
import types
from pathlib import Path
import numpy as np
import pytest

MANUAL = (Path(__file__).resolve().parents[3]
          / "manuals" / "defect" / "phase-1-ingest.md")


def _contract_block() -> str:
    if not MANUAL.exists():
        pytest.skip(f"manual not found at {MANUAL}")
    text = MANUAL.read_text()
    m = re.search(r"## The calling contract.*?```python\n(.*?)```", text, re.S)
    assert m, "the calling-contract code block is gone from phase-1-ingest.md"
    return m.group(1)


def _scene(nz=360, ny=340, nframes=8, seed=0):
    """Frames with a powder ring, some single-crystal spots, gaps, noise, and a dead frame 0."""
    rng = np.random.default_rng(seed)
    bcz, bcy = nz/2 + 3.0, ny/2 - 4.0
    zz, yy = np.mgrid[0:nz, 0:ny]
    r = np.hypot(zz - bcz, yy - bcy)
    frames = np.zeros((nframes, nz, ny), np.float32)
    for k in range(nframes):
        f = rng.normal(300.0, 12.0, (nz, ny))
        for rad in (58.0, 96.0):                    # continuous powder rings
            f += 5200.0*np.exp(-0.5*((r-rad)/1.15)**2)
        for (cz, cy, amp) in [(0.32, 0.72, 9e4), (0.74, 0.30, 7e4),
                              (0.24, 0.28, 6e4), (0.68, 0.76, 5e4)]:
            pz, py = cz*nz, cy*ny                   # spots, present on all frames
            f += amp*np.exp(-0.5*(((zz-pz)/1.9)**2 + ((yy-py)/1.9)**2))
        frames[k] = f
    frames[:, 120:126, :] = -1.0                    # module gap -> bad pixels
    frames[0] = np.abs(rng.normal(0.0, 2.0, (nz, ny)))   # shutter closed: a DEAD frame
    return frames, bcz, bcy


def _write_poni(path, *, row, col, px_m, lsd_m, lam_m, shape):
    path.write_text(
        "poni_version: 2.1\n"
        "Detector: Detector\n"
        f'Detector_config: {{"pixel1": {px_m}, "pixel2": {px_m}, '
        f'"max_shape": [{shape[0]}, {shape[1]}]}}\n'
        f"Distance: {lsd_m}\n"
        f"Poni1: {(row + 0.5)*px_m}\n"
        f"Poni2: {(col + 0.5)*px_m}\n"
        "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\n"
        f"Wavelength: {lam_m}\n")
    return path


def _integrate_v2_or_stub(monkeypatch):
    """The contract imports midas_integrate_v2, which midas_defect does not depend on.

    Use the real function when it is installed -- that is the boundary the manual
    is about. Otherwise register a minimal reader so the defect half still runs.
    """
    try:
        from midas_integrate_v2.compat.pyfai import poni_file_to_row_col  # noqa: F401
        return True
    except ImportError:
        pass

    def poni_file_to_row_col(path):
        kv = dict(line.split(":", 1) for line in Path(path).read_text().splitlines() if ":" in line)
        px = float(re.search(r'"pixel1":\s*([0-9.eE+-]+)', kv["Detector_config"]).group(1))
        return float(kv["Poni1"])/px - 0.5, float(kv["Poni2"])/px - 0.5

    pkg, compat, pyfai = (types.ModuleType(n) for n in
                          ("midas_integrate_v2", "midas_integrate_v2.compat",
                           "midas_integrate_v2.compat.pyfai"))
    pyfai.poni_file_to_row_col = poni_file_to_row_col
    for mod in (pkg, compat, pyfai):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return False


def test_manual_calling_contract_executes(tmp_path, monkeypatch):
    frames, bcz, bcy = _scene()
    nframe, nz, ny = frames.shape
    PX, LSD, LAM, OMEGA0, DOMEGA = 172.0, 150000.0, 0.42459, -3.5, 1.0
    poni = _write_poni(tmp_path/"scene.poni", row=bcz, col=bcy, px_m=PX*1e-6,
                       lsd_m=LSD*1e-6, lam_m=LAM*1e-10, shape=(nz, ny))
    real_integrate = _integrate_v2_or_stub(monkeypatch)

    class _TiffStub:
        @staticmethod
        def imread(idx):
            return frames[idx]

    monkeypatch.setitem(sys.modules, "tifffile", _TiffStub)   # the block does `import tifffile`
    ns = dict(path=lambda k: k, NFRAME=nframe, PONI=str(poni), LSD=LSD, PX=PX, LAM=LAM,
              ROW_BC=bcz, COL_BC=bcy, NROW=nz, NCOL=ny, OMEGA0=OMEGA0, DOMEGA=DOMEGA)
    exec(compile(_contract_block(), str(MANUAL), "exec"), ns)

    # the contract's own outputs must exist and be sane
    for nm in ("live", "fidx", "g", "tth", "az", "mask", "sub", "spots", "omega", "qlab", "q",
               "stth", "rad", "rings", "pw", "keep"):
        assert nm in ns, f"the contract block no longer defines `{nm}`"
    assert ns["mask"].shape == (nz, ny)
    assert ns["tth"].shape == (nz, ny) and ns["az"].shape == (nz, ny)
    assert len(ns["spots"]) > 0, "the contract found no spots at all"
    assert len(ns["keep"]) == len(ns["spots"])
    assert ns["q"].shape == (len(ns["spots"]), 3)

    # the dead frame must be dropped, and omega must be mapped from the LIVE stack
    # back to RAW frame numbers. With raw frame 0 dead, live index f is raw f + 1:
    # flooring the centroid, or using the live index as if it were raw, both fail here.
    assert list(ns["fidx"]) == list(range(1, nframe)), f"live frames {list(ns['fidx'])}"
    f_live = np.asarray(ns["spots"].frame.values, float)
    np.testing.assert_allclose(ns["omega"], OMEGA0 + DOMEGA*(f_live + 1.0), atol=1e-9,
                               err_msg="omega is not the live-stack centroid mapped through fidx")

    if real_integrate:   # the row/col boundary: the PONI point comes back as (row, col)
        assert abs(ns["row0"] - bcz) < 1e-6 and abs(ns["col0"] - bcy) < 1e-6, (
            f"poni_file_to_row_col gave ({ns['row0']}, {ns['col0']}) for a PONI written at "
            f"row {bcz}, col {bcy} -- the axes are swapped at the boundary")

    # THE TRAP THAT COST HALF THE REFLECTIONS: occupancy must be computed, which
    # only happens when detect_powder_rings is given azimuth_deg.
    occ = np.asarray(ns["rings"].occupancy, float)
    assert occ.size == 0 or np.isfinite(occ).all(), (
        "ring occupancy is NaN -- detect_powder_rings was called without "
        "azimuth_deg, which silently disables the continuity test and discards "
        "roughly half the real reflections as powder")

    # SEMANTIC assertions, not just "it ran". `mask`, `tth` and `az` are all the
    # same shape, so a swapped argument order raises NOTHING and quietly computes
    # garbage -- verified by sabotaging the manual. Only checking that the PLANTED
    # structure comes back catches that class of error.
    spots, keep = ns["spots"], ns["keep"]
    sz = np.asarray(spots.row.values, float)[keep]
    sy = np.asarray(spots.col.values, float)[keep]
    planted = [(0.32*nz, 0.72*ny), (0.74*nz, 0.30*ny),
               (0.24*nz, 0.28*ny), (0.68*nz, 0.76*ny)]
    found = sum(bool(np.any(np.hypot(sz-pz, sy-py) < 6.0)) for pz, py in planted)
    assert found >= 3, (
        f"only {found}/4 planted single-crystal spots survived as KEPT. The chain "
        f"ran but did not recover the scene -- check the argument order, which "
        f"raises nothing when mask/tth/az share a shape")
    r_keep = np.hypot(sz - bcz, sy - bcy)
    on_ring = np.any([np.abs(r_keep - rad) < 3.0 for rad in (58.0, 96.0)], axis=0)
    assert on_ring.sum() <= 1, (
        f"{int(on_ring.sum())} kept spots sit on a planted POWDER ring; the "
        f"powder rejection is not working on this scene")


def test_manual_contract_mentions_every_trap_it_documents():
    """The trap table must keep naming the traps; deleting a row is a regression."""
    text = MANUAL.read_text()
    for token in ("azimuth_deg", 'device="cpu"', "return_counts", ".mask",
                  "centre_deg", "use_diplib", "weight_by_radius",
                  "poni_file_to_row_col", "detector_angle_maps", "live_frames", "fidx"):
        assert token in text, f"phase-1-ingest.md no longer documents `{token}`"
