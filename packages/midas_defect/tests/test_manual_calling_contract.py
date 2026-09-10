"""The calling contract in manuals/defect/phase-1-ingest.md must EXECUTE.

Drift is the failure mode this guards. The prose in that doc set was correct
while its signatures were wrong, and six of them were hit in one afternoon by
the person who had just written the rest of it. A doc you cannot run is a doc
that rots silently, so this test extracts the fenced block from the manual and
runs it verbatim against a synthetic scene. Change a signature in
`midas_defect.ingest` and the MANUAL fails CI, which is the point.
"""
from __future__ import annotations
import re
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
    """Frames with a powder ring, some single-crystal spots, gaps and noise."""
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
    tth = 0.05*r                                    # monotonic in radius
    az = np.degrees(np.arctan2(zz - bcz, yy - bcy))
    return frames, tth, az, bcz, bcy


def test_manual_calling_contract_executes():
    from midas_defect.geometry import Geometry
    frames, tth, az, bcz, bcy = _scene()
    nz, ny = frames.shape[1:]
    g = Geometry(lsd_um=349682.0, bcy_px=bcy, bcz_px=bcz, px_um=172.0,
                 wavelength_A=0.42459, n_pix_y=ny, n_pix_z=nz,
                 omega_first_deg=-3.5, omega_step_deg=1.0,
                 n_frames=frames.shape[0], ty_deg=0.0, tz_deg=0.0)

    class _TiffStub:
        @staticmethod
        def imread(idx):
            return frames[idx]

    ns = dict(np=np, tifffile=_TiffStub, g=g, tth=tth, az=az,
              LAM=0.42459, BCR=bcz, BCC=bcy, NFRAME=frames.shape[0],
              path=lambda p, k: k, p=0)
    exec(compile(_contract_block(), str(MANUAL), "exec"), ns)

    # the contract's own outputs must exist and be sane
    for nm in ("mask", "sub", "spots", "qlab", "stth", "rad", "rings", "pw", "keep"):
        assert nm in ns, f"the contract block no longer defines `{nm}`"
    assert ns["mask"].shape == (nz, ny)
    assert len(ns["spots"]) > 0, "the contract found no spots at all"
    assert len(ns["keep"]) == len(ns["spots"])
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
                  "centre_deg", "use_diplib", "weight_by_radius"):
        assert token in text, f"phase-1-ingest.md no longer documents `{token}`"
