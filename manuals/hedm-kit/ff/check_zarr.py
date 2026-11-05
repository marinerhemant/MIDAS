#!/usr/bin/env python
"""Post-zip_convert sanity checks (FF-HEDM manual §3d, §3e, §6d).

Checks the things that fail *silently* downstream: an all-zero dark, a frame
count that is one out, a one-step omega offset, and the RhoD-driven ring count.

The omega check is done against the RAW aero ENCODER, not against the parameter
file, because the parameter file is what we are trying to validate.

Reads omegaStart/omegaStep from exactly where midas_peakfit reads them
(measurement/process/scan_parameters/start and /step — NOT
analysis/process/analysis_parameters/OmegaStart), and reproduces peakfit's
own formula:  omega(used k) = start + step * (k + skipFrame)
(midas_peakfit/zarr_io.py:155-161,368 and orchestrator.py:360).

usage: check_zarr.py <result-dir> [layer-number=1] [omega-sign=-1]
  omega-sign: -1 when the stage is negated (20-ID-D aero), +1 when ω is used as logged.
(Generalised from an earlier single-experiment kit: layer and ω sign are arguments instead of LayerNr_1 and -1.)
"""
import sys, glob, os
import numpy as np

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  *** FAIL: {msg} ***")


res = sys.argv[1]
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OMSIGN = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
lay = os.path.join(res, f"LayerNr_{LAYER}")
zips = glob.glob(os.path.join(lay, "*.MIDAS.zip"))
if not zips:
    print(f"FAIL: no *.MIDAS.zip under {lay}")
    sys.exit(1)
z = zips[0]
print(f"zarr: {z}  ({os.path.getsize(z)/1e9:.2f} GB)")

import zarr
g = zarr.open(z, mode="r")
ap = g["analysis/process/analysis_parameters"]
sp = g["measurement/process/scan_parameters"]


def A(grp, k):
    try:
        return np.ravel(grp[k][:])
    except Exception:
        return None


print("\n--- geometry / config as stored in the zarr ---")
for k in ["Lsd", "YCen", "ZCen", "RhoD", "ImTransOpt", "Wavelength", "tx", "ty", "tz",
          "LatticeParameter", "SpaceGroup", "MinNrSpots", "SkipFrame", "preProcThresh",
          "RingThresh", "darkLoc", "NrPixels", "PixelSize"]:
    v = A(ap, k)
    if v is not None:
        s = np.array2string(v, precision=6, max_line_width=200, threshold=20)
        print(f"  {k:18} = {s}")

# ---------------- omega, checked against the encoder ----------------
print("\n--- omega (manual §3e: a one-step offset is invisible downstream) ---")
start = float(A(sp, "start")[0])
step = float(A(sp, "step")[0])
skip = int(A(ap, "SkipFrame")[0])
data = g["exchange/data"]
nraw = data.shape[0]
nused = nraw - skip
print(f"  scan_parameters/start = {start}   (what peakfit uses as omegaStart)")
print(f"  scan_parameters/step  = {step}")
print(f"  SkipFrame             = {skip}")
print(f"  exchange/data frames  = {nraw}  ->  {nused} used")
print(f"  omega span over used frames = {abs(step)*(nused-1):.4f} deg")

rawfolder = A(ap, "RawFolder")
stem = A(ap, "FileStem")
first = A(ap, "StartFileNrFirstLayer")
pad = A(ap, "Padding")
ext = A(ap, "Ext")


def _s(v):
    if v is None:
        return None
    x = v[0]
    return x.decode() if isinstance(x, bytes) else str(x)


rf, st, ex = _s(rawfolder), _s(stem), _s(ext)
try:
    fn = int(first[0]); pd = int(pad[0])
    raw = os.path.join(rf, f"{st}_{fn:0{pd}d}{ex}")
    import h5py
    with h5py.File(raw, "r") as h:
        aero = h["measurement/instrument/SMS/aero"][:]
    print(f"  encoder source: {os.path.basename(raw)}  ({aero.size} readings)")
    if aero.size != nraw:
        fail(f"encoder has {aero.size} readings but zarr has {nraw} frames")
    worst = 0.0
    for lbl, k in [("first used", 0), ("mid used", nused // 2), ("last used", nused - 1)]:
        got = start + step * (k + skip)
        true = OMSIGN * float(aero[k + skip])   # omega_MIDAS = OMSIGN * omega_logged
        d = got - true
        worst = max(worst, abs(d))
        print(f"    {lbl:11} k={k:5}  pipeline={got:11.4f}  encoder={true:11.4f}  "
              f"delta={d:+.4f} deg")
    jitter = 0.05
    if worst > abs(step) * 0.5:
        fail(f"omega is off by {worst:.4f} deg (>= half a step of {abs(step)}). "
             f"OmegaStart is wrong -- likely one step out.")
    elif worst > jitter:
        print(f"  ! omega max deviation {worst:.4f} deg exceeds nominal jitter {jitter} "
              f"but is under half a step -- check the encoder residual")
    else:
        print(f"  omega OK: max deviation {worst:.4f} deg, inside encoder jitter")
except Exception as e:
    print(f"  (encoder cross-check skipped: {type(e).__name__}: {e})")

# ---------------- dark ----------------
print("\n--- dark (manual §3d: an all-zero dark makes every threshold return 0 peaks) ---")
found = None
for grp in ["exchange/dark", "exchange/bright"]:
    try:
        a = g[grp]
        arr = a[: min(a.shape[0], 12)]
        m = float(np.mean(arr))
        print(f"  {grp:18} shape={str(a.shape):22} mean={m:10.3f} max={float(np.max(arr)):9.1f}")
        if m > 1.0 and found is None:
            found = grp
    except Exception:
        print(f"  {grp:18} absent")
dl = _s(A(ap, "darkLoc"))
print(f"  darkLoc in params = {dl}")
print(f"  non-zero dark group present: {found}")
if found is None:
    print("  NOTE: both dark groups read ~0 in the zarr. That is only a fault if the "
          "data frames still carry the pedestal -- checked below.")

# ---------------- data after preProcThresh ----------------
print("\n--- data frames after preProcThresh ---")
nz = []
for i in [skip, skip + nused // 2, nraw - 1]:
    fr = np.asarray(data[i], dtype=np.float64)
    f = float((fr > 0).mean())
    nz.append(f)
    print(f"  raw frame {i:5}: mean={fr.mean():9.3f} max={fr.max():9.1f} "
          f"nonzero_frac={f:.6f}")
if max(nz) == 0.0:
    fail("every sampled frame is entirely zero -- no peaks can be found")
elif max(nz) > 0.5:
    print("  ! frames are >50% non-zero: the pedestal may not have been removed")

# ---------------- hkls / RhoD ----------------
print("\n--- hkls.csv / RhoD (manual rule 15) ---")
hk = os.path.join(lay, "hkls.csv")
if not os.path.exists(hk):
    print("  hkls.csv absent (hkl stage has not run)")
else:
    rows = [r.split() for r in open(hk).read().splitlines() if r.strip()]
    hdr, body = rows[0], rows[1:]
    ri = hdr.index("RingNr") if "RingNr" in hdr else None
    if ri is None:
        print(f"  header: {hdr[:10]}")
    else:
        rings = sorted({int(r[ri]) for r in body})
        print(f"  rows={len(body)}  distinct rings={len(rings)}  max ring={max(rings)}")
        if max(rings) >= 500:
            fail(f"hkls.csv reaches ring {max(rings)} >= 500 -- overruns the indexer's "
                 f"fixed 500-ring array; RhoD is wrong (rule 15)")
        rhod = A(ap, "RhoD")
        npx = A(ap, "NrPixels")
        pxs = A(ap, "PixelSize")
        yc, zc = A(ap, "YCen"), A(ap, "ZCen")
        if all(v is not None for v in (rhod, npx, pxs, yc, zc)):
            N, px = float(npx[0]), float(pxs[0])
            cy, cz = float(yc[0]), float(zc[0])
            corner = max(np.hypot(cy - a, cz - b)
                         for a in (0.0, N - 1) for b in (0.0, N - 1))
            expect = corner * px
            got = float(rhod[0])
            print(f"  RhoD stored = {got:.1f} um;  corner_px*px = {expect:.1f} um;  "
                  f"ratio = {got/expect:.3f}")
            if abs(got / expect - 1.0) > 0.02:
                print(f"  ! RhoD is NOT corner_px*px (rule 15). Self-consistent only if the "
                      f"distortion coefficients were FIT at this RhoD.")

print("\n" + ("=" * 60))
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f_ in FAILURES:
        print(f"  - {f_}")
    sys.exit(2)
print("all zarr checks passed")
