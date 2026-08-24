# Tomography — runbook

**Last updated:** 2026-08-23

> Part of the **tomo doc set**. Spine: [`README.md`](README.md).
> This file is the volatile one — where things run *today*, what healthy looks
> like *under stated conditions*, and where to pick up. Everything here has a
> shelf life; the spine and the envelope do not.

---

## Where it runs

| what | where |
|---|---|
| `midas_tomo` (FBP + shift sweep) | any host with the shared env; CPU by default, `--gpu` if the CUDA engine was built |
| Shared env, **prod** | `/home/beams12/S1IDUSER/opt/envs/midas/bin/python` — non-editable |
| Shared env, **dev** | `/home/beams12/S1IDUSER/opt/envs/midas-dev/bin/python` — editable into `~s1iduser/opt/MIDAS_canonical` |
| Heavy work prefix | `KMP_DUPLICATE_LIB_OK=TRUE`, and `nice -n 19` while anyone else is on the box |

**The data are not all on one host.** As of 2026-08-23:

| dataset | where | note |
|---|---|---|
| bt_1id_jun25b NMC811 tomo (`.raw` + sidecar) | `chiltepin:/scratch/s1iduser/bt_1id_jun25b/tomo/` | 378 GB dir; the per-sample `.raw` are 118 MB each |
| bt_1id_jun25b PF results | `chiltepin:/scratch/s1iduser/bt_1id_jun25b/pilatus/` | s5pf1/L2 is a verified reference layer — **read-only** |
| bt_1id_jul26 Ce FF + NF + tomo | `copland:/gdata/dm/1ID/2026/bt_1id_jul26/data/` | **`/gdata` is NOT mounted on chiltepin** |

That split is the thing that bites: chiltepin cannot see `/gdata`, and the
bt_1id_jun25b `/scratch` is a local RAID no other host can see.

## Running a reconstruction

**Use the one command.** It reads the scan's own record, so nothing about the
frame layout or the geometry is hand-entered:

```bash
midas-tomo-reconstruct <expt>/metadata/<expt>/<scan>/<scan>_TomoFastScan.dat \
    --root <local dir holding the scan's image folder> \
    --out  <output dir> \
    [--crop ROW0 ROW1 COL0 COL1] [--measure-tilt] [--delta-beta N] \
    -nCPUs 8
```

It ingests, finds the rotation-axis shift coarse-then-fine, reconstructs at
it, writes NXtomoproc with provenance, and prints the `SampleShape` call.
**It stops on an uncertified shift**; `--no-strict` overrides and marks the
geometry unverified.

The older `midas-tomo -dataFN ...` still works and is the right tool when you
already have a staged binary and want one specific sweep. Its `--shifts` output
has one reconstruction per candidate and **the index you pick does not travel
in the file** — `--find-shift` will choose and report it.

Frame layouts are now **derived from the scan record and cross-checked**
against its own first/last image numbers, not counted by hand:

| dataset | layout | source |
|---|---|---|
| bt_1id_jun25b `nmc811s5tomo1` | 10 front white @7317, 3601 projections @7327, 10 back white @10928, 10 dark @10938 | `scanrecord`; matches the hand-counted script exactly |
| bt_1id_jul26 `tomo_Ce_ht525_s2` | 10 front white @74, **1801** projections @84, 10 back white @1885, 10 dark @1895 | same; the folder holds ~3 such scans |

*Corrected 2026-08-23:* an earlier note here said bt_1id_jul26 was "10 flats, 5463
projections, 10 flats, 10 darks". That was three scans counted as one.

## What healthy looks like, with conditions

| check | healthy | condition it depends on |
|---|---|---|
| transmission in clear air | 0.97–1.03 | flats and darks correctly assigned; restricted to the illuminated region |
| clear columns either side of the sample | > ~100 px at **every** angle | sample smaller than the FOV |
| μ·D across angles | stable to a few percent | a compact, non-truncated sample |
| shift sweep | one visually sharp shift, neighbours blurred | enough contrast to judge; weak on a phase-contrast scan |
| threshold sweep | `V_illum` stationary over a band | a real density step at the boundary |

**There is no single healthy number.** A transmission of 0.18 is healthy for Ce
at 95 keV (μ·D 1.6) and would mean something is badly wrong for NMC811 at 52 keV,
where 0.95 is normal.

## Current pick-up point

State as of 2026-08-23. The workflow (T1–T5) is **built and gated**; the
science it was built for is not finished.

**Done and verified**

* `scanrecord` / `ingest` — the regenerated `data_nmc811s5tomo1.raw` is
  **byte-identical** to the beamline's hand-made file (sha256 `d933c7167a271406`).
* Automatic centring reproduces the human's **+13.00** exactly (0.000 px) on
  bt_1id_jun25b, though the two-criterion consensus declines to certify it.
* `detector_tilt` — Ce roll **−0.006 ± 0.030°**, consistent with zero.
* Paganin retrieval, exact round trip and a bit-exact null at `delta_beta=0`.
* `SampleShape`, the four readers, the corrected grain-size estimator, and the
  V1/V2 registration checks with the mirror meta-null.

**Resolved since the last revision of this file**

* ~~Ce pixel size~~ — **0.69 µm**, from the per-scan `tomo_metastr`. The 1.17 µm
  in `tomocupy_args.yml` is the PointGrey value and is wrong for both beamtimes.
  The 1.9× inconsistency closes to 1.09×.
* bt_1id_jun25b is **0.708 µm**, not 1.17 either, so its FOV is 90.6 µm and the
  specimen ≈29 µm — the "≈48 µm in a 149.8 µm FOV" above was computed at the
  wrong pixel size.

**Open**

1. **No dataset has both FF grains and a tomogram.** Ce has tomo + raw FF, but
   the Ce FF has never been reconstructed; the only `Grains.csv` in that
   beamtime is Au3 cubes, which has no tomo. Validating the grain-size
   correction requires reconstructing the Ce FF.
2. **The bt_1id_jun25b tomogram cannot produce a sample mask** — no threshold
   plateau, mask filling the field of view, and Paganin did not rescue it.
   Consistent with μ·D 0.05 at D≈100 mm.
3. **`sharpness(method='tv')` looks biased for centring** — it maximises
   `−mean|∇f|`, preferring the least-gradient image, and lands 1.6 px low on
   bt_1id_jun25b. Pre-existing; the fix is a design choice.
4. **The threshold-stationarity diagnostic needs a scale-invariant form.**
   Percentile thresholds pin `radius_spread` at 4.642; a fixed absolute range
   is unfair to anything that changes the value scale.
5. **`.par` column mapping not established** for the stage vertical that
   registers tomo against FF/NF. `bt_1id_jul26_tomopar.par` is polluted with
   `EPICS exception` lines that must be filtered before parsing.

**Known:** NF for the Ce sample was very hard to reconstruct. That weakens the
NF grain-size cross-check but not the primary absorption gate, which is the
Friedel-pair regression — FF only, zero free parameters.
