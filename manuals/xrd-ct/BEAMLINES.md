# XRD-CT — beamline specifics

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

Per-beamline operational facts: how to reach the host, what the files look like, and the
conventions that are **not** recoverable from a finished reconstruction.

Everything here was verified on the machine on the date shown. Where a previously-recorded
claim did not reproduce, that is stated rather than quietly dropped.

---

## APS 11-ID-C

**Verified 2026-08-18 on `11idc-controls.xray.aps.anl.gov`.**

### Reach

```bash
ssh 11idc          # ~/.ssh/config alias
```

**★ The hop authenticates as `11idcuser@chiltepin`, NOT the bare `chiltepin` alias** (which is
`s1iduser`). Using the wrong one fails in a way that looks like a key problem.

Host: RHEL 9, bash login shell, **20 cores, 30 GB RAM**. It shares the `/home/beams*`
filesystem.

### Environment — always by full path

```bash
KMP_DUPLICATE_LIB_OK=TRUE /home/beams12/S1IDUSER/opt/envs/midas/bin/python
```

The system `/bin/python3` has **no numpy, no tifffile, no midas**. Conda is not on the
non-interactive ssh PATH, so the full path is mandatory, not stylistic.

### Detector and frames

| | |
|---|---|
| Detector | 2880 × 2880, **150 µm** pixels |
| Format | qxrd TIFF, written from Windows |
| Frame flavour used | **"Subtracted Data"** — already background-subtracted, `float32`, **carries negative values** |
| Energy / λ | 106.9 keV, **λ = 0.11595 Å** |

**Pass the frames through unaltered.** Clipping the negatives is a silent edit to the data
being fitted and biases every integrated intensity upward.

### The `.tif.metadata` sidecar

Each frame has a sidecar. `file` reports it as
`Generic INItialization configuration [normalization]\015`.

**★ CRLF line endings — verified on all 5084 sidecars across two trees.** Lines end `\r\n`,
so:

```bash
grep '^width=2880$' frame.tif.metadata      # MATCHES NOTHING -- the line ends 2880\r
grep '^width=2880' frame.tif.metadata       # works
```

This is the trap that actually bites, and it fails **silently** — an empty grep result reads as
"the key is absent" rather than "your pattern is wrong".

**A previously-recorded claim that did NOT reproduce.** These sidecars contain a binary
`QDateTime` blob with a bare `%`, and it was recorded here that this makes `configparser` raise
`InterpolationSyntaxError`, requiring `ConfigParser(strict=False, interpolation=None)`.
**Tested 2026-08-18 on 5084 sidecars (all of `TomoData` sampled at 202, plus
`CeO2_107keV_Calib`): zero naive-read failures.** Every file has exactly one `%` line and
`ConfigParser().read()` handles all of them.

So the defensive form is **cheap insurance, not a required fix**:

```python
c = ConfigParser(strict=False, interpolation=None)   # harmless; use it if you like
```

Do not present it as a known bug. If you do hit an interpolation error on some other qxrd tree,
record the file — it would be new information.

### ★ Metadata `Distance` is the nominal stage setting

```
distance = 1600.0000        # in every sidecar
```

**The refined geometry is 1632.2 mm.** The beamline calibration file says 1579.5 mm. **All three
disagree and the two stored values are both wrong** for the data as collected — a 2 % and 3.3 %
error, i.e. 20 000–33 000 µε of apparent strain.

The calibrant filename itself carries the nominal value (`CeO2_FB_D1600-000000.tif`), so the
name is not evidence either.

**Refine the distance from the data. This is a halt condition** (spine, phase 1.1).

### Working directories

```
~/MIDAS/wd/TomoData/                 5056 frames + sidecars, 157 GB   (11idcuser)
~/MIDAS/wd/dt_survey/                survey, calibration, caches, scripts
~/MIDAS/wd/dt_survey/calib/          geometry_for_dt.json, residual_corr.bin, overlays
~/MIDAS/wd/dt_survey/cake_cache.h5              0.62 GB
~/MIDAS/wd/dt_survey/cake_cache_resid.h5        0.62 GB, residual correction applied
~/MIDAS/wd/CeO2/  LaB6/  CeO2_107keV_Calib/     earlier calibration trees
```

**`sentosa` sees `/home/beams/11IDCUSER/...` directly** — no copy needed to run heavier fits
there.

### ★ Never re-read the TIFFs

Reading the raw frames is **I/O bound at ~260 MB/s**, and 64 parallel workers gave
**8.0 frames/s against 6.7 serial** — parallelism buys almost nothing. The whole set is already
cached; fitting all 5054 projections off the cache takes **1.4 s** on 16 workers.

### ★ The cake axis order

`cake_cache_resid.h5` → `cake` is **`(translation, ω, η, R)` = `(14, 361, 36, 850)`**.

**η comes BEFORE R** — the opposite of the 1-ID integrated `.bin` layout. Verified by collapsing
each axis on one frame:

| axis | length | max/median | verdict |
|---|---|---|---|
| 0 | 36 | **1.03** (smooth) | **η** |
| 1 | 850 | **181** (sharp rings) | **R** |

This matches the `midas_dt` API contract — `ReducedFrame.intensity` is documented `(n_eta,
n_r)`.

<!-- Two things will mislead you here. build_cache.py has `n_r, n_eta = probe.shape`, which is
     backwards, and creates the dataset as (n_t, n_om, n_r, n_eta) -- so reading that script
     to learn the layout gives the wrong answer. The file also carries an attr
     `eta_deg_fixed: "was written with n_r/n_eta swapped; data unaffected"`, recording that the
     label arrays were once wrong and were repaired. The DATA was always (eta, R). -->

### The dataset

14 translations (`hxz` 0.000 → 1.300 mm, 100 µm step) × 361 ω = **5054 projections**.
Cake: R 350–1200 px at 1 px (850 bins), η 36 bins of 10°.

**ω sign is NOT determined, and probably cannot be from this data.** Both signs give identical
diameter and CV, differing only by a mirror (centroid column 6.22 vs 5.78). A near-symmetric rod
carries almost no handedness information.

**So the spine's "confirm the ω sign" halt does not apply as written here.** Record it as
**undetermined** and report the map as **mirror-ambiguous**. Do not pick a sign to clear the
gate — that converts an honest ambiguity into a false claim.

### Sample: it is a capillary

**1.1 mm is the capillary's OUTER diameter.** The ceria column inside a ~0.1 mm wall is ~0.9 mm,
which is what all three measurement routes give (0.930 / 0.930 / 0.859 mm). **The
reconstruction sees the powder, not the glass.**

Recorded as a discrepancy *before* the wall was known — and absorption was correctly ruled out
throughout, because absorption suppresses the centre and would **widen** the profile, i.e. the
wrong direction for a diameter that came out small.

**Still open:** the translation profile is **asymmetric** — `hxz1.300` carries 7× the signal of
`hxz0.000` although both sit equally far outside a column centred at 0.718 mm. A 38.5 µm
eccentricity is far too small to do that. Unexplained.

---

## APS 1-ID

Facts established on the DAC Ti set; see `LAB_NOTEBOOK.md` §7 for the full format decode.

| | |
|---|---|
| **ω NEGATED** | the aerotech stage convention — negate **every** ω. `conventions.aps_1id_omega` |
| **First frame is a throwaway** | 1-ID writes one every acquisition. Check whether `HeadSize` already skips it — double-skipping loses a real frame, not skipping shifts every ω by one step |
| Integrated `.bin` layout | **`[nR][nEta]`** — R before η, **opposite to 11-ID-C** |
| R↔2θ | `.REtaAreaMap.csv`, which carries the tilts and distortion (p0–p3) |
| Snake | **detected from the data**, never read from a flag |

Worked example (DAC Ti S1): `652 ω × 1207 η × 2400 R` float64, 25 files = 25 translations,
`startOme −169`, `omeStep 0.25`, `HeadSize 8396800`.

---

## Adding a beamline

Record, in this order — the first three are the ones that cannot be recovered later:

1. **ω sign**, and how it was established (or that it is undetermined, and why).
2. **First-frame handling.**
3. **Snake**, and whether it was detected or assumed.
4. Reach: ssh alias, and which account the hop authenticates as.
5. Environment path.
6. Detector geometry, λ, and whether the stored distance is trustworthy (**assume not**).
7. Frame format, dtype, and whether values may be negative.
8. Cake axis order, **verified by collapsing each axis** rather than read from a script.
9. Metadata quirks — and test them, rather than inheriting a claim.
