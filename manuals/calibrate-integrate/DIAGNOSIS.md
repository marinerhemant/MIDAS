# Diagnosis — symptom → discriminating test → cause → lever

Indexed by **symptom**, because the step that produced a symptom is rarely the
step you are on. Every entry carries a test that can come back the other way; an
entry whose test cannot exonerate the cause it names does not belong here.

Numbers are from the reference dataset (48-panel Pilatus, 20-ID, CeO2). On other
detectors the *test* transfers; the *value* may not.

---

## Spiky or wildly wrong intensity near η = 0, ±90, ±180

**Test.** Read `SubPixelLevel` from the parameter file. Then integrate the same
frame twice against the *same* map, once with the truncating lookup and once with
round-to-parent (`rint(raw + sp_c) == raw`).

**If rounding removes almost all of the difference** → the CUDA integrator is
truncating the fractional sub-pixel coordinate and reading the neighbouring
pixel. Measured: 24.32 max relative in-band, 1218 bins beyond 2×, and rounding
removed 99.998 % of it.

**Lever.** `SubPixelLevel 1`. `0` is bit-identical. Do not raise it; the level is
not a quality knob.

**If rounding changes nothing** → not this. Look at the mask, or at ring
sharpness against the radial bin width.

---

## Every ring sits uniformly inside or outside where it should

**Test.** Contour the per-pixel R map at the ideal radii over the raw frame
(spine §6). Then compare against a from-scratch `make_seed`, which uses only the
image.

**If the seeder disagrees with the parameter file by more than a pixel** → the
active geometry block is wrong for these frames. On the reference dataset the
active `Lsd` was 3.0 mm short, ring radii landed 2.60 px inside the data, and the
block's own `FileStem` pointed at a different file in another user's directory.

**Lever.** Recalibrate from scratch (spine §4). Do not nudge the existing block —
if it belongs to another dataset, every coefficient in it is suspect, not just
`Lsd`.

**If the seeder agrees** → the offset is not the distance. Check the wavelength
and the calibrant lattice constant.

---

## Panel shifts rail at their bound

**Test.** Which per-panel degrees of freedom are refined? Then decompose the
fitted (δy, δz) field onto its global modes: uniform translation, radial
expansion, rotation.

**If δLsd or δp₂ is refined and railing** → wrong functional form. A module is
misplaced *in the detector plane*, giving a constant ΔR; a per-panel δLsd gives
ΔR ∝ R. Fitting the first with the second rails it: 16 of 48 panels with δLsd
free, 4 with δp₂ also removed, 2 with modules-only at a ±2 px bound.

**If a large fraction of the field sits in the radial-expansion mode** → it is
trading against `Lsd`. Measured 11 %, of which ~73 % is absorbable into `Lsd`.
Enable the expansion gauge (`add_panel_no_expansion_constraint`).

**Lever.** `enable_lsd=False, enable_p2=False`, bound the in-plane shift at ~2 px.
Sanity-check the result against any previous calibration's magnitudes; on the
reference detector the modules sit within ~1.5 px.

**If nothing rails and the RMS is comparable to a previous calibration** → fine.

---

## Rings look right but absolute intensities are wrong

**Test.** Compare the map entry count against a map built with a much larger
per-row buffer, or against the C `DetectorMapper`, which has no cap.

**If the counts differ** → the v1 Python mapper truncated. It drops `frac` and
`areaWeight` together, so a *normalised* profile still looks plausible while
absolute flux and bin occupancy are wrong. Measured: 42 471 entries and 1 485
whole bins lost at `RBinSize 0.5` on a 1475-wide detector, warned only under
`verbose=True`.

**Lever.** `midas-integrate ≥ 0.5.0`, or pass `per_row_max_entries` explicitly.
v2 has no map step and is not affected.

---

## Integrated pattern is subtly wrong only near module boundaries

**Test.** Is `PanelShiftsFile` named, and readable from the working directory?
Watch for `PanelShiftsMissingWarning`.

**If it fires** → the panel layout is applied with **zero** shifts. Paths are
resolved relative to the *working directory*, so a correct file in the wrong cwd
looks identical to no file at all.

**If it does not fire, and v2 is below 0.9.0** → shifts were discarded silently,
without any warning. Measured against v1-with-panels: RMS 0.116 px, max 0.560 px,
6.5 % of pixels beyond half an R bin.

**Lever.** Version floor, and run from the directory holding the shifts file.

---

## A metric returns NaN or a blank for one azimuthal band

**Test.** Count bins with data in that band, per ring, before blaming the code.

**If the band is mostly empty** → a module gap runs through it. At η = ±90 on the
reference detector only ~29 % of bins carry data, and rings clearing an
"≥ 8 valid bins" threshold dropped to 0 of 40 for one kernel and 4 of 40 for
another while a different kernel got 10.

**Lever.** This is the metric's threshold, not a defect — no integration path
emits NaN. Report the band as under-sampled and do not weight it.

---

## Cardinal-angle oscillation, and whether to enable GradientCorrection

**Test.** High-frequency residual of I(η) about a smooth fit, **with a control at
a non-cardinal angle** (45° works). Without the control you cannot separate
aliasing from ordinary noise.

**If a cardinal band exceeds the control** → real aliasing. Enable
`GradientCorrection 1` and integrate with v1 `--mode gradient`; v2 has no
gradient branch.

**If the control is worse than the cardinal bands** → there is nothing to
correct. That is what the reference data showed: control 1.809 %, η = 0 at
1.229 %, η = 180 at 1.614 %. Enabling the correction there lowers everything
uniformly, control included, which is smoothing rather than a fix.

**Do not import the published 5 % figure.** It is a property of the paper's
detector, not a constant — Lab Notebook §4.

---

## CPU and GPU results differ

**Test.** Which stack?

**v2** → they should agree **exactly**; it is float64 throughout. Measured max
absolute difference 0 over 648 000 bins. Any difference is a bug worth chasing.

**v1** → expect ~1e-7 relative from float32 reduction order. That is not a bug.
If it exceeds ~1e-5, check that both runs used the same `Map.bin`.


## Distortion coefficients sit on their bounds

**Test.** Two candidate causes, and they are told apart by *which* coefficients
rail. Run both gates
(`packages/midas_calibrate_v2/midas_calibrate_v2/pipelines/diagnostics.py:281`
and `:401`) and look at the per-parameter 1σ from the Laplace covariance.

**If the railed ones are `a_k`/`phi_k` and the azimuth gate reports a narrow
wedge** → the harmonics are not identifiable. A k-fold harmonic needs azimuth;
over a narrow arc it is degenerate with the beam centre (1-fold) and the tilts
(2-fold). Measured on a detector with the beam centre off the panel corner, 66–73°
of each ring: 3, 4, 7 and 7 of 15 coefficients pinned at ±0.002 across four
panels.

**If the railed ones are `iso_R*` and `RhoD` is much larger than the outer ring**
→ the radial terms have no lever. At ρ_max = 0.32, ρ⁶ = 1e-03 and the fitted
coefficients came back with 1σ of 0.9 to 15 on values of order 1e-03.

**Lever.** For the first, `refine_distortion="radial"` or `"none"` (rule 11);
for the second, `RhoD` = outer ring radius in µm (rule 12). Then confirm the loop
settles — a railed coefficient and a wandering loop are the same illness.

**If nothing rails and every refined coefficient is several σ from zero** → the
distortion is real and determined. Keep it.

---

## The E↔M loop will not settle

**Test.** Look at the *honest* per-iteration strain — re-extracted at each
post-refinement geometry, not the optimiser's own objective
(`packages/midas_calibrate_v2/midas_calibrate_v2/pipelines/single.py:184`). Then
re-run with the distortion frozen, changing nothing else.

**If the trace oscillates by more than ~2× and freezing settles it** → too many
sloppy directions. Measured on one frame: full distortion gave 232, 181, 613,
779 µε; `"radial"` gave 199, 284, 1380, 2718 µε; `"none"` gave 91, 72, 139, 154;
`"none"` plus a per-ring quality filter converged in two iterations at 84.2, 84.4
and stopped.

Note what that costs if unnoticed: with an oscillating trace, "best of history"
returns whichever iterate was luckiest. The 72 µε above is such an iterate — the
next iteration undid it.

**If freezing does not settle it** → not the parameterisation. Check the seed
(H2), the ring assignment, and whether the ring set is dominated by rings the
frame does not really carry (rule 13).

**If the in-loop number looks fine but re-extraction disagrees with it** → you are
reading the M-step objective, not fit quality. On a real two-calibrant frame that
gap was 41 µε in-loop against 418 µε re-extracted.

---

## Two calibrants agree, but you are not sure that means anything

**Test.** Read the absolute per-phase residual *before* the ratio
(`packages/midas_calibrate_v2/midas_calibrate_v2/loss/diagnostics.py:139`). Then
ask whether both phases could be sitting on a common noise floor.

**If the absolute residual is well above the 100 µε gate** → the agreement is not
evidence. Two calibrants described equally badly agree by construction. Measured:
one run reported 193 / 196 µε, "agree to within 1.02×", and was read as a pass;
a converged run on the same frame gave 45.6 / 69.0 µε — ratio 1.51, i.e. worse
agreement and far better absolute.

**If the absolute residual is at the floor and the phases still differ by more
than ~1.5×** → real disagreement, and it is the useful result. The honest
uncertainty on the geometry is the spread between the phases, not the fit's
formal σ. Suspect the assumed lattice constants first (da/a is degenerate with
dLsd/Lsd), then a genuine per-phase sample position (§4b.3).

**If the absolute residual is at the floor and the phases agree** → the geometry
describes both powders. This is the only combination that is a pass.

---

## The calibration passes every gate but you doubt the distance

**Test.** Compare λ from three sources: the parameter file, the filename /
metadata, and the beamline log. Then ask whether the fit could have told you.

**If they disagree** → λ and `Lsd` are degenerate in a single-distance powder
fit, so the refiner absorbed the error into the distance and the strain gate
passed anyway. On a planted 1 % wavelength error the right and wrong hypotheses
differed by only 0.063 vs 0.083 px RMS — the data cannot separate them.

**Lever.** Take λ from the beamline, not the fit. To break the degeneracy you
need several *exactly known* distances: `midas-calibrate-v2 --lsd-offsets`
refines one shared `L0` plus the known travel and a shared λ. Merely calibrating
at several distances with a free `Lsd` per image does **not** work — λ and the
distances rescale together.

**If they agree** → the distance is as good as the ring fit says.

---

## Reading a frame fails with `can't open directory (/usr/local/lib/plugin)`

**Test.** Open the file with `h5py` and read the dataset's filter pipeline
rather than its data:

```python
import h5py
with h5py.File(path) as f:
    plist = f["exchange/data"].id.get_create_plist()
    for i in range(plist.get_nfilters()):
        print(plist.get_filter(i)[0], plist.get_filter(i)[3])
```

**If a filter id is printed** (32008 bitshuffle, 32004 LZ4, 32015 zstd) → the
dataset is compressed with a plugin filter and HDF5 cannot find the plugin. It
is looking in its compiled-in default, `/usr/local/lib/plugin`, because nothing
told it otherwise. This is an EIGER / Dectris / ESRF file.

**Lever.** `hdf5plugin` ships the plugin binaries but registers them **only when
imported** — having it in `pyproject.toml` does nothing at runtime. Every MIDAS
package whose library code reads HDF5 imports it in its `__init__.py`, so
`import midas_calibrate_v2` (or any sibling) is enough. Check with
`h5py.h5z.filter_avail(32008)`. If you have written a new package that reads
HDF5, it needs the same guarded import.

**If no filter is printed** → the failure is not compression; look at the
dataset path (`data_loc`) instead.

---

## A whole detector, or a fixed fraction of it, reads as ~4.29e9

**Test.** Ask whether the extreme value is exactly the dtype maximum, and how
much of the frame it covers:

```python
import numpy as np
print(a.dtype, np.iinfo(a.dtype).max, (a == np.iinfo(a.dtype).max).mean())
```

**If it is exactly `2**32-1` (or `2**16-1`)** → these are bad-pixel sentinels
written at the *top* of the range, the Dectris EIGER convention. They are not
counts. On a real EIGER2 16M frame they are 7.10 % of the detector and map out
the module gaps. Every `img[img < 0] = 0` guard in the wild fails open on them,
so the symptom is often not this obvious — it shows up downstream as a beam
centre pulled toward a module gap, or a seed distance that will not converge.

**Lever.** `read_image(..., return_mask=True)` flags the unsigned dtype-max by
default, zeroes it, warns, and hands back the mask; write that mask out for
`--mask`. Do not build the mask statistically when the detector already tells
you — the §2 azimuthal-median recipe is for detectors with no bad-pixel map.

**If the extreme value is not the dtype max** → it is a real count or a
saturation, not a sentinel; check the detector's saturation level instead.
