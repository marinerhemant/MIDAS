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
