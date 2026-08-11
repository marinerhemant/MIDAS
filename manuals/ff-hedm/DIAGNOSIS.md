# FF-HEDM diagnosis reference

Symptom → discriminating test → cause → lever, for far-field HEDM. Read by
`midas_ff_report_beamreport.py` via beamreport; each entry attaches to a symptom the
generic diagnostics detect.

Source of the content: `manuals/Reconstruction_Reports.md` §4. Every entry carries a
test that can come back the other way — an entry that cannot exonerate the cause it
names does not belong here.

---

## Detector centre offset

symptom: trend.amplitude_constant
coord: eta_deg

**Test.** The amplitude of the azimuthal sinusoid is compared across bins of ring
radius. A rigid detector-centre shift displaces every ring by the same *absolute*
distance, so the amplitude in µm stays constant with radius. If instead it grows with
radius, the centre is not the cause and this entry does not apply — see the
sample-displacement entry.

Confirm it is global rather than per-grain: compare the mean of the per-grain (dy, dz)
offsets against their scatter. Mean much larger than scatter means a common offset.
Mean near zero with large scatter is genuine per-grain position spread, which is not a
bug.

**Cause.** The beam centre used in reconstruction differs from the true one. Amplitude
divided by pixel size gives the offset in pixels; Δtan shows the same amplitude 90° out
of phase.

**Lever.** Recalibrate BC, Lsd, tilts and distortion against a powder calibrant taken
with the same geometry, then re-index. Transplant `{Lsd, BC, tx, ty, tz, p0..p10,
RhoD}` as one block — a new distortion set with an old RhoD silently corrupts the
distortion.

## Sample displacement or distance error

symptom: trend.amplitude_growing
coord: eta_deg

**Test.** Same comparison as above, read the other way: the sinusoid amplitude grows
with ring radius rather than staying constant in µm. If it is flat in absolute units,
this entry does not apply and the detector-centre entry does.

**Cause.** A sample displacement or an Lsd error, both of which scale the ring pattern
rather than translating it.

**Lever.** Refine Lsd against a calibrant. Powder cannot constrain `tx` (rotation about
the beam) — keep it fixed there and refine it from grains in a second pass.

## Position refinement runaway

symptom: param.residual_correlated
coord: Z

**Test.** Correlate each grain's fitted Z against its own vertical residual. Near-zero
correlation, with residual flat against Z, means the Z values are supported by the
spots and the spread is physical. A strong negative correlation — core grains at
roughly zero residual, tail grains carrying residuals pointing back toward the beam
centre — means the spots contradict the assigned Z.

Rule out geometry before concluding: compare the ring composition of the core against
the tail. If they are identical, a ring-dependent tilt or distortion error is not the
cause.

**Cause.** The position fit is placing grains outside the illuminated slab, where they
could not have diffracted. The tail is a fitting artifact, not structure.

**Lever.** Set `Hbeam` / `BeamThickness` to the true per-layer beam rather than the
full sample height. A ten-layer 100 µm scan often carries `Hbeam 1000`, which lets Z
roam ±500 µm. Grains outside the beam cannot diffract, so this is a physical prior, not
a fudge. Then re-check that the dz residual stays flat against Z.

## Bound-limited positions

symptom: bound.pileup
coord: Z

**Test.** Divergence-to-bound leaves a pile-up *at* the bound. If the outer shell holds
close to zero percent of grains, the bound is not being reached and this entry does not
apply — look instead at whether the residual supports the fitted positions.

**Cause.** The optimiser is running into `Hbeam/2` or `Rsample` rather than converging.

**Lever.** Widen the bound only if the physics justifies it; usually the correct move is
the opposite, since a generous bound is what let positions roam in the first place.
Never set these to the actual sample dimensions.

## Reference-lattice or wavelength scale error

symptom: systematic.common_offset

**Test.** Look at how the per-ring radial bias behaves in ppm. Constant ppm across
rings points at Lsd, since `δR/R = δLsd/Lsd` is ring-independent. Growing ppm with 2θ
points at the reference lattice or wavelength instead. If the ppm range across rings is
under roughly 200, neither is worth chasing.

**Cause.** A shared offset in the strain-free reference, the wavelength, or the
detector distance. For a cubic free-standing polycrystal the equilibrium condition
reduces exactly to a zero volume-averaged hydrostatic strain, so any nonzero mean *is*
the d0 error.

**Lever.** Recover d0 with `midas_stress.recover_d0_cubic_free_standing`. The
correction is purely isotropic, so deviatoric strain is unchanged: it fixes bias, never
scatter. Report the stress impact as `eps_iso × 3K` — it is usually the headline number
and often hundreds of MPa.

## Two populations in completeness or spot count

symptom: split.bimodal

**Test.** Check whether the split is spatial. Map the two populations onto grain
positions: if the split follows position, it is the illumination footprint — which part
of the sample the beam actually covered — and not a reconstruction defect.

If the split is *not* spatial, bin grains by radial distance from the rotation axis and
histogram within each bin. Modes that move with radius indicate a smooth geometric
effect. Mode positions fixed across radius, with only the population fraction shifting,
indicate a discrete algorithmic branch.

**Cause.** Spatial split: illumination coverage. Non-spatial with fixed modes: a solver
branch, most often the Friedel-pair position path succeeding versus falling back.

**Lever.** For the footprint case, nothing to fix — report it as a property of the
scan. For the branch case, re-run a subset with `UseFriedelPairs 0`; if the split
collapses, the Friedel path is the branch. Expect the bad branch to also carry inflated
|Z|, internal angle and strain error, and verify they co-move before blaming one cause.
