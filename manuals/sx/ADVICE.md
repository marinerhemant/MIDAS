# Reading the advice report (`midas_sx.advice.assess`)

PROVISIONAL. Claim a0c60827daf1: four fresh-context lenses all returned UNCERTAIN; every number in the validation reproduced. Two checks were not detections as first written and were repaired; the repair went through a second four-lens /verify (claim a56630731be7): statistics, reproduction and artifact SURVIVES, physics UNCERTAIN on scope, verdict PROVISIONAL (limits below). The wording of the actions and the quoted gains was NOT tested; only which finding fires and at which severity.

## What it is

`assess(runs, crystal, acquisition=Acquisition(...), ...)` takes the reductions of one or more rotation runs and returns an `AdviceReport` of `Finding`s. Each finding has: an id, a severity (`blocking`, `limiting`, `note`), a title, the measured values, the RULE with its threshold (all thresholds are in `advice.THRESHOLDS`), the evidence it was computed from, an ACTION for the person at the beamline, a GAIN (what the fix bought on this data, when known) and what the check does NOT cover. `report.ok` is false when anything is blocking or limiting. `report.numbers` carries numbers reported whether or not a finding was raised.

Pass what you know in `Acquisition`: `exposure_s`, `count_cutoff` (detector header), `has_i0`, `has_calibrant`, `max_range_deg` (the sample environment's limit), `dac` (the cell model, so blocked reflections are not counted as missing). `assess(..., d_min=...)` sets a resolution target.

## The checks (rule, what it is for, how far it was tested)

| id | Fires when | Tested in simulation |
|---|---|---|
| scan_range | rotation range below 180 deg: few in-run Friedel pairs. Limiting only if the geometry was refined from the data, fewer than 30 in-run pairs exist, and the range is not set by the environment | 40 deg scan: limiting (26 pairs against 30: thin margin); a scan equal to the declared allowance: note |
| completeness, redundancy, next_setting, direction_holes | share of accessible unique reflections seen at least once/twice, mean multiplicity, and the omega setting that would add most | one run: completeness and redundancy limiting; clean 6 runs at high counts: none |
| resolution_limit | outermost shell of the expected reflections is mostly undetected (`resolution_signal_limited`, limiting) or the data reach the detector edge | after a survivorship repair; the 0.66 A target is tuned to the simulator |
| systematic_floor | equivalents disagree by more than counting (f at least 8 % and at least 2 x counting: limiting) | limiting from per-frame scatter sigma about 0.14 and above; floor about 0.58 x sigma |
| intensity_drift | smooth scale range at least 0.15 in log, R_int gain at least 25 %, and a permuted-frame placebo gains under half as much | amplitude 0.10 and above limiting, 0.05 a note |
| dynamic_range | REPAIRED (Amendment 7): saturated (limiting) when at least 2 % of observations have a peak pixel at 99 % of the cutoff, or with no cutoff when at least 3 blobs tie at the maximum above 1e4 counts; separately a note for the 25 %-of-cutoff count-rate range, which is NOT a detection of saturation | second /verify PROVISIONAL (3 lenses SURVIVES, physics UNCERTAIN). Blind when the background exceeds about 1 % of the cutoff; the tie rule is not run when a cutoff is declared; the 2 % bar counts observations, not intensity |
| omega_sampling | REPAIRED (Amendment 7): note when at least 25 % of the blobs lie on a single frame (the width is an RMS of integer frame indices, not an extent) or the median RMS width exceeds 30 frames (the oversampled rule is untested) | second /verify PROVISIONAL. Single-frame fraction by step 0.25, 0.5, 0.75, 1.0 deg: 0.008, 0.18, 0.38, 0.50 (simulated); the bar 0.25 is crossed near 0.58 deg, so a 0.5 deg step is NOT flagged. The bar was chosen after seeing 0.50 against 0.013 and is specific to the simulator's spot width: real L-alanine spots are about 3.5 x broader, so on such data the check fires only well above 1 deg |
| geometry_consistency | refined wedge, a tilt or their run-to-run spread above 0.1 deg: a NOTE (the fit models it); limiting only for a non-converged or poor orientation fit | wedge 0.1 to 0.4 deg reported as a note, silent at 0.05 deg; it never became limiting |
| contamination | crowded strong reflections, near-beam scattering, constant-2theta rings among unexplained strong blobs | ring found at amplitude 0.5 (limiting) and 0.25 (a note), blobs average 11.999 deg for an injected 12.0 |

## Worked example: L-alanine, three runs (`validation/results/advice_validation/amend7/a39_all_amend7.log`)

2 limiting, 6 notes.
- LIMITING crowded strong reflections: 23 % of the strongest blobs are lost as crowded. Action: a smaller step, a longer detector distance, a smaller beam or crystal; check the lost reflections are not the low-angle ones.
- LIMITING intensity drift: the scale varies by 78 % within a run (log range 0.58); a smooth scale cuts R_int from 15.7 to 8.2 %; the permuted-frame placebo gives 15.3 %. Action: check that the crystal is centred on the axis and the beam covers it consistently; record the incident intensity per frame.
- Notes: wedge up to 0.41 deg (the fit models it); detector reaches the edge with signal to spare; scan range 170 deg with 38, 57 and 47 in-run Friedel pairs (scan 180 deg or more); a systematic floor of 5.7 % (2.1 x counting), small in absolute terms; 13 ring-like peaks among unexplained blobs; 2 of 3937 observations above 25 % of the cutoff.
- No omega_sampling and no saturation finding on this data. One real clipped reflection IS in this data (run 04: a blob at 1.0000 of the cutoff, frame pgw240050_04_00416, row 760, col 257) but it was rejected as crowded, so it is not an observation and the saturation finding stays silent; `blobs_clipped_per_run` in the numbers shows it. The strongest reflections are the ones most often dropped as crowded.

## What the validation does NOT show

- Advice WORDING and quoted gains are untested. Only the firing and the severity.
- One simulated structure, three orientations, single clean injected defects, one real crystal. Not tested on the Rochelle or thiourea sets.
- Specificity: no blocking/limiting finding on 12 fresh clean scenes (seeds 9-20), but only 6 are independent: the 95 % upper bound on a false-positive rate is 0.39 (0.24 if the 12 were independent, which they are not).
- THRESHOLDS cite L-alanine numbers (drift 0.37-0.58, floor 5.9 and 14.2 %, omega width 1.3-2.4): thresholds and the real-data test share a data set. The simulated floor dose that fires (0.145) matches that data's 14.2 %.
- The real-data match (14 of 14) is a regression check, not an independent test: three expectations are the refined wedges read from the input reduction, the rest call the same package functions as the hand numbers.
- The validation took seven amendments after seeing results: two module repairs found by the validation (cell aperture, resolution survivorship), one option (`d_min`), scenario errors of the registration, and the D2/D3 repair after /verify. As registered it was REFUTED.

## Limits of the saturation and omega checks (from the second verify)

- Saturation is tested at 99 % of the declared cutoff on the background-subtracted blob peak: a background above 1 % of the cutoff hides it (a uniform 2500 counts on a 226888 cutoff: 437 raw pixels at the cutoff, none flagged). With no cutoff declared, at least 3 blobs within 0.1 % of the maximum above 1e4 counts count as clipping; with a WRONG cutoff declared (2 x too high) nothing is flagged. Declare the header's cutoff.
- The 2 % bar counts observations. 24 clipped blobs (1.6 %) held 25.5 % of the integrated intensity and raised no finding.
- A cutoff declared below the data makes the saturated finding fire together with a note that peaks exceed the declared cutoff.

## Acquisition lessons the report rests on (from L-alanine)

Record I0 for every frame; measure a calibrant (CeO2 or LaB6) at the same energy and distance; scan 180 deg or 360 deg for in-run Friedel pairs; use a step of about a quarter of the spot width; attenuate or shorten the exposure for the strongest low-angle reflections; check the crystal is centred on the axis.
