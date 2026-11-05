# hedm-kit: far-field HEDM for a minor cubic phase on a Varex station (20-ID-D style)

A set of scripts and a procedure to go from raw Varex sweeps to **indexed grains of a target phase** (a cubic phase that sits next to a
dominant phase), with the checks that decide whether the grains are real. It was built on one reference beamtime (a battery composite:
a dominant layered oxide plus a cubic sulfide electrolyte, Sm K-edge, 20-ID-D). Everything marked PROVISIONAL below rests on that one beamtime.

**Status words used here.** VERIFIED = checked by a person against the instrument or by a fresh independent test. ESTABLISHED = survived `/verify`.
PROVISIONAL = one dataset, not independently verified. UNVERIFIED = a convention or prior we did not test. A recipe that ran is not a result.

## What it does NOT do
Not a general FF-HEDM tutorial (see `manuals/FF_Analysis.md`). Not tomography (use the `tomo` skill/package). The pf-HEDM part is **experimental**: on the
reference beamtime it ran end to end but returned a grain at 1 of 11 positions. Strain from the grains was unusable there (RMS strain error 3 to 6 thousand
microstrain per grain). Positions from far-field are good to ~100 micrometres.

## Layout
```
kit_env.sh                    source it (EXP=<experiment>): paths, WORK, NCPUS, kit_check, pick_gpus
selftest.sh                   no-data self-test: syntax, imports, the line finder on synthetic input
experiments/_template/        experiment.env: every beamtime FACT with its source (energy, phases, calibration files, sample list)
survey/survey_scans.py        one row per scan folder AND per file: frames, aero span, stale scan_parameters, dark location, slits
tools/look_frames.py          look at raw frames before any threshold
tools/ring_survey.py          beam centre from the data, radial profile, rings -> d, line counts and chance rate of candidate phases
tools/phase_hunt.py           lines the dominant phase does not explain; cubic indexing; candidate table; planted control
tools/phase_hunt_all.py       the hunt over a CSV of scan folders, plus one summary table
tools/ring_f2.py              per-ring |F|^2 of a phase from a CIF: which rings are intrinsically weak
tools/fit_hex_lattice.py      a, c of a hexagonal dominant phase from its ring radii
calib/run_cali.sh             CeO2/LaB6 calibration at the REAL energy
ff/make_bg_dark.py            per-pixel temporal median as the dark (without it no zipper threshold works on dense rings)
ff/thresh_rule.py             threshold rule from background-subtracted test frames
ff/make_params.py             Parameters.txt from a calibration output plus sample keys
ff/check_zarr.py, ff/ff_check.py   checks on the zip and on a finished run
ff/target_ff.sh               the chain for ONE layer of the target phase: prep | seed | index | nulls | readout
ff/run_arm.py, ff/arm_compare.py  omega-scrambled null arms; distinct-orientation counts
ff/ff_readout.py              real grains versus the null arms
pf/pf_peak_one.sh, pf_run.sh  EXPERIMENTAL pf recipe
```

## Procedure
Put your outputs in `$WORK` (default `/data/$USER/<EXP>_work`, local to ONE host: results made on one machine are not visible on another).
Long jobs: `nohup ... > log 2>&1 &` and check the log, never inline over ssh. Mac has no `setsid` and no GNU `timeout`.
The hosts are SHARED: `export NCPUS=16` before `source kit_env.sh` unless you need more (the default is every core up to 64).
Run every command from the kit folder, after `export EXP=<experiment>; source kit_env.sh`; `$RAW`, `$WORK` and `$PY` come from that.

0. **Facts first.** Read the beamtime explog and fill `experiments/<EXP>/experiment.env`: energy (the absorption edge actually used, **not** the file-name value),
   distance, phases, calibration files, who verified the omega sign. The Varex file does not record the energy.
1. **Survey.** `$PY survey/survey_scans.py $RAW/varexD $WORK/survey.csv ["^name-regex$"]` writes `survey.csv` (per folder) and `survey_files.csv` (per file); without the regex it walks every folder (slow on a big beamtime), and
   loose files in the root folder are listed as `(loose files in the root folder)`. `layer_pitch_um` is the median |step of samY| between layer files. Read the per-file table: the first file hides truncated sweeps,
   and `scan_parameters` inside a file can be stale (the real sweep is the aero readback span).
2. **Look at raw frames.** `tools/look_frames.py <data.h5> <dark_before.h5> <prefix>`. Frame 0 of every sweep is a throwaway (`SKIPFRAME 1`).
   The dark lives in `exchange/bright` of the `dark_before` file; `exchange/dark` there is all zero.
3. **Calibrate** at the real energy: `calib/run_cali.sh`. Two exposures (0 and 180 degrees) should agree on Lsd. Read the ring overlay PNG.
4. **Hunt for the phase.** An experiment ships (or you write) a `samples.csv` (name,folder,energy_keV,lsd_mm) and a `candidates.json`. `--primary` is `PRIMARY_SG:PRIMARY_LAT` from `experiment.env` with the lattice as COMMAS, e.g. `166:2.876,2.876,14.20062,90,90,120`.
   `mkdir -p $WORK/hunt; $PY tools/phase_hunt_all.py samples.csv $WORK/hunt --workers 2 --primary <sg:a,b,c,al,be,ga> --candidates candidates.json` (2 to 4 minutes per folder: one 24 GB sweep each).
   Read `$PY tools/phase_hunt.py --help` first (the header explains every column). In the summary `ring`/`spot` = lines the primary phase does not explain; "k of n obs, w in window" = k of the n observed ring lines match a line
   of that cell, and the cell has w lines in the observed d window. The key rules: a line list with 6+ lines fits some cubic cell by chance (read the candidate table with its chance rate, not the free fit);
   continuous rings show up in the median image, single crystallites only in the max image; and **no lines found is not a negative until the planted control recovers its planted rings.**
   On the reference data the control recovered 7 of 20 planted rings at 1000 spots per ring (35 %, bar 80 %), so folders with "none seen" were INCONCLUSIVE, not negative.
   **How to run the control** on the folder you doubt: an unplanted run `$PY tools/phase_hunt.py <file> $WORK/hunt/base <E_keV> <lsd_mm> --primary ...` gives `A_ref` in `base_hunt.json`; then
   `$PY tools/phase_hunt.py <file> $WORK/hunt/plant <E_keV> <lsd_mm> --primary ... --plant 9.85,F,1000,0.1 --aref <A_ref>` (cell, family P/I/F, spots per ring, fraction of A_ref); count how many `planted.R_px` radii
   appear among the lines found. The reference control was run on a weak 85wt cathode file.
5. **Structure factor of the target** (optional, recommended): `tools/ring_f2.py phase.cif --lsd .. --lambda .. --a <calibrated cell>`. Weak rings are weak by structure,
   not by failure (reference phase: rings with |F|^2 per spot of 1 to 5 percent of the strongest). Declare the CIF as `TARGET_CIF`: `hkls.csv` then carries an F2 column.
6. **FF chain for one layer:** `ff/target_ff.sh prep <scan_folder> <file_nr>` (long: peakfit 2.7 h for one 1442-frame layer on 2 GPUs). `DRY=1` still runs the first stages for real (about 11 minutes, a ~12 GB zip under `$WORK`, one ring-thresh process at ~9 GB RAM) and only prints the peakfit command;
   the RingThresh answer is the final "Paste into the parameter file" block of `ringthresh.txt` (also written as one line to `RingThresh.txt`),
   then `seed`, `index`, `nulls`, `readout`. Run the stages one at a time and read each output.
   `GATE=<x>` (default 0.5) is applied by the `index` and `nulls` stages; the reference comparison used `GATE=0.35`.
7. **Read the readout against what you decided BEFORE you saw it.** `ff/ff_readout.py` prints grains per arm, N_above (real grains above every null grain) and the AUC.
8. **pf:** see `pf/pf_run.sh` and its header. Experimental.

## Traps (each one cost a run on the reference beamtime)
1. **Energy from file names.** The name said 46.000 keV; the explog said 46.834 (Sm K edge): 1.8 percent in wavelength, a void 14 h run.
2. **Zipper memory.** One zipper holds ~44 GB (24 GB sweep). Eleven at once killed six jobs (OOM). Start them 100 s apart, two at a time.
3. **Pseudo-dark.** On dense continuous rings no preproc threshold works without subtracting a per-pixel median. `ff/make_bg_dark.py`.
4. **The zip carries the parameters it was built with.** Build the ungated zip with the full ring list before `midas-ring-thresh`, or it sees one ring and writes `RingThresh 1:75`.
5. **Binning out of GPU memory** on dense layers: `--binning-device cpu` (3 min). A parameter-refresh guard refuses a changed NrFilesPerSweep on prebuilt zips: `--no-refresh-params`.
6. **Seed ring.** Seeds are the spots of `OverAllRingToIndex` with omega in [-90, 90]. The reference run was seeded from the 1 percent ring and found 59 grains above the null;
   seeding from a ring with a moderate |F|^2 and ~10 000 seeds found 842 (raw metric, one sample, PROVISIONAL). `ff/target_ff.sh seed` prints the table and, with the wave-3
   `midas-pipeline`, its recommendation; the binning stage also logs a warning for a weak seed ring.
7. **Do not weight completeness by |F|^2.** `ConfidenceMetric weighted` found fewer real grains above the null than raw at all three seed rings tried (it boosts the crowded rings
   where chance matching is already 55 to 69 percent). Hard-culling faint reflections measured badly on another phase (registry note on `ForbiddenF2Threshold`).
8. **Confidence at the completeness floor.** Real grains sat at 0.50 to 0.61, chance orientations at 0.37 +- 0.03 (max 0.48): the gap is ~0.15. A pile-up at the gate is a cut, not a population edge.
   **Always run the null arms.** A dense dominant phase gave 17 374 "grains" at confidence 0.59 to 0.64 that look like chance matches (no spatial structure).
9. **Zero-grain arm exits 1** in `midas-process-grains` before 0.13.2 (crash on the empty list). Zero grains is then the result.
10. **Rings near the other phase.** A ring within ~9 px of a dominant-phase line collects its wing spots (rings 9 and 10 of the reference run). The ring selection excludes rings within 7 px; check the wings.
11. **Omega sign and mirror.** Sign: say who verified it and how. The mirror (`IMTRANS`) and the pf positions direction are separate facts. A wrong sign kills the effect, so the null arms and a Friedel check expose it.
12. **Matrix convention.** MIDAS Grains.csv orientation matrices map crystal to lab. `midas_plotting.ff.pole_figure` plotted `g.T @ h` before its fix (wave-3 release); check the side with a hkl-integrality test on real spots.
13. **A CIF from a licensed database must not go into a shared folder.** Record the entry number and where to get it.

## Costs (reference beamtime, one 1442-frame layer, 2 A6000, 64 cores)
Pseudo-dark and hkl: minutes. Peakfit: 2.7 h for the target rings only. Transforms: 3 min. Indexing: 26 s for ~900 seeds, ~5 min for ~10 000. A null arm: transforms 3 min plus indexing.
The whole pf pass of 41 scans would be ~30 h of peakfit on two GPUs.

## Test it first
`EXP=_template ./selftest.sh` (syntax, imports, a synthetic planted line). Then run each step on ONE layer and compare with the experiment's `reference/` outputs, if it has them.
