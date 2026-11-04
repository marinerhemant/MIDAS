# validation/

The registered tests of midas-sx, kept so a number in the README can be traced to the code that made it. This directory is NOT part of the wheel (`packages.find include = midas_sx*`).

## What is in git

- `PREREGISTER.md`, `DESIGN.md`, `RELEASE_READINESS.md`, `DATA_HUNT*.md`: the registrations (fixed before the data), the readings and their /verify outcomes, the design limits.
- `*.py`, `*.sh` here: the scripts of the simulated validation (A1 to A36). `real/*.py`, `real/PREREG_*.md`: the scripts and registrations for the three public data sets (A37 Rochelle salt, A38 thiourea, A39 L-alanine). `synthesis/make_synthesis.py`: builds a summary page from the results.
- `simulate.py`: a shim; the independent simulator lives in `../tests/support/simulate.py` because the unit tests import it.
- `results/`: ONLY the records the readings cite (claim texts `CLAIM_*.md`, verdict files, the small tables and summaries named in a reading), 284 files, about 2.7 MB. `results/.gitignore` is an allowlist.
- `results/MANIFEST.csv`: every OTHER result file (1764 files, 60 MB: csv, npz, pkl, logs, json of the runs) with its size, sha256, archive path and a script name hint (name match only, not a record).

## What is not in git, and how to get it back

All of it is regenerable by the scripts above (simulated runs are seeded; the real data sets are public Zenodo records, see `real/PREREG_*.md` for the DOIs). The outputs were archived once, before the reorganisation:

    /gdata/dm/MPE/OrthrosJr/analysis/sharma_work/midas_sx_validation_archive/   (on copland)

Restore with `rsync -a copland:<that dir>/results/ validation/results/` and check with `sha256sum -c` against the manifest columns. Scripts carry hard-coded Mac and copland paths (`/Volumes/T5 EVO/...`, `/gdata/dm/MPE/OrthrosJr/...`); they are records of what was run, not portable tools. `MIDAS_SX_PATH` points the real-data scripts at the code.

## Status words

Used exactly as logged in `PREREGISTER.md`: ESTABLISHED means a four-lens fresh-context /verify survived; everything else is PROVISIONAL, INCONCLUSIVE, REFUTED or NO GAIN as written there. See `../../../manuals/sx/ENVELOPE.md`.
