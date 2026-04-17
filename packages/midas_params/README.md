# midas-params

Parameter-file registry, validator, and (coming soon) wizard for MIDAS
FF-HEDM, NF-HEDM, PF-HEDM, and radial-integration pipelines.

## Status

- **Registry**: FF + NF + PF + RI (~195 keys).
- **Parser**: MIDAS text format (single-entry, multi-entry, aliases, inline comments).
- **Validator**: 14 per-key validators, 11 cross-field rules, typo detection.
- **Discovery**: filename parsing, directory scan, HDF5/Zarr probing.
- **Wizard**: interactive + non-interactive, seeds from existing / calibration / dataset files.
- **Rings**: Bragg-ring geometry helper (λ + Lsd + lattice → visible ring list).
- **Diagnose**: LLM-ready payload (`format=json|prompt`) with source + registry + pipeline context.
- **CLI**: `validate`, `inspect`, `rings`, `wizard`, `diagnose`.

## Why

`AutoCalibrateZarr` emits a refined detector geometry file. Running a full
analysis from it requires ~25 additional parameters (sample material, rotation
scan, ring selection, indexing tolerances). Users typically copy an example
file, forget to change something, then file support tickets. This package is
the first step toward an interactive wizard + validator that catches most
common mistakes before they reach the pipeline.

## Quick usage

### CLI

```bash
# Check a param file
midas-params validate my_params.txt --path ff

# Same, as JSON (for CI / LLM / IDE integrations)
midas-params validate my_params.txt --path ff --json

# Auto-extract what we can from a dataset file
midas-params inspect /data/exp/sample_000042.ge3

# Non-interactive build: seed from calibration + dataset, write param file
midas-params wizard --path ff \
    --out new_params.txt \
    --from-calibration refined_MIDAS_params.txt \
    --dataset /data/exp/sample_000001.ge3 \
    --non-interactive

# Or run interactively
midas-params wizard --path ff --out new_params.txt \
    --from-calibration refined_MIDAS_params.txt

# Figure out which rings fall on the detector
midas-params rings --from refined_MIDAS_params.txt --max-rings 10
# Or with CLI args
midas-params rings --wavelength 0.22291 --lsd 1000000 \
    --lattice 4.08 4.08 4.08 90 90 90 --space-group 225 --rhod 200000

# Build an LLM-ready diagnosis of a broken param file
midas-params diagnose my_params.txt --path ff --format prompt > diagnosis.txt
# Paste diagnosis.txt into Claude / GPT to get explanations + proposed fixes
midas-params diagnose my_params.txt --path ff --format json > diagnosis.json
```

The wizard seeds values in priority order:

```
--from-existing file  >  --from-calibration file  >  --dataset probe  >  registry typical / default
```

### Python

```python
from midas_params import Path
from midas_params.validator import validate, format_report

report = validate("my_params.txt", Path.FF)
print(format_report(report))

if not report.ok:
    for issue in report.errors:
        print(f"line {issue.line}: {issue.message}")

# Structured output (every issue is a dataclass — JSON-serializable)
from dataclasses import asdict
payload = [asdict(i) for i in report.issues]
```

### Discovery standalone

```python
from midas_params import discover_from_file, discover_from_calibration_file, merge

# From a single raw frame
d1 = discover_from_file("/data/exp/sample_000042.ge3")
# From a refined_MIDAS_params.txt
d2 = discover_from_calibration_file("refined_MIDAS_params.txt")
# Merge with priority (earlier wins)
seeded = merge(d2, d1)
print(seeded.extracted)     # e.g. {"Lsd": 1000000, "FileStem": "sample", ...}
print(seeded.confidence)    # "high" / "medium" / "low"
print(seeded.source)        # "param-file:refined.txt" / "dir-scan (20 files)" / ...
```

## Architecture

```
registry.py      — single source of truth: list[ParamSpec]
schema.py        — dataclasses (ParamSpec, CrossFieldRule, ValidationIssue)
validators.py    — per-key validation functions, looked up by name
crossfield.py    — multi-key consistency rules
parser.py        — MIDAS text format → typed dict with line numbers
validator.py     — engine: walks registry + rules, produces ValidationReport
discovery.py     — auto-extract from raw frame files / HDF5 / Zarr / param files
rings.py         — Bragg ring enumeration + detector projection
wizard.py        — interactive + non-interactive param-file builder
diagnose.py      — LLM-ready payload builder (validator + registry + primer)
cli.py           — `midas-params` entry point
```

Everything in the registry is JSON-serializable (no function objects), so
external tools — including an LLM diagnosis layer — can consume it directly
by reading the dataclasses as dicts.

## Adding a parameter

1. Add a `ParamSpec(...)` entry in `registry.py` under the appropriate
   category.
2. Reference validator names (strings) in `validators=(...)` — resolved against
   `validators.VALIDATORS` at load time.
3. For new cross-field rules, add a function to `crossfield.py`, register it
   in `RULES`, and add a `CrossFieldRule(...)` declaration in `RULE_SPECS`.

Defaults MUST come from source, not guesses. For MIDAS_ParamParser.c the
authoritative defaults are in `midas_config_defaults()`. For NF inline
parsers, check the executable's C source.

## Tests

```bash
cd packages/midas_params
pytest
```

Tests validate against real `FF_HEDM/Example/Parameters.txt` and
`NF_HEDM/Example/ps_au.txt`, so changes to those files can surface here.

## Related docs

- [manuals/FF_Parameters_Reference.md](../../manuals/FF_Parameters_Reference.md)
  — full FF parameter reference (source-verified).
- [manuals/NF_Parameters_Reference.md](../../manuals/NF_Parameters_Reference.md)
  — full NF parameter reference.
