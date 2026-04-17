"""Command-line interface for midas-params.

Subcommands:
  validate  — check a param file against FF/NF/PF/RI rules
  inspect   — show what discovery extracts from a dataset file
  wizard    — interactive param-file builder (FF/NF only for now)

Usage:
  midas-params validate params.txt --path ff
  midas-params inspect /data/exp/sample_000001.tif
  midas-params wizard --path ff --from-calibration refined.txt --out new.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path as FsPath

from .schema import Path, Severity


def _path_from_str(s: str) -> Path:
    try:
        return Path(s.lower())
    except ValueError:
        raise SystemExit(f"--path must be one of: ff, nf, pf, ri (got {s!r})")


def cmd_validate(args: argparse.Namespace) -> int:
    from .validator import validate, format_report

    report = validate(args.file, _path_from_str(args.path))
    if args.json:
        # Serialize the report as JSON (for CI, LLM consumption, IDE plugins).
        payload = {
            "param_file": report.param_file,
            "path": report.path.value,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "ok": report.ok,
            "issues": [
                {
                    "severity": i.severity.value,
                    "key": i.key,
                    "line": i.line,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "rule": i.rule,
                    "stage": i.stage.value if i.stage else None,
                }
                for i in report.issues
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        use_color = sys.stdout.isatty() and not args.no_color
        print(format_report(report, use_color=use_color))
    return 0 if report.ok else 1


def cmd_inspect(args: argparse.Namespace) -> int:
    from .discovery import discover_from_file

    result = discover_from_file(args.dataset)
    if args.json:
        print(json.dumps({
            "extracted": result.extracted,
            "confidence": result.confidence,
            "source": result.source,
            "warnings": result.warnings,
        }, indent=2, default=str))
        return 0

    use_color = sys.stdout.isatty() and not args.no_color
    green = "\033[32m" if use_color else ""
    yellow = "\033[33m" if use_color else ""
    dim = "\033[2m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    print(f"Inspecting: {args.dataset}")
    print("=" * (12 + len(str(args.dataset))))
    if not result.extracted:
        print(f"  {yellow}No parameters extracted.{reset}")
    else:
        col_w = max(len(k) for k in result.extracted) + 2
        for key, val in sorted(result.extracted.items()):
            conf = result.confidence.get(key, "?")
            src = result.source.get(key, "?")
            color = green if conf == "high" else yellow
            print(f"  {color}{key:<{col_w}}{reset} = {val!r:<30} {dim}[{conf}, {src}]{reset}")
    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .validator import validate
    from .diagnose import build_diagnosis_payload, format_diagnosis_prompt

    report = validate(args.file, _path_from_str(args.path))
    payload = build_diagnosis_payload(
        report,
        include_source=not args.no_source,
        include_registry_context=not args.no_registry,
        include_primer=not args.no_primer,
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, default=str))
    else:  # prompt
        print(format_diagnosis_prompt(payload))
    return 0


def cmd_rings(args: argparse.Namespace) -> int:
    from .parser import parse_typed
    from .rings import enumerate_rings, format_ring_table, recommend_rings

    # If --from is given, pull values from a param file; else use CLI args.
    values: dict = {}
    if args.param_file:
        parsed = parse_typed(args.param_file)
        values = dict(parsed.values)

    # CLI args override param-file values
    for name in ("wavelength", "lsd", "rhod", "space_group", "nr_pixels_y", "px"):
        v = getattr(args, name, None)
        if v is not None:
            values[name] = v
    if args.lattice:
        values["lattice"] = [float(x) for x in args.lattice]

    wl = values.get("wavelength") or values.get("Wavelength")
    lsd_raw = values.get("lsd") or values.get("Lsd")
    lsd = lsd_raw[0] if isinstance(lsd_raw, list) else lsd_raw
    lat = values.get("lattice") or values.get("LatticeConstant")
    sg = values.get("space_group") or values.get("SpaceGroup")
    rhod = values.get("rhod") or values.get("RhoD") or 1e9
    ny = values.get("nr_pixels_y") or values.get("NrPixelsY") or values.get("NrPixels")
    px = values.get("px")

    missing = [name for name, v in
               [("wavelength", wl), ("lsd", lsd), ("lattice", lat), ("space_group", sg)]
               if v is None]
    if missing:
        print(f"error: missing required inputs: {missing}", file=sys.stderr)
        print(f"       provide via --from <param-file> or "
              f"--wavelength/--lsd/--lattice/--space-group flags", file=sys.stderr)
        return 2

    try:
        rings = enumerate_rings(
            wavelength=wl, lsd_um=lsd, lattice=list(lat), space_group=sg,
            rho_d_um=rhod,
            nr_pixels_y=ny, px_um=px,
            max_rings=args.max_rings,
        )
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        import json as _json
        from dataclasses import asdict
        print(_json.dumps([asdict(r) for r in rings], indent=2, default=str))
        return 0

    use_color = sys.stdout.isatty() and not getattr(args, "no_color", False)
    print(f"Detector: Lsd={lsd:g} µm  RhoD={rhod:g} µm  "
          f"λ={wl:g} Å  SG={sg}  a={lat[0]:g} Å")
    print()
    print(format_ring_table(rings, use_color=use_color))
    print()
    rec = recommend_rings(rings)
    if rec:
        print(f"Suggested RingThresh / OverAllRingToIndex: {rec}")
        print(f"  RingThresh 1 10    # and so on for rings {rec}")
    return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    from .wizard import run_wizard

    return run_wizard(
        path=_path_from_str(args.path),
        output=args.out,
        from_calibration=args.from_calibration,
        from_existing=args.from_existing,
        dataset_file=args.dataset,
        non_interactive=args.non_interactive,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="midas-params",
                                 description="MIDAS parameter-file tools.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # validate
    sv = sub.add_parser("validate", help="Check a parameter file.")
    sv.add_argument("file", help="Parameter file to validate.")
    sv.add_argument("--path", required=True, choices=["ff", "nf", "pf", "ri"],
                    help="Which MIDAS pipeline to validate against.")
    sv.add_argument("--json", action="store_true",
                    help="Emit JSON report (for CI / tooling).")
    sv.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    sv.set_defaults(func=cmd_validate)

    # inspect
    si = sub.add_parser("inspect", help="Auto-extract params from a dataset file.")
    si.add_argument("dataset", help="Path to a raw frame, HDF5, or Zarr.")
    si.add_argument("--json", action="store_true")
    si.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    si.set_defaults(func=cmd_inspect)

    # diagnose
    sd = sub.add_parser("diagnose",
                         help="Build an LLM-ready diagnosis payload for a param file.")
    sd.add_argument("file", help="Parameter file to diagnose.")
    sd.add_argument("--path", required=True, choices=["ff", "nf", "pf", "ri"])
    sd.add_argument("--format", choices=["json", "prompt"], default="prompt",
                    help="json for tooling, prompt for pasting into an LLM.")
    sd.add_argument("--no-source", action="store_true",
                    help="Omit file contents from the payload.")
    sd.add_argument("--no-registry", action="store_true",
                    help="Omit registry context from the payload.")
    sd.add_argument("--no-primer", action="store_true",
                    help="Omit pipeline primer from the payload.")
    sd.set_defaults(func=cmd_diagnose)

    # rings
    sr = sub.add_parser("rings", help="Enumerate visible Bragg rings for a crystal.")
    sr.add_argument("--from", dest="param_file", default=None,
                    help="Read wavelength/Lsd/lattice/SG from this param file.")
    sr.add_argument("--wavelength", type=float, default=None, help="λ in Å.")
    sr.add_argument("--lsd", type=float, default=None, help="Sample-detector distance in µm.")
    sr.add_argument("--lattice", nargs=6, default=None,
                    metavar=("a", "b", "c", "alpha", "beta", "gamma"),
                    help="Lattice parameters (Å, deg).")
    sr.add_argument("--space-group", type=int, dest="space_group", default=None,
                    help="Space group number (1–230).")
    sr.add_argument("--rhod", type=float, default=None, help="RhoD in µm (max ring radius).")
    sr.add_argument("--nr-pixels-y", type=int, dest="nr_pixels_y", default=None,
                    help="Detector Y pixels (optional, for half-detector check).")
    sr.add_argument("--px", type=float, default=None, help="Pixel size in µm.")
    sr.add_argument("--max-rings", type=int, default=20, dest="max_rings")
    sr.add_argument("--json", action="store_true")
    sr.add_argument("--no-color", action="store_true")
    sr.set_defaults(func=cmd_rings)

    # wizard
    sw = sub.add_parser("wizard", help="Interactive parameter-file builder.")
    sw.add_argument("--path", required=True, choices=["ff", "nf", "pf", "ri"])
    sw.add_argument("--out", required=True, help="Output parameter file path.")
    sw.add_argument("--from-calibration", default=None,
                    help="Seed values from an existing MIDAS param file "
                         "(e.g. refined_MIDAS_params.txt from AutoCalibrateZarr).")
    sw.add_argument("--from-existing", default=None,
                    help="Seed values from an existing param file to edit in place.")
    sw.add_argument("--dataset", default=None,
                    help="Auto-extract file-discovery params from this dataset file.")
    sw.add_argument("--non-interactive", action="store_true",
                    help="Fail on missing values instead of prompting (for CI).")
    sw.set_defaults(func=cmd_wizard)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
