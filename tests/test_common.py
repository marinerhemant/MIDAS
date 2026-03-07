#!/usr/bin/env python3
"""
test_common.py — Shared test diagnostics for the MIDAS benchmark suite.

Provides:
  - Pre-flight checks (binaries, Python packages, example data, disk space)
  - Binary staleness detection (source newer than compiled binary)
  - Environment fingerprint (OS, compiler, Python, package versions)
  - Structured diagnostic output on failure
  - --diagnose flag: dumps full diff + environment info to a report file
  - --save-on-fail flag: saves generated output for sharing
  - Actionable error messages per failure type

Usage from a test script:

    from test_common import (
        add_common_args, run_preflight, print_environment,
        DiagnosticReporter, check_binary_staleness,
    )

    # In parse_args():
    add_common_args(parser)

    # Before running:
    run_preflight(required_binaries=[...], required_packages=[...],
                  required_data_files=[...])
    print_environment()
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve MIDAS_HOME
# ---------------------------------------------------------------------------

def get_midas_home() -> Path:
    """Resolve MIDAS_HOME from environment or script location."""
    env = os.environ.get("MIDAS_HOME")
    if env:
        return Path(env)
    # tests/ is one level below MIDAS root
    return Path(__file__).resolve().parent.parent


MIDAS_HOME = get_midas_home()

# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser):
    """Add shared diagnostic arguments to any test's argument parser."""
    group = parser.add_argument_group("Diagnostics (shared)")
    group.add_argument(
        "--diagnose", action="store_true",
        help="On failure, dump a full diagnostic report to a file for sharing")
    group.add_argument(
        "--save-on-fail", action="store_true",
        help="Save generated output files alongside reference on failure")
    group.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight checks (binaries, packages, data)")

# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------

def _get_compiler_version() -> str:
    """Try to detect the C compiler version used for building."""
    for cc in ["gcc", "cc", "clang"]:
        try:
            result = subprocess.run(
                [cc, "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return "unknown"


def _get_package_version(pkg_name: str) -> str:
    """Get installed version of a Python package."""
    try:
        mod = __import__(pkg_name)
        return getattr(mod, "__version__", "installed (unknown version)")
    except ImportError:
        return "NOT INSTALLED"


def get_environment_info() -> dict:
    """Collect environment information as a dict."""
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "compiler": _get_compiler_version(),
        "midas_home": str(MIDAS_HOME),
        "packages": {},
    }
    for pkg in ["numpy", "zarr", "h5py", "pandas", "scipy", "blosc"]:
        info["packages"][pkg] = _get_package_version(pkg)
    return info


def print_environment():
    """Print a formatted environment fingerprint header."""
    info = get_environment_info()
    print()
    print("┌" + "─" * 68 + "┐")
    print("│  MIDAS Test Environment" + " " * 44 + "│")
    print("├" + "─" * 68 + "┤")
    print(f"│  Platform : {info['platform']:<55}│")
    print(f"│  Machine  : {info['machine']:<55}│")
    print(f"│  Python   : {info['python']:<55}│")
    print(f"│  Compiler : {info['compiler'][:55]:<55}│")
    print(f"│  MIDAS    : {str(MIDAS_HOME)[:55]:<55}│")
    pkgs = info["packages"]
    pkg_str = ", ".join(f"{k}={v}" for k, v in pkgs.items() if v != "NOT INSTALLED")
    # Wrap long package strings
    lines = textwrap.wrap(pkg_str, width=55)
    for i, line in enumerate(lines):
        label = "Packages" if i == 0 else "        "
        print(f"│  {label} : {line:<55}│")
    print("└" + "─" * 68 + "┘")
    print()


# ---------------------------------------------------------------------------
# Binary staleness check
# ---------------------------------------------------------------------------

def check_binary_staleness(binary_path: Path, source_paths: list = None):
    """Check if a compiled binary is older than its source files.

    Args:
        binary_path: Path to the compiled binary
        source_paths: Optional list of source file paths to check against.
                      If None, attempts to infer from binary name.
    """
    if not binary_path.exists():
        return  # Will be caught by preflight

    binary_mtime = binary_path.stat().st_mtime

    if source_paths is None:
        # Try to infer source from binary name
        name = binary_path.stem
        src_dirs = [
            MIDAS_HOME / "FF_HEDM" / "src",
            MIDAS_HOME / "NF_HEDM" / "src",
            MIDAS_HOME / "TOMO" / "src",
        ]
        source_paths = []
        for src_dir in src_dirs:
            for ext in [".c", ".cu"]:
                candidate = src_dir / (name + ext)
                if candidate.exists():
                    source_paths.append(candidate)

    stale = []
    for src in source_paths:
        src = Path(src)
        if src.exists() and src.stat().st_mtime > binary_mtime:
            stale.append(src)

    if stale:
        print(f"  ⚠️  Binary may be stale: {binary_path.name}")
        for s in stale:
            print(f"      Source {s.name} is newer than compiled binary")
        print(f"      Recompile with: cd {MIDAS_HOME}/build && cmake --build . --target {binary_path.stem}")
        print()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_binaries(binaries: list) -> list:
    """Check that required binaries exist and are executable."""
    issues = []
    for bin_path in binaries:
        p = Path(bin_path)
        if not p.is_absolute():
            # Search in standard MIDAS bin directories
            candidates = [
                MIDAS_HOME / "FF_HEDM" / "bin" / bin_path,
                MIDAS_HOME / "NF_HEDM" / "bin" / bin_path,
                MIDAS_HOME / "TOMO" / "bin" / bin_path,
                MIDAS_HOME / "build" / "bin" / bin_path,
            ]
            found = False
            for c in candidates:
                if c.exists():
                    p = c
                    found = True
                    check_binary_staleness(c)
                    break
            if not found:
                issues.append({
                    "type": "binary_missing",
                    "name": bin_path,
                    "message": f"Binary '{bin_path}' not found",
                    "action": f"Recompile: cd {MIDAS_HOME}/build && cmake --build . --target {bin_path}",
                })
                continue
        elif not p.exists():
            issues.append({
                "type": "binary_missing",
                "name": str(p),
                "message": f"Binary not found: {p}",
                "action": f"Recompile: cd {MIDAS_HOME}/build && cmake --build . --target {p.stem}",
            })
            continue

        if not os.access(str(p), os.X_OK):
            issues.append({
                "type": "binary_not_executable",
                "name": str(p),
                "message": f"Binary not executable: {p}",
                "action": f"chmod +x {p}",
            })
    return issues


def _check_packages(packages: list) -> list:
    """Check that required Python packages are importable."""
    issues = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            issues.append({
                "type": "package_missing",
                "name": pkg,
                "message": f"Python package '{pkg}' is not installed",
                "action": f"Install with: pip install {pkg}",
            })
    return issues


def _check_data_files(files: list) -> list:
    """Check that required data files exist and are non-empty."""
    issues = []
    for f in files:
        p = Path(f)
        if not p.exists():
            issues.append({
                "type": "data_missing",
                "name": str(p),
                "message": f"Required data file not found: {p}",
                "action": "Ensure the MIDAS example data is intact. Re-clone or restore from backup.",
            })
        elif p.stat().st_size == 0:
            issues.append({
                "type": "data_empty",
                "name": str(p),
                "message": f"Data file is empty (0 bytes): {p}",
                "action": "The file may be corrupted. Re-clone or restore from backup.",
            })
    return issues


def _check_disk_space(work_dir: str = None, min_mb: int = 500) -> list:
    """Check that there is sufficient disk space."""
    issues = []
    check_path = work_dir or str(MIDAS_HOME)
    try:
        usage = shutil.disk_usage(check_path)
        free_mb = usage.free / (1024 * 1024)
        if free_mb < min_mb:
            issues.append({
                "type": "disk_space",
                "name": check_path,
                "message": f"Low disk space: {free_mb:.0f} MB free (need ≥{min_mb} MB)",
                "action": "Free up disk space before running tests.",
            })
    except OSError:
        pass  # Can't check — not a critical issue
    return issues


def run_preflight(required_binaries: list = None,
                  required_packages: list = None,
                  required_data_files: list = None,
                  min_disk_mb: int = 500,
                  work_dir: str = None):
    """Run all pre-flight checks. Prints results and exits on critical failures.

    Args:
        required_binaries: List of binary names (e.g., ["PeaksFittingOMPZarrRefactor"])
        required_packages: List of Python package names (e.g., ["numpy", "h5py"])
        required_data_files: List of absolute paths to required data files
        min_disk_mb: Minimum free disk space in MB
        work_dir: Directory to check for disk space
    """
    print("Pre-flight checks...")
    all_issues = []

    if required_binaries:
        all_issues.extend(_check_binaries(required_binaries))
    if required_packages:
        all_issues.extend(_check_packages(required_packages))
    if required_data_files:
        all_issues.extend(_check_data_files(required_data_files))

    all_issues.extend(_check_disk_space(work_dir, min_disk_mb))

    if not all_issues:
        print("  ✅ All pre-flight checks passed\n")
        return

    # Categorize
    errors = [i for i in all_issues if i["type"] in ("binary_missing", "package_missing", "data_missing")]
    warnings = [i for i in all_issues if i not in errors]

    for issue in warnings:
        print(f"  ⚠️  {issue['message']}")
        print(f"      → {issue['action']}")

    if errors:
        print()
        for issue in errors:
            print(f"  ❌ {issue['message']}")
            print(f"      → {issue['action']}")
        print(f"\n  {len(errors)} critical issue(s) found. Fix them before running tests.")
        sys.exit(1)

    print()


# ---------------------------------------------------------------------------
# Diagnostic reporter
# ---------------------------------------------------------------------------

class DiagnosticReporter:
    """Collects test diagnostic data and produces actionable failure reports.

    Usage:
        reporter = DiagnosticReporter(test_name="test_ff_hedm", args=args)

        # Record a comparison result:
        reporter.record("peaks/summary/data", passed=True, max_diff=0.0)
        reporter.record("grains/summary", passed=False, max_diff=1.5e-3,
                        ref_shape=(3,20), new_shape=(3,20), n_mismatch=5)

        # At the end:
        reporter.summary()
        if reporter.has_failures:
            reporter.save_report()
    """

    def __init__(self, test_name: str, args=None):
        self.test_name = test_name
        self.args = args
        self.results = []
        self.diagnose = getattr(args, "diagnose", False) if args else False
        self.save_on_fail = getattr(args, "save_on_fail", False) if args else False
        self._env_info = None

    @property
    def has_failures(self) -> bool:
        return any(not r["passed"] for r in self.results)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if not r["passed"])

    def record(self, dataset: str, passed: bool, **kwargs):
        """Record a single comparison result.

        kwargs may include: max_diff, ref_shape, new_shape, n_mismatch,
        n_total, ref_sample, new_sample, failure_type
        """
        entry = {"dataset": dataset, "passed": passed}
        entry.update(kwargs)
        self.results.append(entry)

    def get_actionable_message(self, result: dict) -> str:
        """Generate a specific, actionable message for a failure."""
        ft = result.get("failure_type", "")
        ds = result.get("dataset", "")

        if ft == "shape_mismatch":
            ref_s = result.get("ref_shape", "?")
            new_s = result.get("new_shape", "?")
            return (f"Shape mismatch on '{ds}': ref={ref_s}, got={new_s}. "
                    "The number of peaks/frames differs — check if the parameter "
                    "file was modified or if a pipeline stage crashed early.")

        if ft == "all_zeros":
            return (f"All zeros in '{ds}'. The pipeline stage likely crashed — "
                    "check stderr output above for error messages.")

        if ft == "missing_in_output":
            return (f"Dataset '{ds}' missing from output. A pipeline stage may "
                    "have been skipped or crashed. Check the log for errors.")

        if ft == "missing_in_reference":
            return (f"Dataset '{ds}' not in reference file. The reference may "
                    "need to be regenerated for this MIDAS version.")

        md = result.get("max_diff", 0)
        nm = result.get("n_mismatch", 0)
        nt = result.get("n_total", 1)

        if nm > 0 and nm < nt * 0.01:
            return (f"Minor floating-point differences in '{ds}': "
                    f"{nm}/{nt} values differ (max_diff={md:.2e}). "
                    "Likely platform-specific numerical noise. "
                    "Try: python tests/{self.test_name}.py --atol 1e-4")

        if md > 0.1:
            return (f"Large differences in '{ds}': max_diff={md:.2e}. "
                    "This suggests a real regression. Check if binaries "
                    f"are up-to-date: cd {MIDAS_HOME}/build && cmake --build .")

        return (f"Mismatch in '{ds}': {nm}/{nt} values differ, "
                f"max_diff={md:.2e}.")

    def print_failure_details(self, result: dict, max_samples: int = 5):
        """Print detailed comparison for a single failed dataset."""
        ds = result["dataset"]
        msg = self.get_actionable_message(result)
        print(f"\n  💡 {msg}")

        # Print sample values if available
        ref_sample = result.get("ref_sample")
        new_sample = result.get("new_sample")
        if ref_sample is not None and new_sample is not None:
            n = min(max_samples, len(ref_sample))
            print(f"\n      First {n} mismatched values:")
            print(f"      {'Index':<8} {'Reference':<16} {'Got':<16} {'Diff':<16}")
            indices = result.get("mismatch_indices", range(n))
            for i in range(n):
                idx = indices[i] if i < len(indices) else i
                r_val = ref_sample[i]
                n_val = new_sample[i]
                diff = abs(r_val - n_val)
                print(f"      {idx:<8} {r_val:<16.8g} {n_val:<16.8g} {diff:<16.2e}")

        # Print histogram of differences if available
        diff_histogram = result.get("diff_histogram")
        if diff_histogram:
            print(f"\n      Distribution of differences:")
            for label, count, pct in diff_histogram:
                bar = "█" * int(pct / 2)
                print(f"      {label:>12}: {count:>6} ({pct:5.1f}%) {bar}")

    def summary(self):
        """Print the summary table with pass/fail counts."""
        n_p = self.n_pass
        n_f = self.n_fail
        total = n_p + n_f

        print(f"\n{'=' * 70}")
        print(f"  Summary: {n_p} PASS, {n_f} FAIL (out of {total})")
        print(f"{'=' * 70}")

        if not self.has_failures:
            print(f"\n✅ All checks passed.\n")
            return

        # Print failures with actionable messages
        print(f"\n❌ {n_f} check(s) failed:\n")
        for r in self.results:
            if not r["passed"]:
                self.print_failure_details(r)

        print(f"\n{'─' * 70}")
        print("  To investigate further:")
        print(f"    python tests/{self.test_name}.py --diagnose")
        print(f"    python tests/{self.test_name}.py --save-on-fail")
        print(f"\n  Send the generated diagnostic report to hsharma@anl.gov")
        print(f"  for assistance with debugging.")
        print(f"{'─' * 70}\n")

    def save_report(self, output_dir: str = None):
        """Save a full diagnostic report to a JSON file."""
        if not self.diagnose and not self.save_on_fail:
            return None

        if output_dir is None:
            output_dir = str(MIDAS_HOME / "tests")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"diagnostic_{self.test_name}_{timestamp}.json"
        report_path = Path(output_dir) / report_name

        if self._env_info is None:
            self._env_info = get_environment_info()

        report = {
            "test_name": self.test_name,
            "timestamp": self._env_info["timestamp"],
            "environment": self._env_info,
            "results": [],
            "overall": {
                "n_pass": self.n_pass,
                "n_fail": self.n_fail,
                "passed": not self.has_failures,
            },
        }

        for r in self.results:
            entry = dict(r)
            # Convert numpy types to native Python for JSON serialization
            for key in list(entry.keys()):
                val = entry[key]
                if hasattr(val, "item"):
                    entry[key] = val.item()
                elif isinstance(val, (list, tuple)):
                    entry[key] = [v.item() if hasattr(v, "item") else v for v in val]
            # Remove numpy arrays from report (not JSON serializable)
            for key in ["ref_sample", "new_sample", "mismatch_indices"]:
                if key in entry:
                    val = entry[key]
                    if hasattr(val, "tolist"):
                        entry[key] = val.tolist()
            report["results"].append(entry)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n  📋 Diagnostic report saved: {report_path}")
        print(f"     Send this file to hsharma@anl.gov for investigation.\n")
        return report_path

    def save_output_on_fail(self, generated_path: str, reference_path: str = None):
        """Copy generated output alongside reference for comparison."""
        if not self.save_on_fail or not self.has_failures:
            return

        gen_p = Path(generated_path)
        if not gen_p.exists():
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{gen_p.stem}_FAILED_{timestamp}{gen_p.suffix}"
        save_path = gen_p.parent / save_name

        shutil.copy2(str(gen_p), str(save_path))
        print(f"\n  💾 Saved failed output for inspection:")
        print(f"     Generated: {save_path}")
        if reference_path:
            print(f"     Reference: {reference_path}")


# ---------------------------------------------------------------------------
# Utility: build diff histogram from two numpy arrays
# ---------------------------------------------------------------------------

def build_diff_histogram(ref_data, new_data):
    """Build a histogram of absolute differences between two arrays.

    Returns list of (label, count, percentage) tuples.
    """
    import numpy as np

    diff = np.abs(ref_data.ravel().astype(float) - new_data.ravel().astype(float))
    total = len(diff)
    if total == 0:
        return []

    bins = [
        ("exact match", np.sum(diff == 0)),
        ("< 1e-10",     np.sum((diff > 0) & (diff < 1e-10))),
        ("1e-10 – 1e-6", np.sum((diff >= 1e-10) & (diff < 1e-6))),
        ("1e-6 – 1e-3",  np.sum((diff >= 1e-6) & (diff < 1e-3))),
        ("1e-3 – 1e-1",  np.sum((diff >= 1e-3) & (diff < 1e-1))),
        ("≥ 0.1",        np.sum(diff >= 1e-1)),
    ]

    return [(label, int(count), count / total * 100) for label, count in bins]


def get_mismatch_samples(ref_data, new_data, atol=1e-6, rtol=1e-6, max_samples=5):
    """Extract the first few mismatched values for display.

    Returns (ref_samples, new_samples, indices) as lists.
    """
    import numpy as np

    mask = ~np.isclose(ref_data, new_data, atol=atol, rtol=rtol, equal_nan=True)
    indices = np.where(mask.ravel())[0]

    n = min(max_samples, len(indices))
    if n == 0:
        return None, None, []

    ref_flat = ref_data.ravel()
    new_flat = new_data.ravel()
    return (
        ref_flat[indices[:n]].tolist(),
        new_flat[indices[:n]].tolist(),
        indices[:n].tolist(),
    )
