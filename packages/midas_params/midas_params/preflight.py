"""One preflight every MIDAS CLI can run before it does any work.

The parameter-file validator (:mod:`midas_params.validator`) checks what is
*inside* a parameter file. This module checks everything *around* it: which
build of the tool is actually running, whether the paths on the command line
exist, and whether a mistyped flag has an obvious neighbour. Those are the
failures users report as "your code is broken", and none of them are caught by
reading the parameter file.

Three entry points, in ascending order of how much they change a caller:

``MidasArgumentParser``
    Drop-in for :class:`argparse.ArgumentParser`. Same behaviour, except that
    argument errors carry the running tool's version and environment, and a
    did-you-mean for near-miss flags. One-word change at the call site.

``check_environment``
    Which executable is running, from which prefix, at which version. Answers
    "why does my colleague's copy behave differently".

``preflight``
    Runs the environment and path checks, and (when given a parameter file)
    delegates to the existing validator. Warn-only by default.

Why the argparse subclass matters, concretely. A user ran a command that was
correct in every respect except that his shell was in the wrong conda
environment, whose package predated the option he was using. argparse told him::

    error: argument --mode: invalid choice: 'ff'
        (choose from single, multi, bayesian, nn, joint, sensitivity)

which is accurate, unhelpful, and reads as a bug in the tool. The same failure
through this parser also tells him which build he is running and where it came
from, which is the actual answer.
"""
from __future__ import annotations

import argparse
import difflib
import importlib.metadata as _md
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

#: Set to any non-empty value to skip preflight entirely.
ENV_DISABLE = "MIDAS_NO_PREFLIGHT"


# ─── environment ─────────────────────────────────────────────────────────────
@dataclass
class EnvReport:
    tool: str
    executable: Optional[str]           # the resolved console script on PATH
    prefix: str                         # sys.prefix, ie which env is running
    package: Optional[str] = None
    version: Optional[str] = None
    module_file: Optional[str] = None
    shadowed_by: Optional[str] = None   # a *different* copy earlier on PATH
    notes: list[str] = field(default_factory=list)

    def one_line(self) -> str:
        v = f"{self.package} {self.version}" if self.version else "version unknown"
        return f"{self.tool}: {v}, from {self.prefix}"


def check_environment(tool: str, package: Optional[str] = None) -> EnvReport:
    """Describe the build of ``tool`` that is actually running.

    ``package`` is the distribution name (``midas-calibrate-v2``); when given,
    its installed version is reported. Nothing here fails a run — the point is
    to make "which copy am I using" answerable without a support round-trip.
    """
    rep = EnvReport(tool=tool, executable=shutil.which(tool), prefix=sys.prefix)
    if package:
        rep.package = package
        try:
            rep.version = _md.version(package)
        except _md.PackageNotFoundError:
            rep.notes.append(
                f"{package} is not installed in {sys.prefix}; the running "
                "script may come from somewhere else entirely.")

    # A console script earlier on PATH than the interpreter's own bin is the
    # classic "conda env still active" trap: PATH was extended, but the old
    # env's bin still wins.
    if rep.executable:
        own_bin = str(Path(sys.executable).parent)
        found_bin = str(Path(rep.executable).parent)
        if found_bin != own_bin:
            rep.shadowed_by = rep.executable
            rep.notes.append(
                f"'{tool}' on PATH resolves to {rep.executable}, but this "
                f"process is running from {own_bin}. If a conda env is still "
                "active its bin shadows the one you added; 'conda deactivate' "
                "first.")
    return rep


def check_device(requested: Optional[str]) -> list[str]:
    """Warn when ``--device cuda`` cannot actually use a GPU.

    Torch falls back to CPU silently, so a run can be ten times slower than
    expected with nothing in the log to say why. Only imports torch when CUDA
    was actually asked for.
    """
    if not requested or str(requested).lower() != "cuda":
        return []
    try:
        import torch
    except Exception:
        return ["--device cuda was requested but torch is not importable."]
    if torch.cuda.is_available():
        return []
    return [f"--device cuda was requested but torch {torch.__version__} reports "
            "no usable CUDA device, so this will run on CPU. Usually the torch "
            "build and the driver disagree; check `nvidia-smi` against the "
            "cu-version in that torch."]


# ─── paths ───────────────────────────────────────────────────────────────────
def check_paths(paths: dict[str, object], *, must_exist: bool = True) -> list[str]:
    """Check path-valued arguments before a long run starts.

    ``paths`` maps the option name to its value; ``None`` is skipped so callers
    can pass optional arguments unconditionally. Distinguishes "missing" from
    "there but unreadable", because the second is a permissions problem and
    sends you somewhere completely different.
    """
    problems: list[str] = []
    for name, value in paths.items():
        if value in (None, ""):
            continue
        p = Path(str(value))
        if not p.exists():
            if must_exist:
                hint = ""
                parent = p.parent
                if parent.exists() and p.name:
                    near = difflib.get_close_matches(
                        p.name, [c.name for c in parent.iterdir()], n=1, cutoff=0.6)
                    if near:
                        hint = f" Did you mean '{near[0]}'?"
                elif not parent.exists():
                    hint = f" Its directory {parent} does not exist either."
                problems.append(f"{name}: {p} does not exist.{hint}")
            continue
        if not os.access(p, os.R_OK):
            problems.append(
                f"{name}: {p} exists but is not readable by you. If it is on a "
                "shared data tree it may be behind that experiment's group.")
            continue
        if p.is_file() and p.stat().st_size == 0:
            problems.append(f"{name}: {p} is empty (0 bytes).")
    return problems


def check_hdf5_group(path: object, group: Optional[str], *, label: str = "--image-group") -> list[str]:
    """Check that a named HDF5 dataset exists, and name the alternatives.

    Pointing at a group that is not there, or at a metadata group instead of
    the data, is a common and confusing failure.
    """
    if not path or not group:
        return []
    try:
        import h5py
    except Exception:
        return []
    try:
        with h5py.File(str(path), "r") as f:
            if group in f:
                return []
            top = list(f.keys())
            return [f"{label}: '{group}' is not in {path}. Top-level keys are: "
                    f"{', '.join(top)}."]
    except OSError:
        return []          # unreadable/absent is check_paths' job, not ours


# ─── argparse ────────────────────────────────────────────────────────────────
class MidasArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose errors say which build produced them.

    Pass ``package=`` so version and environment can be reported. Everything
    else behaves exactly like the stock parser, so adopting it is a one-word
    change and cannot alter how a correct command line is interpreted.
    """

    def __init__(self, *args, package: Optional[str] = None, **kwargs):
        self._midas_package = package
        super().__init__(*args, **kwargs)

    def add_subparsers(self, **kwargs):
        """Give subcommands the same treatment as the top-level parser.

        argparse builds subparsers from ``parser_class``, which defaults to the
        parent's type — so they would be ``MidasArgumentParser`` already, but
        with ``package=None`` and therefore no version line. Since the version
        is the whole point on a subcommand error (``grain-tx`` on a stale build
        looks identical to ``grain-tx`` on a current one), bind it here.
        """
        pkg = self._midas_package
        if "parser_class" not in kwargs:
            cls = type(self)

            def _factory(*a, **kw):
                kw.setdefault("package", pkg)
                return cls(*a, **kw)

            kwargs["parser_class"] = _factory
        return super().add_subparsers(**kwargs)

    # -- suggestions ---------------------------------------------------------
    def _known_option_strings(self) -> list[str]:
        out: list[str] = []
        for action in self._actions:
            out.extend(action.option_strings)
        return out

    def _suggest(self, message: str) -> Optional[str]:
        """Turn argparse's terse complaint into a next step."""
        # unrecognized flag -> nearest known flag
        if "unrecognized arguments" in message:
            bad = [t for t in message.split(":", 1)[-1].split()
                   if t.startswith("-")]
            for b in bad:
                near = difflib.get_close_matches(
                    b, self._known_option_strings(), n=1, cutoff=0.6)
                if near:
                    return f"Did you mean {near[0]}?"
            return None
        # a flag that is really a positional -> the commonest confusion
        if "unrecognized arguments" not in message and "required" in message:
            required_positionals = [
                a.dest for a in self._actions
                if not a.option_strings and a.required is not False]
            if required_positionals:
                return (f"{', '.join(required_positionals)} "
                        f"{'is a positional argument' if len(required_positionals) == 1 else 'are positional arguments'}"
                        ": give the value on its own, with no --flag in front.")
        # invalid choice -> the version is usually the real story
        if "invalid choice" in message and self._midas_package:
            try:
                v = _md.version(self._midas_package)
            except _md.PackageNotFoundError:
                v = "unknown"
            return (f"If you expected that value to be accepted, check the "
                    f"build: {self._midas_package} {v} from {sys.prefix}. A "
                    "newer version may support it.")
        return None

    def error(self, message):                       # noqa: D102 (argparse API)
        suggestion = self._suggest(message)
        self.print_usage(sys.stderr)
        prog = self.prog
        sys.stderr.write(f"{prog}: error: {message}\n")
        if suggestion:
            sys.stderr.write(f"{prog}: hint: {suggestion}\n")
        if self._midas_package:
            try:
                v = _md.version(self._midas_package)
            except _md.PackageNotFoundError:
                v = "not installed"
            sys.stderr.write(
                f"{prog}: running {self._midas_package} {v} from {sys.prefix}\n")
        self.exit(2)


# ─── the one call ────────────────────────────────────────────────────────────
@dataclass
class PreflightResult:
    ok: bool
    env: Optional[EnvReport] = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def preflight(
    *,
    tool: str,
    package: Optional[str] = None,
    paths: Optional[dict[str, object]] = None,
    param_file: Optional[object] = None,
    pipeline: Optional[str] = None,
    device: Optional[str] = None,
    strict: bool = False,
    skip: bool = False,
    verbose: bool = True,
    stream=None,
) -> PreflightResult:
    """Check the run before it starts. Warn-only unless ``strict``.

    Deliberately cheap: no data is opened beyond an HDF5 key listing, so this
    stays in the tens of milliseconds and nobody has a reason to switch it off.

    Set ``MIDAS_NO_PREFLIGHT=1`` (or pass ``skip=True``) to bypass entirely.
    """
    stream = stream or sys.stderr
    if skip or os.environ.get(ENV_DISABLE):
        return PreflightResult(ok=True)

    env = check_environment(tool, package)
    problems = check_paths(paths or {})
    warnings = list(env.notes) + check_device(device)

    if param_file:
        problems.extend(check_paths({"parameter file": param_file}))
        if pipeline and not problems:
            try:
                from midas_params.hook import preflight_validate
                ok = preflight_validate(param_file=str(param_file),
                                        pipeline=pipeline, skip=False,
                                        strict=False)
                if not ok:
                    warnings.append(
                        "the parameter file has validation findings above; "
                        "run `midas-params validate <file> --path "
                        f"{pipeline}` for the full report.")
            except Exception as exc:                # never block on the checker
                warnings.append(f"parameter validation unavailable ({exc}).")

    if verbose and (problems or warnings):
        print(f"[preflight] {env.one_line()}", file=stream)
        for w in warnings:
            print(f"[preflight] note: {w}", file=stream)
        for p in problems:
            print(f"[preflight] problem: {p}", file=stream)

    ok = not problems
    if strict and not ok:
        print("[preflight] stopping before doing any work. Set "
              f"{ENV_DISABLE}=1 to override.", file=stream)
        raise SystemExit(2)
    return PreflightResult(ok=ok, env=env, problems=problems, warnings=warnings)
