"""In-notebook paramstest builder — non-interactive API + optional widget.

Two entry points:

``build_paramstest`` — Python-level builder. Combines registry defaults +
seed values (from a midas-calibrate-v2 ``AutoCalibrationResult`` or a
v1 paramstest file) + user overrides, validates, and writes the file.
Works everywhere (nbconvert, scripts, CI).

``widget`` — interactive ipywidgets form. Renders one input per
wizard-visible FF parameter, pre-filled from seeds. User adjusts,
clicks **Build**, paramstest is written and validated inline. Requires
``ipywidgets`` (gate at import; non-notebook callers don't pay the cost).

Both share ``_seeds_from_object`` for the AutoCalibrationResult adapter
and ``_format_value_for_file`` from :mod:`midas_params.wizard` for the
on-disk serialization.
"""

from __future__ import annotations

import os
from pathlib import Path as FsPath
from typing import Any, Mapping

from .discovery import (
    DiscoveryResult,
    discover_from_calibration_file,
    merge,
)
from .registry import for_path, required_for, wizard_visible_for, by_name
from .schema import ParamSpec, Path, Severity
from .wizard import _format_value_for_file, _validate_and_report, _derive_seeds, WizardState


__all__ = ["build_paramstest", "widget", "seeds_from_calibration_result"]


# ─── Seed adapter ────────────────────────────────────────────────────────────

# Mapping from AutoCalibrationResult attribute → paramstest key name. ``BC``
# (combined Y/Z) and the distortion harmonics are special-cased below.
_ATTR_TO_PARAM = {
    "Lsd":       "Lsd",
    "tx":        "tx",
    "ty":        "ty",
    "tz":        "tz",
    "NrPixelsY": "NrPixelsY",
    "NrPixelsZ": "NrPixelsZ",
}


def seeds_from_calibration_result(result: Any) -> dict[str, Any]:
    """Extract a seed dict from a midas-calibrate-v2 ``AutoCalibrationResult``
    (or any duck-typed object exposing ``.Lsd, .BC_y, .BC_z, .tx, .ty, .tz,
    .pxY, .pxZ, .NrPixelsY, .NrPixelsZ, .distortion``).

    The result's distortion is a ``Dict[str, float]`` of v2 harmonics
    (``iso_R2/4/6, a1..a6, phi1..phi6``); each becomes a top-level key
    that matches the v1 paramstest layout the C tools expect.
    """
    seeds: dict[str, Any] = {}
    for attr, key in _ATTR_TO_PARAM.items():
        if hasattr(result, attr):
            v = getattr(result, attr)
            # Keep zeros — ``tx=0`` from a powder calibration is meaningful
            # (powder is structurally blind to tx) and must propagate.
            if v is not None:
                seeds[key] = float(v) if isinstance(v, (int, float)) else v
    # Beam center — pair (Y, Z) → "BC y z"
    if hasattr(result, "BC_y") and hasattr(result, "BC_z"):
        seeds["BC"] = [float(result.BC_y), float(result.BC_z)]
    # Pixel size — use the (Y+Z)/2 convention the rest of the codebase uses
    # (to_integrate.py:215, to_v1.py:109).
    pxY = float(getattr(result, "pxY", 0.0) or 0.0)
    pxZ = float(getattr(result, "pxZ", 0.0) or 0.0)
    if pxY > 0:
        seeds["px"] = 0.5 * (pxY + pxZ) if pxZ > 0 else pxY
    # Distortion harmonics — flatten each into a top-level key.
    dist = getattr(result, "distortion", None)
    if isinstance(dist, Mapping):
        for k, v in dist.items():
            if v is not None:
                seeds[k] = float(v)
    return seeds


# ─── Non-interactive builder ─────────────────────────────────────────────────


def _resolve_seeds(
    seed_from: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve seed_from (None / dict / path / object) into (values, source-tags)."""
    if seed_from is None:
        return {}, {}
    if isinstance(seed_from, dict):
        return dict(seed_from), {k: "user-dict" for k in seed_from}
    if isinstance(seed_from, (str, FsPath, os.PathLike)):
        dr = discover_from_calibration_file(str(seed_from))
        return dict(dr.extracted), dict(dr.source)
    # Otherwise: treat as object with attrs (AutoCalibrationResult)
    s = seeds_from_calibration_result(seed_from)
    return s, {k: "calibration-result" for k in s}


def _ensure_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(str(path).lower())


def _materialise_values(
    path: Path,
    seeds: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pick a final value for every wizard-visible param in this pipeline path.

    Precedence (highest → lowest): user overrides > seeds > spec.typical >
    spec.default. Parameters with no value at any level are skipped.
    """
    out: dict[str, Any] = {}
    overrides = dict(overrides or {})
    for spec in wizard_visible_for(path):
        if spec.name in overrides:
            out[spec.name] = overrides.pop(spec.name)
        elif spec.name in seeds:
            out[spec.name] = seeds[spec.name]
        elif spec.typical is not None:
            out[spec.name] = spec.typical
        elif spec.default is not None:
            out[spec.name] = spec.default
    # Carry through anything the user explicitly set even if it isn't in the
    # wizard-visible set (e.g. hidden_in_wizard keys, custom keys). Anything
    # left in ``overrides`` after the loop falls in here.
    for k, v in overrides.items():
        out[k] = v
    return out


def _normalise_value(spec: ParamSpec, v: Any) -> Any:
    """Coerce ``v`` into the form ``_format_value_for_file`` expects.

      * INT params: cast to int (calibration results pass floats for
        NrPixelsY/Z — we want ``NrPixelsY 2880``, not ``2880.0``).
      * *_LIST + multi_entry:  ``[a, b]`` (flat scalars) → ``[[a, b]]``
        (one entry). ``[[1, 150], [2, 150]]`` is left alone (two entries).
        This lets users pass ``BC=[y, z]`` and get ``BC y z`` on one line
        (since BC is registered as multi_entry but in practice always one
        occurrence).
    """
    from .schema import ParamType
    if v is None:
        return v
    if spec.type == ParamType.INT and isinstance(v, float):
        return int(v)
    if spec.multi_entry and spec.type.name.endswith("_LIST") and isinstance(v, list):
        if v and not isinstance(v[0], list):
            return [v]
    return v


def _render(values: dict[str, Any]) -> str:
    """Render a values dict to paramstest text using the registry serialisers."""
    lines: list[str] = []
    for k, v in values.items():
        spec = by_name().get(k)
        if spec is None:
            # Unknown key — write it through verbatim ("Key v1 v2 …").
            if isinstance(v, list):
                lines.append(f"{k} " + " ".join(str(x) for x in v))
            elif isinstance(v, bool):
                lines.append(f"{k} {1 if v else 0}")
            else:
                lines.append(f"{k} {v}")
            continue
        lines.extend(_format_value_for_file(spec, _normalise_value(spec, v)))
    return "\n".join(lines) + "\n"


def build_paramstest(
    *,
    out_path: str | os.PathLike,
    seed_from: Any = None,
    overrides: dict[str, Any] | None = None,
    path: Path | str = "ff",
    validate: bool = True,
) -> FsPath:
    """Non-interactive paramstest builder.

    Parameters
    ----------
    out_path
        Where the paramstest will be written.
    seed_from
        Source of refined-geometry / detector seed values. Can be:
          * ``None`` (no seeds, registry defaults only),
          * a ``dict`` of ``{param_name: value}``,
          * a path to an existing v1 paramstest (uses
            :func:`midas_params.discover_from_calibration_file`),
          * a midas-calibrate-v2 ``AutoCalibrationResult`` (or any object
            with the equivalent attributes — see
            :func:`seeds_from_calibration_result`).
    overrides
        Explicit user-supplied values; highest precedence.
    path
        Pipeline path — one of ``"ff"`` / ``"nf"`` / ``"pf"`` / ``"ri"`` or
        the corresponding :class:`midas_params.Path` enum.
    validate
        Run the validator on the written file (default True). Validation
        failures are printed; returns the path anyway so callers can inspect.

    Returns
    -------
    pathlib.Path
        The written paramstest path.
    """
    p = _ensure_path(path)
    seeds, _src = _resolve_seeds(seed_from)
    values = _materialise_values(p, seeds, overrides)
    out = FsPath(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(values))
    if validate:
        # Returns 0 on no errors, nonzero on errors. We don't surface the int
        # since the file is written either way; callers can re-validate via
        # the CLI / public API if they want a structured report.
        _validate_and_report(str(out), p)
    return out


# ─── Optional ipywidgets form ────────────────────────────────────────────────


def widget(
    *,
    out_path: str | os.PathLike,
    seed_from: Any = None,
    overrides: dict[str, Any] | None = None,
    path: Path | str = "ff",
    show_required_only: bool = False,
) -> Any:
    """Interactive ipywidgets form for in-notebook paramstest editing.

    Renders one labelled input per wizard-visible parameter, pre-filled
    from ``seed_from`` + ``overrides`` + registry defaults. The **Build
    paramstest** button writes ``out_path`` and prints a validation summary.

    Notes
    -----
    * The widget *requires* ``ipywidgets`` (``pip install ipywidgets``).
      Non-notebook callers should use :func:`build_paramstest` instead.
    * Multi-entry keys (e.g. ``RingThresh``) are shown as a single text
      input where the user types space-separated values per occurrence,
      one per line. Lists (e.g. ``BC``, ``LatticeParameter``) take
      space-separated tokens on a single line.
    * Validation feedback appears below the **Build** button after each
      click; the file is overwritten on every click so users can iterate.
    """
    try:
        import ipywidgets as W
        from IPython.display import display, clear_output
    except ImportError as exc:
        raise ImportError(
            "midas_params.notebook.widget requires `ipywidgets` and "
            "`ipython` — install with `pip install ipywidgets`."
        ) from exc

    p = _ensure_path(path)
    seeds, src = _resolve_seeds(seed_from)
    initial = _materialise_values(p, seeds, overrides)
    out = FsPath(out_path)

    rows: list[Any] = []
    inputs: dict[str, Any] = {}

    visible = required_for(p) if show_required_only else wizard_visible_for(p)
    # Group required first, then everything else, so users see the must-set
    # fields at the top.
    req_names = {s.name for s in required_for(p)}
    visible_sorted = sorted(visible, key=lambda s: (s.name not in req_names, s.name))

    for spec in visible_sorted:
        val = initial.get(spec.name, spec.default if spec.default is not None else "")
        if isinstance(val, list):
            text = " ".join(str(x) for x in val)
        else:
            text = "" if val is None else str(val)

        origin = src.get(spec.name, "default")
        label = spec.name + ("*" if spec.name in req_names else "")
        ti = W.Text(
            value=text,
            description=label,
            placeholder=spec.description[:60] if spec.description else "",
            style={"description_width": "200px"},
            layout=W.Layout(width="700px"),
        )
        inputs[spec.name] = ti
        rows.append(W.HBox([ti, W.Label(value=f" ({origin})",
                                         layout=W.Layout(width="180px"))]))

    out_label = W.Label(value=f"→ {out}")
    btn = W.Button(description="Build paramstest",
                   button_style="primary",
                   icon="cog")
    log = W.Output(layout=W.Layout(border="1px solid #ccc", padding="6px",
                                    margin="6px 0 0 0"))

    def _click(_b: Any) -> None:
        with log:
            clear_output(wait=True)
            # Collect values; coerce each by spec.type via _materialise.
            overrides_now: dict[str, Any] = {}
            for name, ti in inputs.items():
                spec = by_name().get(name)
                txt = ti.value.strip()
                if not txt:
                    continue
                # Best-effort coercion. Multi-entry / list types use
                # split-by-whitespace; numerics try float then int.
                if spec is None:
                    overrides_now[name] = txt
                    continue
                try:
                    from .schema import ParamType
                    if spec.type == ParamType.FLOAT:
                        overrides_now[name] = float(txt.split()[0])
                    elif spec.type == ParamType.INT:
                        overrides_now[name] = int(float(txt.split()[0]))
                    elif spec.type == ParamType.BOOL:
                        overrides_now[name] = txt.split()[0].lower() in ("1", "y", "yes", "true")
                    else:
                        # FLOAT_LIST / INT_LIST / STR / multi-entry
                        toks = txt.split()
                        if spec.multi_entry:
                            overrides_now[name] = toks if len(toks) > 1 else toks[0]
                        elif spec.type.name.endswith("_LIST"):
                            overrides_now[name] = [
                                float(x) if "." in x or "e" in x.lower() else int(x)
                                for x in toks
                            ]
                        else:
                            overrides_now[name] = txt
                except (ValueError, IndexError) as e:
                    print(f"  ⚠ {name}: could not parse {txt!r} ({e}); skipped")
                    continue
            build_paramstest(
                out_path=out, seed_from=seed_from, overrides=overrides_now,
                path=p, validate=True,
            )
            print(f"\n✓ wrote {out}  ({out.stat().st_size} bytes)")

    btn.on_click(_click)

    return W.VBox([
        W.HTML(f"<b>midas-params wizard</b> — {p.value.upper()} pipeline "
               f"({len(visible_sorted)} parameters; * = required)"),
        *rows,
        W.HBox([btn, out_label]),
        log,
    ])
