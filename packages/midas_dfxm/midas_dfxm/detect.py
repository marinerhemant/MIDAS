"""Inverse dislocation typing: identify type / Burgers (incl. sign) / position.

Phase 5 of ``implementation_plan.md`` — the novel spine.

Given multi-reflection weak-beam DFXM images of an *unknown* dislocation, recover
its slip system, Burgers vector **including sign**, character, and core position by
matching the differentiable per-dislocation Stroh forward (:mod:`midas_dfxm.dislocation`)
against the data. This targets the explicit gap in the current state of the art:

* Borgi et al. (2025) do semi-automated typing by cross-correlating one experimental
  image against ~840 forward-simulated candidates — but with **isotropic** elasticity,
  and the Burgers-vector **sign is unrecoverable** from a single image. Here the
  candidate forwards use **anisotropic (Stroh)** elasticity, and the sign is recovered
  from the **weak-beam contrast asymmetry** across a multi-reflection set (validated:
  the wrong-sign template has a large residual where the right sign is exact).

Two stages, mirroring the honest split elsewhere in the package:
1. **Discrete search** over the slip-system catalog x {edge, screw} x {+, -} — the
   part that is genuinely combinatorial.
2. **Differentiable position refinement** (gradient descent on the core position via
   ``midas_invert.fit``) — the part where autodiff helps.

Sign note: for an **edge**, +b and -b are physically distinct (compression/tension
lobes swap) and recoverable. For a **screw**, the two signs are opposite handedness;
they are distinguishable only through the (weaker) anisotropic field — reported, but
with lower confidence, and flagged in the label.

Everything is torch-differentiable and device-portable (Stroh uses complex ``eig`` ->
CPU/CUDA, not MPS).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from midas_invert.optimize import fit

from .conventions import GoniometerSetting
from .dislocation import (
    StrohDislocation,
    dislocation_deformation_field,
    fcc_slip_systems,
    stroh_dislocation,
)
from .field import DeformationField
from .forward import voxel_intensity
from .resolution import aligned_resolution
from .scan import reference_q_nom


@dataclass
class DislocationLabel:
    """Identified dislocation descriptor."""

    slip_normal: tuple
    burgers: tuple          # signed [uvw]
    character: str          # 'edge' | 'screw'
    residual: float
    core_position: tuple | None = None
    sign_confident: bool = True   # False for screw (sign = handedness, weaker)

    def __repr__(self):
        pos = "" if self.core_position is None else f", pos={tuple(round(x, 2) for x in self.core_position)}"
        return (f"DislocationLabel(plane={self.slip_normal}, b={self.burgers}, "
                f"{self.character}, residual={self.residual:.2e}{pos})")


def weak_beam_stack(
    field: DeformationField,
    reflections: Sequence[Sequence[int]],
    *,
    offset_deg: float = 0.04,
    sigma_par: float = 8e-3,
    sigma_perp: float = 8e-3,
    center: GoniometerSetting | None = None,
) -> torch.Tensor:
    """Per-voxel weak-beam intensities across reflections: ``(R, N)``.

    Each reflection is imaged at a fixed rocking offset off its aligned setting —
    the weak-beam condition whose contrast asymmetry carries the Burgers-vector sign.
    Differentiable in the field.
    """
    center = center or GoniometerSetting()
    setting = GoniometerSetting(mu=center.mu, omega=center.omega,
                                chi=center.chi + offset_deg, phi=center.phi)
    rows = []
    for hkl in reflections:
        q_nom = reference_q_nom(field, tuple(hkl), center)
        res = aligned_resolution(q_nom, sigma_par=sigma_par, sigma_perp=sigma_perp)
        rows.append(voxel_intensity(field, tuple(hkl), setting, res))
    return torch.stack(rows, dim=0)


def _normalize(stack: torch.Tensor) -> torch.Tensor:
    """Per-reflection zero-mean, unit-norm — a scale/offset-invariant image descriptor."""
    x = stack - stack.mean(dim=1, keepdim=True)
    return x / (x.norm(dim=1, keepdim=True) + 1e-30)


def match_residual(observed_norm: torch.Tensor, template_stack: torch.Tensor) -> torch.Tensor:
    """Scale-invariant residual between an observation and a template stack (scalar)."""
    return (observed_norm - _normalize(template_stack)).pow(2).mean()


def _candidate_dislocations(
    C6, systems, *, characters, signs, positions_core, core_radius_um, burgers_length_A, crystal,
):
    """Yield ``(DislocationLabel-stub, StrohDislocation)`` over the discrete grid."""
    dtype, device = C6.dtype, C6.device
    for normal, b_dir in systems:
        b_vec = torch.as_tensor(b_dir, dtype=dtype, device=device)
        n_vec = torch.as_tensor(normal, dtype=dtype, device=device)
        for character in characters:
            base_line = b_vec if character == "screw" else torch.linalg.cross(n_vec, b_vec)
            for sgn in signs:
                b_signed = tuple((sgn * b_vec).tolist())
                disl = stroh_dislocation(
                    C6, burgers=b_signed, slip_normal=normal, line=base_line,
                    burgers_length_A=burgers_length_A, core_position=positions_core,
                    core_radius_um=core_radius_um, crystal=crystal,
                )
                yield (normal, tuple(int(round(x)) for x in b_signed), character, sgn), disl


def identify_dislocation(
    observed: torch.Tensor,
    positions: torch.Tensor,
    C6: torch.Tensor,
    reflections: Sequence[Sequence[int]],
    *,
    systems=None,
    characters=("edge", "screw"),
    signs=(1, -1),
    offset_deg: float = 0.04,
    core_radius_um: float = 0.4,
    burgers_length_A: float = 2.556,
    crystal=None,
    refine_position: bool = True,
    top_k: int = 3,
    weak_beam_kw: dict | None = None,
) -> list[DislocationLabel]:
    """Identify an unknown dislocation from its multi-reflection weak-beam images.

    ``observed`` is ``(R, N)`` (R reflections, N voxels). Ranks the slip-system x
    character x sign catalog by residual against the anisotropic Stroh forward, then
    optionally refines the core position of the top candidates by gradient descent.
    Returns the ``top_k`` :class:`DislocationLabel` s, best first.
    """
    systems = systems if systems is not None else fcc_slip_systems()
    wb_kw = weak_beam_kw or {}
    obs_n = _normalize(observed)
    center_core = tuple(positions.mean(dim=0).tolist())

    scored = []
    for label, disl in _candidate_dislocations(
        C6, systems, characters=characters, signs=signs,
        positions_core=center_core, core_radius_um=core_radius_um,
        burgers_length_A=burgers_length_A, crystal=crystal,
    ):
        field = dislocation_deformation_field(positions, disl)
        stack = weak_beam_stack(field, reflections, offset_deg=offset_deg, **wb_kw)
        r = float(match_residual(obs_n, stack))
        scored.append((r, label, disl))
    scored.sort(key=lambda t: t[0])

    out = []
    for r, label, disl in scored[:top_k]:
        normal, b_signed, character, sgn = label
        pos = None
        if refine_position:
            r, pos = _refine_core(observed, positions, disl, reflections, offset_deg, wb_kw)
        out.append(DislocationLabel(
            slip_normal=normal, burgers=b_signed, character=character,
            residual=r, core_position=pos,
            sign_confident=(character == "edge"),
        ))
    return out


def _coarse_core_grid(obs_n, positions, disl, reflections, offset_deg, wb_kw, n_grid):
    """Coarse residual scan over the (x, y) bounding box; return the argmin core.

    The residual-vs-core-position landscape is multimodal (a dislocation's weak-beam
    contrast is dipolar), so a global coarse scan is needed before gradient descent —
    the same coarse->fine pattern used for the multimodal thickness fit in midas_2d.
    """
    lo = positions[:, :2].min(dim=0).values
    hi = positions[:, :2].max(dim=0).values
    xs = torch.linspace(float(lo[0]), float(hi[0]), n_grid)
    ys = torch.linspace(float(lo[1]), float(hi[1]), n_grid)
    best_r, best_c = float("inf"), None
    z = float(disl.core_position[2])
    for xx in xs.tolist():
        for yy in ys.tolist():
            core = torch.tensor([xx, yy, z], dtype=positions.dtype, device=positions.device)
            field = dislocation_deformation_field(positions, replace(disl, core_position=core))
            stack = weak_beam_stack(field, reflections, offset_deg=offset_deg, **(wb_kw or {}))
            r = float(match_residual(obs_n, stack))
            if r < best_r:
                best_r, best_c = r, core
    return best_c


def _refine_core(observed, positions, disl, reflections, offset_deg, wb_kw, n_grid=13):
    """Locate the core by coarse grid scan, then gradient-refine against the forward."""
    obs_n = _normalize(observed)
    core = _coarse_core_grid(obs_n, positions, disl, reflections, offset_deg, wb_kw, n_grid)
    core = core.detach().clone().requires_grad_(True)

    def loss_fn():
        d2 = replace(disl, core_position=core)
        field = dislocation_deformation_field(positions, d2)
        stack = weak_beam_stack(field, reflections, offset_deg=offset_deg, **(wb_kw or {}))
        return match_residual(obs_n, stack)

    fit([core], loss_fn, steps=120, lr=0.1)
    with torch.no_grad():
        final = float(loss_fn())
    return final, tuple(core.detach().tolist())
