"""Joint multi-mounting FF indexing.

A sample measured in several *mountings* (the same sample re-mounted in known orientations, e.g. the
three orthogonal mountings of a cubic multi-anvil cell) gives one ordinary MIDAS FF layer directory per
mounting, each with its own spots, bins and omega windows. Indexing each mounting on its own wastes
information when the per-mounting omega coverage is small: every grain predicts few spots per mounting,
so real grains fall below the completeness cut and chance solutions can pass it.

:class:`JointIndexer` indexes all mountings together, in one sample frame:

1. Seeds come from every mounting. For a seed of mounting ``k`` the usual per-seed candidate set
   (orientation x position, exactly :func:`midas_index.pipeline.process_seed` steps 1-2) is built in
   ``k``'s frame and mapped to the sample frame.
2. Each candidate is mapped into every mounting ``j`` (``O_j = M_j O``, ``p_j = M_j p + s_j``),
   forward-predicted with ``j``'s geometry and omega windows, and matched against ``j``'s spots.
   The score is the JOINT completeness ``sum_j matches_j / sum_j predicted_j``.
3. Accepted seeds are clustered by ORIENTATION only. FF matching is position-insensitive (a grain
   shifted by 1 mm matches the same spots), so an indexing position is only as good as the
   ``StepsizePos`` grid and cannot separate grains.
4. Each cluster representative is refined (orientation + position, 6 DOF) by Gauss-Newton on the
   detector y/z and omega residuals of its matched spots in ALL mountings, with the spot assignment
   held fixed within each Gauss-Newton pass and re-done between passes. A refinement is kept only if
   it lowers the y/z residual RMS and stays inside ``Rsample``.
5. Refined grains are merged (misorientation and distance).

Mounting convention: ``M_k`` maps SAMPLE-frame vectors into mounting ``k``'s frame, and ``s_k`` is a
position offset (registration) in mounting ``k``'s frame. Orientation matrices follow the MIDAS
Grains.csv convention (crystal -> sample).

Validated on simulated 100 um garnet cubes in a 23-degree-window multi-anvil geometry (three
mountings), see the MIDAS nfdev_jul26 HPcat_P2 analysis.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .compute import matching, orientation_grid, position_grid
from .compute import seeds as seeds_module
from .indexer import Indexer
from .pipeline import IndexerContext, _spot_to_gv

__all__ = ["Mounting", "JointGrain", "JointResult", "JointIndexer", "load_mount_matrix", "main"]


@dataclass
class Mounting:
    """One mounting: a MIDAS FF layer directory plus its sample->mounting rotation and offset."""

    layer_dir: str | os.PathLike
    rotation: np.ndarray                      # (3, 3), sample frame -> mounting frame
    shift_um: np.ndarray = field(default_factory=lambda: np.zeros(3))
    param_file: str = "paramstest.txt"        # relative to layer_dir


@dataclass
class JointGrain:
    orient_mat: np.ndarray        # (3, 3) sample frame
    position: np.ndarray          # (3,) um, sample frame
    completeness: float           # joint matches / joint predicted
    n_matches: int
    n_predicted: int
    matches_per_mounting: np.ndarray
    seed_mounting: int
    seed_spot_id: int


@dataclass
class JointResult:
    grains: list[JointGrain]
    n_seeds: int
    n_seeds_accepted: int

    def orient_mats(self) -> np.ndarray:
        return np.array([g.orient_mat.reshape(9) for g in self.grains]).reshape(-1, 9)

    def positions(self) -> np.ndarray:
        return np.array([g.position for g in self.grains]).reshape(-1, 3)

    def save_npz(self, path: str | os.PathLike) -> None:
        np.savez(
            path, om=self.orient_mats(), pos=self.positions(),
            frac=np.array([g.completeness for g in self.grains]),
            m=np.array([g.n_matches for g in self.grains]),
            v=np.array([g.n_predicted for g in self.grains]),
            per=np.array([g.matches_per_mounting for g in self.grains]).reshape(len(self.grains), -1),
            seed_mount=np.array([g.seed_mounting for g in self.grains]),
            seed_spot=np.array([g.seed_spot_id for g in self.grains]),
        )

    def save_csv(self, path: str | os.PathLike) -> None:
        """Grains.csv-like table in the SAMPLE frame (one row per grain)."""
        n_m = len(self.grains[0].matches_per_mounting) if self.grains else 0
        head = (["GrainID"] + [f"O{r}{c}" for r in (1, 2, 3) for c in (1, 2, 3)] + ["X", "Y", "Z",
                "Completeness", "NMatches", "NPredicted"] + [f"NMatches_M{j}" for j in range(n_m)]
                + ["SeedMounting", "SeedSpotID"])
        with open(path, "w") as f:
            f.write("%" + "\t".join(head) + "\n")
            for i, g in enumerate(self.grains, 1):
                row = ([i] + list(g.orient_mat.reshape(9)) + list(g.position)
                       + [g.completeness, g.n_matches, g.n_predicted]
                       + list(g.matches_per_mounting) + [g.seed_mounting, g.seed_spot_id])
                f.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + "\n")


def load_mount_matrix(path: str | os.PathLike) -> np.ndarray:
    """3x3 rotation from a text file, a .npy, or a .npz holding ``mount``."""
    p = Path(path)
    if p.suffix == ".npz":
        return np.asarray(np.load(p)["mount"], dtype=np.float64)
    if p.suffix == ".npy":
        return np.asarray(np.load(p), dtype=np.float64).reshape(3, 3)
    return np.loadtxt(p, dtype=np.float64).reshape(3, 3)


def _rotvec(w: torch.Tensor) -> torch.Tensor:
    th = torch.linalg.norm(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    if float(th) < 1e-12:
        return eye
    k = w / th
    K = torch.zeros(3, 3, device=w.device, dtype=w.dtype)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _miso_deg(a9: np.ndarray, b9: np.ndarray, sg: int) -> float:
    from midas_stress.orientation import misorientation_om_batch
    return math.degrees(float(misorientation_om_batch(a9.reshape(1, 9), b9.reshape(1, 9), sg)[0]))


class JointIndexer:
    """Joint indexer over several mountings (see module docstring)."""

    def __init__(
        self,
        mountings: list[Mounting],
        *,
        device: str | torch.device | None = None,
        min_completeness: float | None = None,
        min_nr_spots: int | None = None,
        cluster_deg: float = 0.5,
        merge_deg: float = 0.5,
        merge_um: float = 30.0,
        chunk: int = 200_000,
        contexts: list[IndexerContext] | None = None,
        seeds: list[tuple[int, int]] | None = None,
    ) -> None:
        if not mountings and contexts is None:
            raise ValueError("at least one mounting is required")
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.cluster_deg, self.merge_deg, self.merge_um, self.chunk = cluster_deg, merge_deg, merge_um, chunk
        if contexts is None:
            contexts, seeds = self._load(mountings)
        self.ctxs = contexts
        self.seeds = list(seeds or [])
        self.M = [torch.tensor(np.asarray(m.rotation, dtype=np.float64), device=self.device, dtype=self.dtype)
                  for m in mountings]
        self.S = [torch.tensor(np.asarray(m.shift_um, dtype=np.float64), device=self.device, dtype=self.dtype)
                  for m in mountings]
        p0 = self.ctxs[0].params
        self.min_frac = float(min_completeness if min_completeness is not None
                              else getattr(p0, "MinMatchesToAcceptFrac", 0.5))
        self.min_n = int(min_nr_spots if min_nr_spots is not None else getattr(p0, "MinNrSpots", 3))
        self.space_group = int(p0.SpaceGroup)
        self.rsample = float(p0.Rsample)

    # ---------------------------------------------------------------- loading
    def _load(self, mountings):
        ctxs, seeds = [], []
        for k, m in enumerate(mountings):
            cwd = os.getcwd()
            os.chdir(m.layer_dir)          # hkls.csv is read relative to the layer directory
            try:
                ind = Indexer.from_param_file(m.param_file, device=self.device, dtype=self.dtype)
                ind.load_observations(cwd=".")
            finally:
                os.chdir(cwd)
            o = ind._observations
            ctxs.append(IndexerContext(params=ind.params, hkls_real=o["hkls_real"], hkls_int=o["hkls_int"],
                                       obs=o["spots"], bin_data=o["bin_data"], bin_ndata=o["bin_ndata"],
                                       device=self.device, dtype=self.dtype))
            ids = o["spot_ids"][:, 0] if o["spot_ids"].ndim == 2 else o["spot_ids"]
            seeds += [(k, int(s)) for s in ids]
        return ctxs, seeds

    # ---------------------------------------------------------------- frames
    def to_mount(self, R_s: torch.Tensor, p_s: torch.Tensor, j: int):
        return self.M[j] @ R_s, p_s @ self.M[j].T + self.S[j]

    def to_sample(self, R_k: torch.Tensor, p_k: torch.Tensor, k: int):
        return self.M[k].T @ R_k, (p_k - self.S[k]) @ self.M[k]

    # ---------------------------------------------------------------- scoring
    def score(self, R_s: torch.Tensor, p_s: torch.Tensor, ref_rad: float, want_rows: bool = False):
        """Joint match of sample-frame candidates ``(N,3,3), (N,3)``.

        Returns total matches (N,), total predicted (N,), per-mounting matches (N, n_mountings) and,
        if ``want_rows``, the per-mounting ``(theor, valid, MatchResult)``."""
        dev, dt = self.device, self.dtype
        m_tot = torch.zeros(R_s.shape[0], device=dev, dtype=dt)
        v_tot = torch.zeros_like(m_tot)
        per, rows = [], []
        for j, c in enumerate(self.ctxs):
            Rj, pj = self.to_mount(R_s, p_s, j)
            theor, valid = c.adapter.simulate(Rj, pj, lattice=None)
            res = matching.compare_spots(
                theor=theor, valid=valid, obs=c.obs, bin_data=c.bin_data, bin_ndata=c.bin_ndata,
                ref_rad=torch.full((R_s.shape[0],), ref_rad, device=dev, dtype=dt),
                margin_rad=c.params.MarginRad, margin_radial=c.params.MarginRadial,
                eta_margins=c.eta_margins, ome_margins=c.ome_margins,
                eta_bin_size=c.params.EtaBinSize, ome_bin_size=c.params.OmeBinSize,
                n_eta_bins=c.n_eta_bins, n_ome_bins=c.n_ome_bins, rings_to_reject=c.rings_to_reject,
                distance=c.params.Distance, pos=pj, max_n_cap=c.bin_max_count)
            m_tot += res.n_matches.to(dt)
            v_tot += valid.sum(-1).to(dt)
            per.append(res.n_matches.detach().cpu().numpy())
            if want_rows:
                rows.append((theor, valid, res))
        return m_tot, v_tot, np.stack(per, 1), rows

    # ---------------------------------------------------------------- seeds
    def candidates(self, k: int, spot_id: int):
        """Per-seed candidate set of mounting ``k`` (``process_seed`` steps 1-2), in k's frame."""
        c = self.ctxs[k]
        p = c.params
        dev, dt = self.device, self.dtype
        r = c.find_obs_row_by_id(spot_id)
        if r < 0:
            return None
        so = c.obs[r]
        ys, zs, om, rrad = float(so[0]), float(so[1]), float(so[2]), float(so[3])
        eta, rn = float(so[6]), int(so[5])
        if rn not in c.ring_hkl:
            return None
        ring_rad = p.get_ring_radius(rn)
        ring_rad = ring_rad if ring_rad > 0 else rrad
        tth = c.ring_ttheta[rn]
        if p.UseFriedelPairs == 1:
            yz = seeds_module.generate_ideal_spots_friedel(
                ys=ys, zs=zs, ttheta_deg=tth, eta_deg=eta, omega_deg=om, ring_nr=rn, ring_rad=ring_rad,
                rsample=p.Rsample, hbeam=p.Hbeam, ome_tol=p.MarginOme, radius_tol=p.MarginRadial,
                obs_spots=c.obs, device=dev, dtype=dt)
            if yz.shape[0] == 0:
                yz = seeds_module.generate_ideal_spots_friedel_mixed(
                    ys=ys, zs=zs, ttheta_deg=tth, eta_deg=eta, omega_deg=om, ring_nr=rn,
                    ring_rad=ring_rad, lsd=p.Distance, rsample=p.Rsample, hbeam=p.Hbeam,
                    step_size_pos=p.StepsizePos, ome_tol=p.MarginOme, radial_tol=p.MarginRadial,
                    eta_tol_um=p.MarginEta, obs_spots=c.obs, device=dev, dtype=dt)
        else:
            yz = seeds_module.generate_ideal_spots(
                ys=ys, zs=zs, ttheta_deg=tth, eta_deg=eta, ring_rad=ring_rad, rsample=p.Rsample,
                hbeam=p.Hbeam, step_size=p.StepsizePos, device=dev, dtype=dt)
        Rc, Pc = [], []
        for i in range(yz.shape[0]):
            y0, z0 = float(yz[i, 0]), float(yz[i, 1])
            pn = _spot_to_gv(p.Distance, y0, z0, om, device=dev, dtype=dt)
            Rs = orientation_grid.generate_candidate_orientations(
                hkl=c.ring_hkl[rn], plane_normal=pn, stepsize_orient_deg=p.StepsizeOrient, ring_nr=rn,
                space_group=p.SpaceGroup, hkl_int=c.ring_hkl_int[rn], abcabg=p.LatticeConstant)
            if Rs.shape[0] == 0:
                continue
            pos, _ = position_grid.build_position_grid(
                seed_y0=torch.tensor([y0], device=dev, dtype=dt), seed_z0=torch.tensor([z0], device=dev, dtype=dt),
                ys=ys, zs=zs, omega_deg=om, distance=p.Distance, r_sample=p.Rsample,
                step_size=p.StepsizePos, h_beam=p.Hbeam)
            if pos.shape[0] == 0:
                continue
            no, npos = Rs.shape[0], pos.shape[0]
            Rc.append(Rs.unsqueeze(1).expand(no, npos, 3, 3).reshape(-1, 3, 3))
            Pc.append(pos.unsqueeze(0).expand(no, npos, 3).reshape(-1, 3))
        if not Rc:
            return None
        return torch.cat(Rc), torch.cat(Pc), rrad

    def index_seed(self, k: int, spot_id: int):
        """Best joint candidate for one seed, or None if it fails the acceptance cut."""
        cand = self.candidates(k, spot_id)
        if cand is None:
            return None
        Rk, Pk, rrad = cand
        Rs, Ps = self.to_sample(Rk, Pk, k)
        best = None
        for c0 in range(0, Rs.shape[0], self.chunk):
            m, v, per, _ = self.score(Rs[c0:c0 + self.chunk], Ps[c0:c0 + self.chunk], rrad)
            frac = torch.where(v > 0, m / v.clamp(min=1), torch.zeros_like(m))
            i = int(torch.argmax(frac * 1e6 + m))            # completeness, then matches
            key = (float(frac[i]), float(m[i]))
            if best is None or key > best[0]:
                best = (key, c0 + i, float(v[i]), per[i])
        (bf, bm), bi, bv, bper = best
        if bf < self.min_frac or bm < self.min_n:
            return None
        return dict(k=k, sid=spot_id, R=Rs[bi].cpu().numpy(), p=Ps[bi].cpu().numpy(), frac=bf, m=bm,
                    v=bv, per=bper, rrad=rrad)

    # ---------------------------------------------------------------- refinement
    def _assignment(self, R, p, rrad):
        _, _, _, rows = self.score(R[None], p[None], rrad, want_rows=True)
        return [(torch.where(res.matched[0])[0], res.matched_obs_row[0][res.matched[0]])
                for (_theor, _valid, res) in rows]

    def _residuals(self, R, p, pairs):
        out = []
        for j, (t_idx, o_rows) in enumerate(pairs):
            if t_idx.numel() == 0:
                continue
            Rj, pj = self.to_mount(R[None], p[None], j)
            theor, _ = self.ctxs[j].adapter.simulate(Rj, pj, lattice=None)
            th = theor[0][t_idx]
            ob = self.ctxs[j].obs[o_rows]
            dom = torch.remainder(th[:, 6] - ob[:, 2] + 180.0, 360.0) - 180.0
            r_om = torch.deg2rad(dom) * torch.hypot(ob[:, 0], ob[:, 1]).clamp(min=1.0)
            out.append(torch.stack([th[:, 10] - ob[:, 0], th[:, 11] - ob[:, 1], r_om], 1).reshape(-1))
        return torch.cat(out) if out else torch.zeros(0, device=self.device, dtype=self.dtype)

    def refine(self, R0: np.ndarray, p0: np.ndarray, rrad: float, n_outer: int = 6, n_inner: int = 4):
        """Fixed-assignment Gauss-Newton on (orientation, position); guarded (see module docstring)."""
        dev, dt = self.device, self.dtype
        R = torch.tensor(R0, device=dev, dtype=dt)
        p = torch.tensor(p0, device=dev, dtype=dt)
        r = self._residuals(R, p, self._assignment(R, p, rrad)).reshape(-1, 3)
        rms0 = float(r[:, :2].pow(2).mean().sqrt()) if r.numel() else float("inf")
        R_start, p_start = R.clone(), p.clone()
        for _ in range(n_outer):
            pairs = self._assignment(R, p, rrad)
            if sum(int(t.numel()) for t, _ in pairs) < 5:
                break
            for _ in range(n_inner):
                r0 = self._residuals(R, p, pairs)
                J = torch.zeros(r0.numel(), 6, device=dev, dtype=dt)
                for d in range(6):
                    e = torch.zeros(6, device=dev, dtype=dt)
                    e[d] = 1e-5 if d < 3 else 0.2
                    J[:, d] = (self._residuals(_rotvec(e[:3]) @ R, p + e[3:], pairs) - r0) / e[d]
                step = torch.linalg.lstsq(J, -r0[:, None]).solution[:, 0]
                R = _rotvec(step[:3]) @ R
                p = p + step[3:]
                if float(torch.linalg.norm(step[3:])) < 0.05 and float(torch.linalg.norm(step[:3])) < 1e-6:
                    break
        r = self._residuals(R, p, self._assignment(R, p, rrad)).reshape(-1, 3)
        rms1 = float(r[:, :2].pow(2).mean().sqrt()) if r.numel() else float("inf")
        if not (rms1 < rms0) or float(torch.linalg.norm(p)) > self.rsample:
            R, p = R_start, p_start
        return R.cpu().numpy(), p.cpu().numpy()

    # ---------------------------------------------------------------- driver
    def run(self, max_seeds: int = 0, log=None) -> JointResult:
        log = log or (lambda s: None)
        t0 = time.time()
        seeds = self.seeds[:max_seeds] if max_seeds else self.seeds
        best = []
        for n, (k, sid) in enumerate(seeds):
            b = self.index_seed(k, sid)
            if b is not None:
                best.append(b)
            if n % 100 == 0:
                log(f"[{time.time()-t0:.0f}s] seed {n}/{len(seeds)} accepted {len(best)}")
        # cluster by orientation only (indexing positions are StepsizePos-coarse)
        order = sorted(range(len(best)), key=lambda i: (-best[i]["frac"], -best[i]["m"]))
        reps = []
        for i in order:
            if not any(_miso_deg(best[i]["R"], best[r]["R"], self.space_group) <= self.cluster_deg for r in reps):
                reps.append(i)
        log(f"[{time.time()-t0:.0f}s] accepted seeds {len(best)}, orientation clusters {len(reps)}")
        grains = []
        for i in reps:
            b = best[i]
            R, p = self.refine(b["R"], b["p"], b["rrad"])
            m, v, per, _ = self.score(torch.tensor(R, device=self.device, dtype=self.dtype)[None],
                                      torch.tensor(p, device=self.device, dtype=self.dtype)[None], b["rrad"])
            grains.append(JointGrain(orient_mat=R, position=p, completeness=float(m[0] / max(float(v[0]), 1.0)),
                                     n_matches=int(m[0]), n_predicted=int(v[0]), matches_per_mounting=per[0],
                                     seed_mounting=b["k"], seed_spot_id=b["sid"]))
        # merge refined duplicates
        grains.sort(key=lambda g: (-g.completeness, -g.n_matches))
        kept: list[JointGrain] = []
        for g in grains:
            if not any(_miso_deg(g.orient_mat, h.orient_mat, self.space_group) <= self.merge_deg
                       and np.linalg.norm(g.position - h.position) <= self.merge_um for h in kept):
                kept.append(g)
        log(f"[{time.time()-t0:.0f}s] grains {len(kept)}")
        return JointResult(grains=kept, n_seeds=len(seeds), n_seeds_accepted=len(best))


# -------------------------------------------------------------------- CLI
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="midas-index-joint",
        description="Joint FF indexing over several mountings of one sample (see midas_index.joint).")
    ap.add_argument("--mount", action="append", required=True, metavar="LAYER_DIR:ROTATION_FILE",
                    help="one per mounting: a processed MIDAS FF layer directory and its sample->mounting "
                         "3x3 rotation (text file, .npy, or .npz with 'mount'). Repeat for each mounting.")
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.csv and <out>.npz)")
    ap.add_argument("--param-file", default="paramstest.txt", help="per-layer params file (relative)")
    ap.add_argument("--min-completeness", type=float, default=None)
    ap.add_argument("--min-nr-spots", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seeds", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    mounts = []
    for item in a.mount:
        layer, rot = item.rsplit(":", 1)
        mounts.append(Mounting(layer_dir=layer, rotation=load_mount_matrix(rot), param_file=a.param_file))
    ji = JointIndexer(mounts, device=a.device, min_completeness=a.min_completeness, min_nr_spots=a.min_nr_spots)
    res = ji.run(max_seeds=a.max_seeds, log=lambda s: print(s, file=sys.stderr, flush=True))
    res.save_csv(a.out + ".csv")
    res.save_npz(a.out + ".npz")
    print(f"midas-index-joint: {len(res.grains)} grains from {res.n_seeds_accepted}/{res.n_seeds} seeds "
          f"-> {a.out}.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
