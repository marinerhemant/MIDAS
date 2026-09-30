"""Joint mixture fit of ALL candidate orientations in one column (one raster point), monochromatic rotation data.

Model on the union voxel set P (windows around every observed reflection of every orientation, masked voxels excluded):

    model(v) = c + sum_j sum_h a[j,h] * sum_k w[j,k] * Kern(v - pos(j,k,h))

* pos(j,k,h) = (frame, row, col) of reflection h of orientation j rotated by its component offset (sample-frame rotation
  vector, :func:`~.forward.predict_torch`);
* Kern = the MEASURED anisotropic kernel (:class:`~.kernel.GaussKernel3D`), unit flux, with one fitted global width scale;
* a >= 0 per-reflection brightness and w >= 0 per-component weights, by alternating NNLS over ALL orientations jointly
  (orientations sharing voxels are fitted together; nothing is masked as contamination); c a constant;
* offsets by Adam.

Design carried over from the validated Laue implementation (LaueMatching pipeline/analysis/column_content), including
the fixes that implementation paid for:
* windows are the bounding boxes of the DETECTED blobs touching each predicted reflection (never the fit's own model),
  dilated and clipped -- fixed-size windows truncated real spread;
* optimiser step 3e-4 rad (2e-3 moved components ~0.1 deg/step and collapsed the fit);
* the final linear solve restarts from the BEST iterate's (a, w), and (a, w) are reset if either collapses to zero
  (alternating NNLS has an absorbing zero state).

Shares follow the definition that is honest about misses: share_j = flux_j / (modelled flux + UNEXPLAINED flux over ALL
detected-blob voxels), so an undetected domain lowers the others' shares instead of inflating them.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage as ndi

from .forward import DT, observable_reflections, predict_torch

def _nnls(D: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact NNLS  min ||D^T x - y||, x >= 0, D (m, NP).

    Solved in the m-dimensional space: with G = D D^T = L L^T (Cholesky; a 1e-12 relative ridge keeps coincident
    components factorable), ||D^T x - y||^2 = ||L^T x - L^-1 D y||^2 + const, so scipy.optimize.nnls on the (m, m)
    system gives the same minimiser as on the (NP, m) design (~100x faster; checked in tests)."""
    G = (D @ D.T).detach().numpy(); b = (D @ y).detach().numpy()
    return torch.as_tensor(_nnls_gram(G, b), dtype=D.dtype)


def _nnls_gram(G: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NNLS from the normal equations: G = D D^T (m, m), b = D y. Same minimiser as _nnls, without ever forming D."""
    from scipy.optimize import nnls
    G = G + 1e-12 * max(float(np.trace(G)) / max(len(G), 1), 1e-300) * np.eye(len(G))
    L = np.linalg.cholesky(G)
    x, _ = nnls(L.T, np.linalg.solve(L, b), maxiter=50 * len(G))
    return x


class ColumnFit:
    """Fit the orientation content of one column.

    sub     (nF, H, W) background-subtracted frames (midas_defect.ingest.subtract_background output)
    mask    (nF, H, W) or (H, W) bool, True = EXCLUDED voxel -- the midas_defect convention (build_mask().mask);
            add saturated voxels to it. (Passing a True=usable array here silently fits nothing.)
    labels  (nF, H, W) int blob labels (midas_defect.ingest.find_blobs_3d(..., return_labels=True))
    U_list  orientation matrices (crystal -> sample, as midas_defect.domains.Domain.U)
    """

    WF_MIN, WF_MAX, WP_MIN, WP_MAX, DIL_F, DIL_P, TOUCH_F, TOUCH_P = 5, 13, 13, 81, 2, 6, 1, 2

    def __init__(self, sub, mask, labels, U_list, B, hkl_all, geom, kernel, *, K: int = 24, omega_sign: int = 1,
                 fast_linear: bool = True):
        self.geom, self.kern, self.K, self.omega_sign = geom, kernel, K, omega_sign
        self.fast_linear, self._gram = fast_linear, None      # fast_linear=False: the dense reference path (tests)
        self.B = np.asarray(B, float)
        self.sub = np.asarray(sub, float)
        nF, H, W = self.sub.shape
        valid = ~np.asarray(mask, bool)
        if valid.ndim == 2:
            valid = np.broadcast_to(valid, self.sub.shape)
        self.valid = valid
        self.labels = np.asarray(labels)
        self.blob = (self.labels > 0) & valid
        objs = ndi.find_objects(self.labels)
        self.u = torch.zeros((), dtype=DT, requires_grad=True)
        self.U0, self.obs, self.boxes = [], [], []
        for U in U_list:
            obs = observable_reflections(U, self.B, hkl_all, geom, omega_sign=omega_sign)
            keep, boxes = [], []
            for i in range(len(obs.hkl)):
                f, r, c = int(round(obs.frame[i])), int(round(obs.row[i])), int(round(obs.col[i]))
                sl = self.labels[max(f - self.TOUCH_F, 0):f + self.TOUCH_F + 1, max(r - self.TOUCH_P, 0):r + self.TOUCH_P + 1,
                                 max(c - self.TOUCH_P, 0):c + self.TOUCH_P + 1]
                ids = set(np.unique(sl)) - {0}
                if not ids:
                    continue
                f0 = min(objs[i_ - 1][0].start for i_ in ids) - self.DIL_F; f1 = max(objs[i_ - 1][0].stop - 1 for i_ in ids) + self.DIL_F
                r0 = min(objs[i_ - 1][1].start for i_ in ids) - self.DIL_P; r1 = max(objs[i_ - 1][1].stop - 1 for i_ in ids) + self.DIL_P
                c0 = min(objs[i_ - 1][2].start for i_ in ids) - self.DIL_P; c1 = max(objs[i_ - 1][2].stop - 1 for i_ in ids) + self.DIL_P
                hf, hp, mf, mp = self.WF_MAX // 2, self.WP_MAX // 2, self.WF_MIN // 2, self.WP_MIN // 2
                f0, f1 = max(f0, f - hf), min(f1, f + hf); r0, r1 = max(r0, r - hp), min(r1, r + hp); c0, c1 = max(c0, c - hp), min(c1, c + hp)
                f0, f1 = min(f0, f - mf), max(f1, f + mf); r0, r1 = min(r0, r - mp), max(r1, r + mp); c0, c1 = min(c0, c - mp), max(c1, c + mp)
                keep.append(i); boxes.append((max(f0, 0), min(f1, nF - 1), max(r0, 0), min(r1, H - 1), max(c0, 0), min(c1, W - 1)))
            sel = np.asarray(keep, int)
            for fld in ("hkl", "branch", "wrap", "frame", "row", "col", "two_theta_deg"):
                setattr(obs, fld, getattr(obs, fld)[sel] if len(sel) else getattr(obs, fld)[:0])
            self.U0.append(np.asarray(U, float)); self.obs.append(obs); self.boxes.append(boxes)
        mask = np.zeros(self.sub.shape, bool)
        for bx in self.boxes:
            for f0, f1, r0, r1, c0, c1 in bx:
                mask[f0:f1 + 1, r0:r1 + 1, c0:c1 + 1] = True
        mask &= valid
        self.mask = mask
        self.pix = np.flatnonzero(mask.ravel()); self.NP = len(self.pix)
        idx = np.full(self.sub.size, -1, np.int64); idx[self.pix] = np.arange(self.NP)
        self.pidx, self.pref, self.vf, self.vr, self.vc = [], [], [], [], []
        for bx in self.boxes:
            pi, pr, qf, qr, qc = [], [], [], [], []
            for h, (f0, f1, r0, r1, c0, c1) in enumerate(bx):
                ff, rr, cc = np.mgrid[f0:f1 + 1, r0:r1 + 1, c0:c1 + 1]
                ii = idx[(ff * H * W + rr * W + cc).ravel()]
                k = ii >= 0
                pi.append(ii[k]); pr.append(np.full(int(k.sum()), h)); qf.append(ff.ravel()[k]); qr.append(rr.ravel()[k]); qc.append(cc.ravel()[k])
            cat = lambda v, dt: torch.as_tensor(np.concatenate(v) if v else np.zeros(0), dtype=dt)  # noqa: E731
            self.pidx.append(cat(pi, torch.long)); self.pref.append(cat(pr, torch.long))
            self.vf.append(cat(qf, DT)); self.vr.append(cat(qr, DT)); self.vc.append(cat(qc, DT))
        self.y = torch.as_tensor(self.sub.ravel()[self.pix], dtype=DT)
        self.nref = [len(o.hkl) for o in self.obs]

    # ------------------------------------------------------------------ model
    def kscale(self):
        return 1.0 + 0.25 * torch.tanh(self.u)                   # (0.75, 1.25)

    def blocks(self, omegas):
        out = []
        for j, om in enumerate(omegas):
            if self.nref[j] == 0 or len(self.pidx[j]) == 0:
                out.append(None); continue
            f, r, c = predict_torch(self.U0[j], self.B, self.obs[j], om, self.geom, omega_sign=self.omega_sign)   # (K, n_j)
            rf = self.pref[j]
            df = self.vf[j][None] - f[:, rf]; dr = self.vr[j][None] - r[:, rf]; dc = self.vc[j][None] - c[:, rf]
            out.append(self.kern(df, dr, dc, r[:, rf], c[:, rf], scale=self.kscale()))
        return out

    def _scatter_rows(self, j, V):
        out = torch.zeros(V.shape[0], self.NP, dtype=DT); out.index_add_(1, self.pidx[j], V); return out

    def _per_refl(self, j, v):
        out = torch.zeros(self.nref[j] * self.NP, dtype=DT)
        out.index_add_(0, self.pref[j] * self.NP + self.pidx[j], v)
        return out.reshape(self.nref[j], self.NP)

    def model(self, Bk, a, w, c=0.0, zero_c=False):
        m = torch.zeros(self.NP, dtype=DT)
        for j in range(len(Bk)):
            if Bk[j] is None:
                continue
            m = m.index_add(0, self.pidx[j], (w[j][:, None] * Bk[j]).sum(0) * a[j][self.pref[j]])
        return m if zero_c else m + c

    # -- normal equations without the dense design ------------------------------------------------------------------
    # The design D (m x NP) is almost all zeros: an orientation's rows touch only the union of its reflection windows.
    # G = D D^T and b = D y are assembled exactly from that structure (same numbers as the dense product up to
    # floating-point summation order), so the cost no longer scales with m^2 * NP.

    def _prepare_gram(self):
        if self._gram is not None:
            return
        J = len(self.U0); uniq, inv = [], []
        for j in range(J):
            u, iv = np.unique(self.pidx[j].numpy(), return_inverse=True)     # a voxel can sit in two of j's windows
            uniq.append(torch.as_tensor(u, dtype=torch.long)); inv.append(torch.as_tensor(iv, dtype=torch.long))
        inter = {}
        for j in range(J):
            for j2 in range(j + 1, J):
                _, ia, ib = np.intersect1d(uniq[j].numpy(), uniq[j2].numpy(), assume_unique=True, return_indices=True)
                if len(ia):
                    inter[(j, j2)] = (torch.as_tensor(ia, dtype=torch.long), torch.as_tensor(ib, dtype=torch.long))
        self._gram = dict(uniq=uniq, inv=inv, inter=inter, a_rows=None)

    def _gram_w(self, Bk, a, yc):
        """(G, b) for the component weights w: rows (j, k) = Bk[j][k] * a[j][pref], scattered onto pidx[j]."""
        st, K = self._gram, self.K
        act = [j for j in range(len(Bk)) if Bk[j] is not None]
        off = {j: i * K for i, j in enumerate(act)}
        V = {j: torch.zeros(K, len(st["uniq"][j]), dtype=DT).index_add_(1, st["inv"][j], Bk[j] * a[j][self.pref[j]][None])
             for j in act}
        m = K * len(act); G = np.zeros((m, m)); b = np.zeros(m)
        for j in act:
            s0 = off[j]
            G[s0:s0 + K, s0:s0 + K] = (V[j] @ V[j].T).numpy()
            b[s0:s0 + K] = (V[j] @ yc[st["uniq"][j]]).numpy()
        for (j, j2), (ia, ib) in st["inter"].items():
            if j in V and j2 in V:
                blk = (V[j][:, ia] @ V[j2][:, ib].T).numpy()
                G[off[j]:off[j] + K, off[j2]:off[j2] + K] = blk; G[off[j2]:off[j2] + K, off[j]:off[j] + K] = blk.T
        return G, b

    def _gram_a(self, Bk, w, yc):
        """(G, b) for the per-reflection brightness a: row (j, h) = window h of orientation j, sparse."""
        import scipy.sparse as sp
        st = self._gram
        act = [j for j in range(len(Bk)) if Bk[j] is not None]
        if st["a_rows"] is None:                                              # (row, col) pairs never repeat: a window has unique voxels
            rows, cols, off = [], [], 0
            for j in act:
                rows.append(self.pref[j].numpy() + off); cols.append(self.pidx[j].numpy()); off += self.nref[j]
            rows, cols = np.concatenate(rows), np.concatenate(cols)
            D0 = sp.csr_matrix((np.arange(1, len(rows) + 1, dtype=np.float64), (rows, cols)), shape=(off, self.NP))
            st["a_rows"] = (D0, D0.data.astype(np.int64) - 1, act, len(rows) == D0.nnz)
        D0, order, act0, unique = st["a_rows"]
        vals = np.concatenate([(w[j][:, None] * Bk[j]).sum(0).numpy() for j in act])
        if act != act0 or not unique:                                          # fallback: rebuild (also sums any duplicates)
            rows, cols, off = [], [], 0
            for j in act:
                rows.append(self.pref[j].numpy() + off); cols.append(self.pidx[j].numpy()); off += self.nref[j]
            D = sp.csr_matrix((vals, (np.concatenate(rows), np.concatenate(cols))), shape=(off, self.NP))
        else:
            D = D0.copy(); D.data = vals[order]
        return (D @ D.T).toarray(), D @ yc.numpy()

    def linear(self, Bk, a, w, n_alt=4):
        if not self.fast_linear:
            return self._linear_dense(Bk, a, w, n_alt)
        self._prepare_gram()
        c = torch.zeros((), dtype=DT); J = len(Bk)
        for _ in range(n_alt):
            G, b = self._gram_w(Bk, a, self.y - c); wv = torch.as_tensor(_nnls_gram(G, b), dtype=DT); p = 0
            for j in range(J):
                if Bk[j] is not None:
                    w[j] = wv[p:p + self.K]; p += self.K
            G, b = self._gram_a(Bk, w, self.y - c); av = torch.as_tensor(_nnls_gram(G, b), dtype=DT); p = 0
            for j in range(J):
                if Bk[j] is not None:
                    a[j] = av[p:p + self.nref[j]]; p += self.nref[j]
            c = (self.y - self.model(Bk, a, w, zero_c=True)).mean()
        return a, w, c

    def _linear_dense(self, Bk, a, w, n_alt=4):
        """Reference implementation: the dense (m x NP) design and _nnls. Kept for the equivalence test."""
        c = torch.zeros((), dtype=DT); J = len(Bk)
        for _ in range(n_alt):
            cols = [self._scatter_rows(j, Bk[j] * a[j][self.pref[j]][None]) for j in range(J) if Bk[j] is not None]
            D = torch.cat(cols, 0); wv = _nnls(D, self.y - c); p = 0
            for j in range(J):
                if Bk[j] is not None:
                    w[j] = wv[p:p + self.K]; p += self.K
            cols = [self._per_refl(j, (w[j][:, None] * Bk[j]).sum(0)) for j in range(J) if Bk[j] is not None]
            D = torch.cat(cols, 0); av = _nnls(D, self.y - c); p = 0
            for j in range(J):
                if Bk[j] is not None:
                    a[j] = av[p:p + self.nref[j]]; p += self.nref[j]
            c = (self.y - self.model(Bk, a, w, zero_c=True)).mean()
        return a, w, c

    # ------------------------------------------------------------------ fit
    def _fit_once(self, n_iter, lr, init_deg, seed):
        g = torch.Generator().manual_seed(seed); J = len(self.U0)
        with torch.no_grad():
            self.u.zero_()
        omegas = [(torch.randn(self.K, 3, generator=g, dtype=DT) * math.radians(init_deg)).requires_grad_(True) for _ in range(J)]
        a = [torch.ones(n, dtype=DT) for n in self.nref]; w = [torch.full((self.K,), 1.0 / self.K, dtype=DT) for _ in range(J)]
        opt = torch.optim.Adam([{"params": omegas, "lr": lr}, {"params": [self.u], "lr": 0.02}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr * 0.05)
        den = float((self.y ** 2).sum()) or 1.0; best = (float("inf"), None, 0.0, None, None)
        for _ in range(n_iter):
            opt.zero_grad()
            Bk = self.blocks(omegas)
            with torch.no_grad():
                a, w, c = self.linear([b.detach() if b is not None else None for b in Bk], a, w)
            loss = ((self.model(Bk, a, w, c) - self.y) ** 2).sum() / den
            loss.backward(); opt.step(); sched.step()
            v = float(loss.detach())
            if v < best[0]:
                best = (v, [o.detach().clone() for o in omegas], float(self.u.detach()), [x.clone() for x in a], [x.clone() for x in w])
            if any(float(x.sum()) <= 0 for x in a + w):
                a = [torch.ones(n, dtype=DT) for n in self.nref]; w = [torch.full((self.K,), 1.0 / self.K, dtype=DT) for _ in range(J)]
        with torch.no_grad():
            self.u.fill_(best[2]); a, w = best[3], best[4]
            Bk = self.blocks(best[1]); a, w, c = self.linear(Bk, a, w)
            loss = float(((self.model(Bk, a, w, c) - self.y) ** 2).sum() / den)
        return loss, best[1], a, w, c, Bk, best[2]

    def fit(self, n_iter: int = 300, lr: float = 3e-4, inits=(0.05, 0.4), seed: int = 0):
        if self.NP == 0 or not any(self.nref):
            self.omegas = self.a = self.w = self.B_ = None; self.loss = float("nan"); return self.loss
        runs = [(d,) + self._fit_once(n_iter, lr, d, seed) for d in inits]
        d, loss, om, a, w, c, Bk, u = min(runs, key=lambda r: r[1])
        with torch.no_grad():
            self.u.fill_(u)
        self.omegas, self.a, self.w, self.c, self.Bk, self.loss, self.init_won = om, a, w, c, Bk, loss, d
        self.init_losses = {str(r[0]): r[1] for r in runs}
        return loss

    # ------------------------------------------------------------------ outputs
    def model_stack(self):
        img = np.zeros(self.sub.size)
        if self.NP:
            img[self.pix] = self.model(self.Bk, self.a, self.w, zero_c=True).detach().numpy()
        return img.reshape(self.sub.shape)

    def residual_stack(self):
        return self.sub - self.model_stack()

    def report(self):
        from .forward import rotvec_to_matrix
        J = len(self.U0)
        if self.NP == 0 or self.omegas is None:
            return dict(orientations=[], unexplained_flux_frac=1.0, C_int=0.0, n_voxels=0, kernel_scale=None,
                        init_won=None, loss=float("nan"))
        fl = [float(self.a[j].sum() * self.w[j].sum()) if self.Bk[j] is not None else 0.0 for j in range(J)]
        full = self.model_stack().ravel(); y = self.sub.ravel() - float(self.c)
        onb = self.blob.ravel()
        yy = np.clip(y[onb], 0, None); mm = np.clip(full[onb], 0, None)
        unexpl = float(np.clip(yy - mm, 0, None).sum()); tot = sum(fl) + unexpl
        per = []
        for j in range(J):
            if self.Bk[j] is None or float(self.w[j].sum()) <= 0:
                per.append(dict(n_refl=self.nref[j], share=0.0, U=self.U0[j].tolist(), spread_rms_deg=None)); continue
            wn = (self.w[j] / self.w[j].sum()).numpy(); om = self.omegas[j].numpy()
            mu = (wn[:, None] * om).sum(0); d = om - mu
            S = (wn[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
            ev = np.sort(np.clip(np.linalg.eigvalsh(S), 0, None))[::-1]
            Umean = rotvec_to_matrix(torch.as_tensor(mu, dtype=DT)).numpy() @ self.U0[j]
            per.append(dict(n_refl=self.nref[j], share=fl[j] / max(tot, 1e-12), share_of_modelled=fl[j] / max(sum(fl), 1e-12),
                            U=Umean.tolist(), spread_rms_deg=math.degrees(math.sqrt(float(np.trace(S)))),
                            spread_principal_deg=[math.degrees(math.sqrt(e)) for e in ev],
                            n_occupied=int((self.w[j] > 0.01 * float(self.w[j].max())).sum())))
        return dict(orientations=per, unexplained_flux_frac=unexpl / max(tot, 1e-12),
                    C_int=float(np.minimum(mm, yy).sum() / max(yy.sum(), 1e-12)), kernel_scale=float(self.kscale().detach()),
                    init_won=self.init_won, init_losses=self.init_losses, loss=self.loss, n_voxels=int(self.NP))
