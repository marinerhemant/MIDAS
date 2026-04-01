"""Reader and interactive plotter for SpotDiagnostics.bin from MIDAS PF-HEDM refinement.

Usage:
    from spot_diagnostics import SpotDiagnostics, SpotDiagPlotter

    diag = SpotDiagnostics('Results/SpotDiagnostics.bin')
    print(diag.summary())

    plotter = SpotDiagPlotter(diag, data_dir='path/to/pfhedm_test')
    plotter.show()                        # Interactive: click map → click spot → see intensity
    plotter.plot_yz_scatter(voxel_nr=56)  # Single-voxel Y-Z scatter
"""

import os
import struct
import numpy as np


SPOT_DIAG_MAGIC = 0x47414944  # "DIAG"
SPOT_DIAG_VERSION = 1
SENTINEL = -999.0

COL_NAMES = [
    'theorY', 'theorZ', 'theorOmega', 'theorEta', 'ringNr', 'hklIndex',
    'theorGx', 'theorGy', 'theorGz', 'theorScanNr',
    'matched', 'obsY', 'obsZ', 'obsOmega', 'obsSpotID', 'obsScanNr',
    'IA', 'diffLen', 'diffOme',
]
NCOLS = len(COL_NAMES)

META_NAMES = [
    'voxelNr', 'posX', 'posY', 'posZ',
    'euler1', 'euler2', 'euler3',
    'a', 'b', 'c', 'alpha', 'beta', 'gamma',
]


class SpotDiagnostics:
    """Reader for SpotDiagnostics.bin produced by FitOrStrainsScanningOMP."""

    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'rb') as f:
            # Header (64 bytes)
            magic, version, nv, nc = struct.unpack('<IIii', f.read(16))
            if magic != SPOT_DIAG_MAGIC:
                raise ValueError(f'Bad magic: 0x{magic:08X} (expected 0x{SPOT_DIAG_MAGIC:08X})')
            if version != SPOT_DIAG_VERSION:
                raise ValueError(f'Unsupported version: {version}')
            self.sentinel = struct.unpack('<d', f.read(8))[0]
            f.read(40)  # reserved

            self.n_voxels = nv
            self.n_cols = nc

            # Directory (12 bytes per voxel)
            dir_data = np.frombuffer(f.read(12 * nv), dtype=np.int32).reshape(nv, 3)
            self.voxel_nrs = dir_data[:, 0].copy()
            self.n_theor = dir_data[:, 1].copy()
            self.n_matched = dir_data[:, 2].copy()

            # Metadata (13 doubles per voxel)
            self.metadata = np.frombuffer(
                f.read(13 * 8 * nv), dtype=np.float64
            ).reshape(nv, 13).copy()

            # Spot data (variable length per voxel)
            self._spot_offsets = np.zeros(nv + 1, dtype=np.int64)
            for i in range(nv):
                self._spot_offsets[i + 1] = self._spot_offsets[i] + self.n_theor[i]
            total_spots = int(self._spot_offsets[nv])
            self.all_spot_data = np.frombuffer(
                f.read(total_spots * nc * 8), dtype=np.float64
            ).reshape(total_spots, nc).copy()

        # Index: voxelNr -> array index
        self._nr_to_idx = {int(nr): i for i, nr in enumerate(self.voxel_nrs)}

    def get_voxel(self, idx):
        """Get voxel data by array index. Returns dict with spots as ndarray."""
        if idx < 0 or idx >= self.n_voxels:
            raise IndexError(f'Index {idx} out of range (0..{self.n_voxels-1})')
        s0 = int(self._spot_offsets[idx])
        s1 = int(self._spot_offsets[idx + 1])
        meta = self.metadata[idx]
        return {
            'voxelNr': int(self.voxel_nrs[idx]),
            'position': meta[1:4],
            'euler': meta[4:7],
            'latc': meta[7:13],
            'nTheor': int(self.n_theor[idx]),
            'nMatched': int(self.n_matched[idx]),
            'spots': self.all_spot_data[s0:s1],
        }

    def get_voxel_by_nr(self, voxel_nr):
        """Get voxel data by voxel number."""
        idx = self._nr_to_idx.get(voxel_nr)
        if idx is None:
            raise KeyError(f'Voxel {voxel_nr} not found')
        return self.get_voxel(idx)

    def completeness_by_ring(self, voxel_nr=None):
        """Per-ring matched/total counts. If voxel_nr is None, aggregate all."""
        if voxel_nr is not None:
            v = self.get_voxel_by_nr(voxel_nr)
            spots = v['spots']
        else:
            spots = self.all_spot_data

        rings = np.unique(spots[:, 4][spots[:, 4] > 0]).astype(int)
        result = {}
        for r in sorted(rings):
            mask = spots[:, 4] == r
            total = int(mask.sum())
            matched = int((spots[mask, 10] > 0.5).sum())
            result[r] = {'total': total, 'matched': matched,
                         'frac': matched / total if total > 0 else 0}
        return result

    def unmatched_spots(self, voxel_nr=None):
        """Return only unmatched spots."""
        if voxel_nr is not None:
            v = self.get_voxel_by_nr(voxel_nr)
            spots = v['spots']
        else:
            spots = self.all_spot_data
        return spots[spots[:, 10] < 0.5]

    def summary(self):
        """Print summary table."""
        lines = [f'SpotDiagnostics: {self.n_voxels} voxels, {NCOLS} cols/spot']
        lines.append(f'{"Vox":>5} {"nTheor":>7} {"nMatch":>7} {"Compl":>6}  '
                     f'{"Pos":>20}  {"Euler":>25}')
        lines.append('-' * 80)
        for i in range(self.n_voxels):
            vn = self.voxel_nrs[i]
            nt = self.n_theor[i]
            nm = self.n_matched[i]
            frac = nm / nt if nt > 0 else 0
            m = self.metadata[i]
            lines.append(
                f'{vn:5d} {nt:7d} {nm:7d} {frac:5.1%}  '
                f'({m[1]:6.1f},{m[2]:6.1f},{m[3]:6.1f})  '
                f'({m[4]:6.1f},{m[5]:6.1f},{m[6]:6.1f})')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Zip file intensity extraction
# ---------------------------------------------------------------------------
def _parse_params(data_dir):
    """Parse Parameters_pfhedm.txt for zip extraction parameters."""
    params = {}
    # Read both files so paramstest.txt values (which have beam center fits) override
    for fn in ['Parameters_pfhedm.txt', 'paramstest.txt']:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            with open(fp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        params[parts[0]] = parts[1].rstrip(';')
    return {
        'omega_step': float(params.get('OmegaStep', '-0.25')),
        'px': float(params.get('px', '200')),
        'n_scans': int(params.get('nScans', '15')),
        'file_stem': params.get('FileStem', 'pfhedm'),
        'padding': int(params.get('Padding', '6')),
        'start_nr': int(params.get('StartFileNrFirstLayer', '1')),
        'im_trans': int(params.get('ImTransOpt', '0')),
        'ybc': float(params.get('YBCFit', params.get('YBC', '1024'))),
        'zbc': float(params.get('ZBCFit', params.get('ZBC', '1024'))),
    }


def _inverse_transform_coords(y, z, trans_opts, ny, nz):
    """Invert ImTransOpt transformations (matching forward_sim_sinogram.py)."""
    yf, zf = float(y), float(z)
    for opt in reversed(trans_opts):
        if opt == 1:    # FlipLR
            yf = ny - 1.0 - yf
        elif opt == 2:  # FlipUD
            zf = nz - 1.0 - zf
        elif opt == 3:  # Transpose
            yf, zf = zf, yf
    return yf, zf


def extract_spot_intensity(data_dir, scan_nr, y_um, z_um, omega_deg,
                           patch_half=10, ome_half=2):
    """Extract intensity from a zip file at a predicted spot position.

    Follows the same coordinate transformation as forward_sim_sinogram.py:
    1. Convert microns → pixels: y_px = y_um / px, z_px = z_um / px
    2. Apply inverse ImTransOpt
    3. Index zarr as data[frame, z_row, y_col]

    Returns: dict with 'frames', 'intensities', 'patches' (list of 2D arrays)
    """
    import zarr

    p = _parse_params(data_dir)
    omega_step = p['omega_step']
    px = p['px']

    # Zip path: data_dir/{scanNr+1}/{FileStem}_{dirNr:0Padding}.MIDAS.zip
    dir_nr = scan_nr + 1
    scan_dir = os.path.join(data_dir, str(dir_nr))
    zip_name = f'{p["file_stem"]}_{dir_nr:0{p["padding"]}d}.MIDAS.zip'
    zip_path = os.path.join(scan_dir, zip_name)

    if not os.path.exists(zip_path):
        return None

    store = zarr.storage.ZipStore(zip_path, mode='r')
    zg = zarr.open_group(store, mode='r')
    data = zg['exchange/data']
    n_frames, nz, ny = data.shape

    # Omega to frame: frame = (180 - omega) / |omega_step|
    center_frame = int(round((180.0 - omega_deg) / abs(omega_step)))

    # Y/Z microns → detector pixel coordinates
    # MIDAS convention: col = YBC - Y_um/px, row = Z_um/px + ZBC
    ybc = p['ybc']
    zbc = p['zbc']
    y_px = ybc - y_um / px   # Y positive = left, col increases right
    z_px = z_um / px + zbc   # Z positive = up, row increases down

    # Apply inverse ImTransOpt to get raw zarr coordinates
    trans_opts = [p['im_trans']] if p['im_trans'] != 0 else []
    y_raw, z_raw = _inverse_transform_coords(y_px, z_px, trans_opts, ny, nz)

    # zarr layout: data[frame, z_row, y_col]
    row = int(round(z_raw))
    col = int(round(y_raw))

    frames = list(range(center_frame - ome_half, center_frame + ome_half + 1))
    intensities = []
    patches = []
    frame_numbers = []

    for fi in frames:
        if fi < 0 or fi >= n_frames:
            intensities.append(0.0)
            patches.append(np.zeros((2 * patch_half + 1, 2 * patch_half + 1)))
            frame_numbers.append(fi)
            continue
        r0 = max(0, row - patch_half)
        r1 = min(nz, row + patch_half + 1)
        c0 = max(0, col - patch_half)
        c1 = min(ny, col + patch_half + 1)
        if r0 < r1 and c0 < c1:
            patch = np.array(data[fi, r0:r1, c0:c1])
            intensities.append(float(np.sum(patch)))
            patches.append(patch)
        else:
            intensities.append(0.0)
            patches.append(np.zeros((2 * patch_half + 1, 2 * patch_half + 1)))
        frame_numbers.append(fi)

    store.close()
    return {
        'frames': frame_numbers,
        'intensities': intensities,
        'patches': patches,
        'center_frame': center_frame,
        'row': row, 'col': col,
        'omega_deg': omega_deg,
        'scan_nr': scan_nr,
    }


class SpotDiagPlotter:
    """Interactive plotting for SpotDiagnostics data.

    Args:
        diag: SpotDiagnostics instance
        data_dir: path to the dataset directory (containing scan subdirs with
                  zip files). Required for intensity extraction on click.
    """

    def __init__(self, diag, data_dir=None):
        self.diag = diag
        self.data_dir = data_dir

    def plot_yz_scatter(self, voxel_nr, ax=None):
        """Y vs Z scatter, colored by matched/unmatched, marker by ring."""
        import matplotlib.pyplot as plt
        v = self.diag.get_voxel_by_nr(voxel_nr)
        spots = v['spots']
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        matched = spots[:, 10] > 0.5
        rings = spots[:, 4].astype(int)
        markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

        for r in sorted(np.unique(rings)):
            rm = rings == r
            mk = markers[r % len(markers)]
            um = rm & ~matched
            if um.any():
                ax.scatter(spots[um, 0], spots[um, 1], marker=mk, s=60,
                           facecolors='none', edgecolors='red', linewidths=1.5,
                           picker=picker, pickradius=8)
            mm = rm & matched
            if mm.any():
                ax.scatter(spots[mm, 0], spots[mm, 1], marker=mk, s=30,
                           c='tab:blue', alpha=0.7, picker=picker, pickradius=8)

        comp = v['nMatched'] / v['nTheor'] if v['nTheor'] > 0 else 0
        ax.set_title(f'Voxel {voxel_nr}: Y-Z ({v["nMatched"]}/{v["nTheor"]} = {comp:.0%})')
        ax.set_xlabel('Y (um)')
        ax.set_ylabel('Z (um)')
        ax.set_aspect('equal')
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
                   markersize=8, label='Matched'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='red', markersize=8, label='Unmatched'),
        ]
        ax.legend(handles=handles, loc='upper right')
        return ax

    def plot_eta_omega(self, voxel_nr, ax=None):
        """Eta vs Omega scatter, colored by match status."""
        import matplotlib.pyplot as plt
        v = self.diag.get_voxel_by_nr(voxel_nr)
        spots = v['spots']
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        matched = spots[:, 10] > 0.5
        ax.scatter(spots[~matched, 2], spots[~matched, 3], c='red', s=50,
                   marker='x', label='Unmatched', zorder=3,
                   picker=picker, pickradius=8)
        ax.scatter(spots[matched, 2], spots[matched, 3], c='tab:blue', s=20,
                   alpha=0.7, label='Matched', zorder=2,
                   picker=picker, pickradius=8)
        ax.set_xlabel('Omega (deg)')
        ax.set_ylabel('Eta (deg)')
        ax.set_title(f'Voxel {voxel_nr}: Eta vs Omega')
        ax.legend(loc='upper right')
        return ax

    def plot_ring_completeness(self, voxel_nr=None, ax=None):
        """Bar chart of matched/total per ring."""
        import matplotlib.pyplot as plt
        comp = self.diag.completeness_by_ring(voxel_nr)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        rings = sorted(comp.keys())
        totals = [comp[r]['total'] for r in rings]
        matchd = [comp[r]['matched'] for r in rings]
        x = np.arange(len(rings))
        ax.bar(x, totals, 0.4, label='Total', color='lightgray', edgecolor='gray')
        ax.bar(x + 0.4, matchd, 0.4, label='Matched', color='tab:blue')
        ax.set_xticks(x + 0.2)
        ax.set_xticklabels([f'R{r}' for r in rings])
        ax.set_ylabel('Count')
        title = f'Voxel {voxel_nr}' if voxel_nr is not None else 'All voxels'
        ax.set_title(f'{title}: Ring Completeness')
        ax.legend()
        return ax

    def plot_scan_heatmap(self, voxel_nr, ax=None, cax=None):
        """Scan x Ring heatmap of expected vs matched spots."""
        import matplotlib.pyplot as plt
        v = self.diag.get_voxel_by_nr(voxel_nr)
        spots = v['spots']
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        rings = sorted(np.unique(spots[:, 4]).astype(int))
        scans_col = spots[:, 9]
        valid = scans_col > -900
        if not valid.any():
            ax.text(0.5, 0.5, 'No scan data', ha='center', va='center',
                    transform=ax.transAxes)
            return ax
        scans = sorted(np.unique(scans_col[valid]).astype(int))
        n_rings = len(rings)
        n_scans = len(scans)
        ring_idx = {r: i for i, r in enumerate(rings)}
        scan_idx = {s: i for i, s in enumerate(scans)}

        expected = np.zeros((n_scans, n_rings), dtype=int)
        matched_arr = np.zeros((n_scans, n_rings), dtype=int)

        for row in spots:
            si = scan_idx.get(int(row[9]), None)
            ri = ring_idx.get(int(row[4]), None)
            if si is None or ri is None:
                continue
            expected[si, ri] += 1
            if row[10] > 0.5:
                matched_arr[si, ri] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(expected > 0, matched_arr / expected, np.nan)
        im = ax.imshow(frac.T, origin='lower', aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1)
        ax.set_xticks(range(n_scans))
        ax.set_xticklabels(scans, fontsize=7)
        ax.set_yticks(range(n_rings))
        ax.set_yticklabels([f'R{r}' for r in rings])
        ax.set_xlabel('Scan Nr')
        ax.set_ylabel('Ring')
        ax.set_title(f'Voxel {voxel_nr}: Scan x Ring Match Fraction')
        if cax is not None:
            plt.colorbar(im, cax=cax, label='Match frac')
        else:
            plt.colorbar(im, ax=ax, label='Match frac')
        return ax

    def plot_missing_by_scan(self, voxel_nr, ax=None):
        """Scatter: missing spots with scan on x-axis, omega on y-axis, colored by ring."""
        import matplotlib.pyplot as plt
        v = self.diag.get_voxel_by_nr(voxel_nr)
        spots = v['spots']
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        unmatched = spots[spots[:, 10] < 0.5]
        if len(unmatched) == 0:
            ax.text(0.5, 0.5, 'No missing spots', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'Voxel {voxel_nr}: Missing by Scan')
            return ax

        scans = unmatched[:, 9]
        valid = scans > -900
        if not valid.any():
            ax.text(0.5, 0.5, 'No scan info', ha='center', va='center',
                    transform=ax.transAxes)
            return ax

        um_valid = unmatched[valid]
        rings = um_valid[:, 4].astype(int)
        ring_colors = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green',
                       4: 'tab:red', 5: 'tab:purple', 6: 'tab:brown'}
        markers = ['o', 's', 'D', '^', 'v', 'P']
        for r in sorted(np.unique(rings)):
            rm = rings == r
            mk = markers[r % len(markers)]
            ax.scatter(um_valid[rm, 9], um_valid[rm, 2], marker=mk, s=60,
                       c=ring_colors.get(r, 'gray'), label=f'R{r}', zorder=3,
                       picker=picker, pickradius=8)
        ax.set_xlabel('Expected Scan Nr')
        ax.set_ylabel('Omega (deg)')
        ax.set_title(f'Voxel {voxel_nr}: Missing Spots ({len(unmatched)} total)')
        ax.legend(fontsize=8, loc='upper right')
        return ax

    def _build_completeness_grid(self):
        """Build NxN completeness grid from voxel metadata for the map panel."""
        d = self.diag
        n = int(np.sqrt(d.n_voxels) + 0.5)
        if n * n != d.n_voxels:
            return None, None, None
        comp = np.full(d.n_voxels, np.nan)
        for i in range(d.n_voxels):
            nt = d.n_theor[i]
            comp[i] = d.n_matched[i] / nt if nt > 0 else 0
        grid = comp.reshape(n, n)
        xs = d.metadata[:, 1]
        ys = d.metadata[:, 2]
        xs_u = np.sort(np.unique(xs))
        ys_u = np.sort(np.unique(ys))
        if len(xs_u) == n and len(ys_u) == n:
            dx = (xs_u[1] - xs_u[0]) / 2 if n > 1 else 2.5
            ext = [xs_u[0] - dx, xs_u[-1] + dx, ys_u[0] - dx, ys_u[-1] + dx]
        else:
            ext = [-0.5, n - 0.5, -0.5, n - 0.5]
        return grid, ext, n

    def show(self, initial_voxel=None):
        """Interactive figure: clickable completeness map + detail panels.

        Layout (3 rows x 3 cols):
          [0,0] Completeness map (click to select voxel)
          [0,1] Y-Z scatter (click spot for intensity)
          [0,2] Eta vs Omega (click spot for intensity)
          [1,0] Ring completeness
          [1,1] Scan x Ring heatmap
          [1,2] Match quality (jet colormap)
          [2,0] Missing spots by scan
          [2,1] Intensity profile (from clicked spot)
          [2,2] Intensity patch (from clicked spot)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        d = self.diag
        if initial_voxel is None:
            initial_voxel = int(d.voxel_nrs[0])

        grid, ext, n = self._build_completeness_grid()

        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        # Persistent colorbar axes
        div_map = make_axes_locatable(axes[0, 0])
        cax_map = div_map.append_axes('right', size='5%', pad=0.05)
        div_scan = make_axes_locatable(axes[1, 1])
        cax_scan = div_scan.append_axes('right', size='5%', pad=0.05)
        div_mq = make_axes_locatable(axes[1, 2])
        cax_mq = div_mq.append_axes('right', size='5%', pad=0.05)

        state = {'current_vn': initial_voxel, 'spots': None}

        def draw_map(highlight_vn=None):
            ax = axes[0, 0]
            ax.clear()
            cax_map.clear()
            if grid is not None:
                im = ax.imshow(grid.T, origin='lower', cmap='RdYlGn',
                               vmin=0, vmax=1, extent=ext)
                ax.set_xlabel('X (um)')
                ax.set_ylabel('Y (um)')
                plt.colorbar(im, cax=cax_map, label='Completeness')
                if highlight_vn is not None:
                    idx = d._nr_to_idx.get(highlight_vn)
                    if idx is not None:
                        px = d.metadata[idx, 1]
                        py = d.metadata[idx, 2]
                        ax.plot(px, py, 'ks', markersize=14,
                                markerfacecolor='none', markeredgewidth=2.5)
            ax.set_title('Completeness (click to select)', fontsize=11)

        def update_voxel(vn):
            state['current_vn'] = vn
            v = d.get_voxel_by_nr(vn)
            state['spots'] = v['spots']
            draw_map(vn)
            # Clear all detail panels
            for r in range(3):
                for c in range(3):
                    if (r, c) == (0, 0):
                        continue
                    axes[r, c].clear()
            cax_scan.clear()
            cax_mq.clear()

            # Row 0: scatter plots (with picker for click-on-spot)
            self.plot_yz_scatter(vn, axes[0, 1])
            self.plot_eta_omega(vn, axes[0, 2])

            # Row 1: ring, scan heatmap, match quality
            self.plot_ring_completeness(vn, axes[1, 0])
            self.plot_scan_heatmap(vn, axes[1, 1], cax=cax_scan)

            # Match quality with jet colormap
            spots = v['spots']
            matched = spots[spots[:, 10] > 0.5]
            if len(matched) > 0:
                sc = axes[1, 2].scatter(matched[:, 17], matched[:, 16],
                                        c=matched[:, 18], cmap='jet', s=20,
                                        )
                axes[1, 2].set_xlabel('diffLen (um)')
                axes[1, 2].set_ylabel('IA (deg)')
                plt.colorbar(sc, cax=cax_mq, label='diffOme')
            else:
                axes[1, 2].text(0.5, 0.5, 'No matched spots', ha='center',
                                va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title(f'Voxel {vn}: Match Quality')

            # Row 2: missing by scan (with picker) + placeholders for intensity
            self.plot_missing_by_scan(vn, axes[2, 0])
            axes[2, 1].text(0.5, 0.5, 'Click a spot in Y-Z or Eta-Omega\n'
                            'to see intensity profile',
                            ha='center', va='center', transform=axes[2, 1].transAxes,
                            fontsize=11, color='gray')
            axes[2, 1].set_title('Intensity Profile')
            axes[2, 2].text(0.5, 0.5, 'Detector patch\nwill appear here',
                            ha='center', va='center', transform=axes[2, 2].transAxes,
                            fontsize=11, color='gray')
            axes[2, 2].set_title('Detector Patch')

            fig.suptitle(
                f'Voxel {vn}: pos=({v["position"][0]:.1f}, {v["position"][1]:.1f}) um, '
                f'{v["nMatched"]}/{v["nTheor"]} matched',
                fontsize=14, fontweight='bold')
            fig.canvas.draw_idle()

        def on_click(event):
            if event.xdata is None or event.ydata is None:
                return

            if event.inaxes == axes[0, 0] and grid is not None:
                # Click on map → select voxel
                cx, cy = event.xdata, event.ydata
                dists = (d.metadata[:, 1] - cx)**2 + (d.metadata[:, 2] - cy)**2
                idx = np.argmin(dists)
                vn = int(d.voxel_nrs[idx])
                update_voxel(vn)
                return

            # Click on spot panels → select spot
            ax = event.inaxes
            if ax not in (axes[0, 1], axes[0, 2], axes[2, 0], axes[1, 2]):
                return
            if state['spots'] is None:
                return

            spots = state['spots']
            cx, cy = event.xdata, event.ydata

            # Find nearest spot using normalized distance (so axes scale doesn't matter)
            if ax == axes[0, 1]:  # Y-Z plot
                xr = ax.get_xlim(); yr = ax.get_ylim()
                sx = (xr[1]-xr[0]) if xr[1]!=xr[0] else 1
                sy = (yr[1]-yr[0]) if yr[1]!=yr[0] else 1
                dists = ((spots[:, 0] - cx)/sx)**2 + ((spots[:, 1] - cy)/sy)**2
            elif ax == axes[0, 2]:  # Eta-Omega plot
                xr = ax.get_xlim(); yr = ax.get_ylim()
                sx = (xr[1]-xr[0]) if xr[1]!=xr[0] else 1
                sy = (yr[1]-yr[0]) if yr[1]!=yr[0] else 1
                dists = ((spots[:, 2] - cx)/sx)**2 + ((spots[:, 3] - cy)/sy)**2
            elif ax == axes[1, 2]:  # Match quality: x=diffLen(17), y=IA(16)
                mask = spots[:, 10] > 0.5
                if not mask.any():
                    return
                xr = ax.get_xlim(); yr = ax.get_ylim()
                sx = (xr[1]-xr[0]) if xr[1]!=xr[0] else 1
                sy = (yr[1]-yr[0]) if yr[1]!=yr[0] else 1
                dists = np.full(len(spots), 1e30)
                dists[mask] = ((spots[mask, 17] - cx)/sx)**2 + ((spots[mask, 16] - cy)/sy)**2
            else:  # Missing-by-scan: x=scan(col9), y=omega(col2)
                mask = (spots[:, 10] < 0.5) & (spots[:, 9] > -900)
                if not mask.any():
                    return
                xr = ax.get_xlim(); yr = ax.get_ylim()
                sx = (xr[1]-xr[0]) if xr[1]!=xr[0] else 1
                sy = (yr[1]-yr[0]) if yr[1]!=yr[0] else 1
                dists = np.full(len(spots), 1e30)
                dists[mask] = ((spots[mask, 9] - cx)/sx)**2 + ((spots[mask, 2] - cy)/sy)**2
            si = np.argmin(dists)
            spot = spots[si]

            y_um = spot[0]
            z_um = spot[1]
            omega = spot[2]
            ring = int(spot[4])
            is_matched = spot[10] > 0.5
            exp_scan = int(spot[9]) if spot[9] > -900 else -1

            # Highlight the clicked spot in all scatter panels
            for a in (axes[0, 1], axes[0, 2], axes[2, 0], axes[1, 2]):
                for coll in list(a.collections):
                    if getattr(coll, '_is_highlight', False):
                        coll.remove()
            # Highlight in Y-Z
            h1 = axes[0, 1].scatter([y_um], [z_um], s=200, facecolors='none',
                                     edgecolors='lime', linewidths=3, zorder=10)
            h1._is_highlight = True
            # Highlight in Eta-Omega
            h2 = axes[0, 2].scatter([omega], [spot[3]], s=200, facecolors='none',
                                     edgecolors='lime', linewidths=3, zorder=10)
            h2._is_highlight = True
            # Highlight in Missing-by-scan (if spot has valid scan)
            if exp_scan >= 0:
                h3 = axes[2, 0].scatter([exp_scan], [omega], s=200,
                                         facecolors='none', edgecolors='lime',
                                         linewidths=3, zorder=10)
                h3._is_highlight = True
            # Highlight in Match quality (if matched)
            if is_matched:
                h4 = axes[1, 2].scatter([spot[17]], [spot[16]], s=200,
                                         facecolors='none', edgecolors='lime',
                                         linewidths=3, zorder=10)
                h4._is_highlight = True

            # Extract intensity from zip
            status = 'MATCHED' if is_matched else 'UNMATCHED'
            info = (f'R{ring}, ome={omega:.1f}, scan={exp_scan}, {status}')

            if exp_scan >= 0:
                result = extract_spot_intensity(
                    self.data_dir, exp_scan, y_um, z_um, omega,
                    patch_half=10, ome_half=2)
            else:
                result = None

            # Plot intensity profile
            axes[2, 1].clear()
            if result:
                frames = result['frames']
                intensities = result['intensities']
                best_idx = int(np.argmax(intensities))
                colors = ['tab:red' if i == best_idx else 'tab:blue'
                          for i in range(len(frames))]
                axes[2, 1].bar(range(len(frames)), intensities, color=colors)
                axes[2, 1].set_xticks(range(len(frames)))
                axes[2, 1].set_xticklabels([str(f) for f in frames], fontsize=8)
                axes[2, 1].set_xlabel('Frame')
                axes[2, 1].set_ylabel('Sum Intensity (21x21)')
                axes[2, 1].set_title(f'Intensity: {info}')
            else:
                axes[2, 1].text(0.5, 0.5, f'No zip data\n{info}',
                                ha='center', va='center',
                                transform=axes[2, 1].transAxes)
                axes[2, 1].set_title('Intensity Profile')

            # Plot patch: sum over all frames (omega-integrated)
            axes[2, 2].clear()
            if result and result['patches']:
                sum_patch = np.zeros_like(result['patches'][0], dtype=float)
                for p in result['patches']:
                    sum_patch += p
                axes[2, 2].imshow(sum_patch, origin='lower', cmap='viridis',
                                  aspect='equal')
                axes[2, 2].set_title(
                    f'Sum patch (frames {result["frames"][0]}-'
                    f'{result["frames"][-1]}), '
                    f'row={result["row"]}, col={result["col"]}')
            else:
                axes[2, 2].text(0.5, 0.5, 'No patch data',
                                ha='center', va='center',
                                transform=axes[2, 2].transAxes)
                axes[2, 2].set_title('Detector Patch')

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_click)
        update_voxel(initial_voxel)
        plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python spot_diagnostics.py <SpotDiagnostics.bin> [--show] [--data-dir DIR]')
        sys.exit(1)

    diag = SpotDiagnostics(sys.argv[1])
    print(diag.summary())
    print()
    print('Ring completeness (all voxels):')
    for ring, info in diag.completeness_by_ring().items():
        print(f'  Ring {ring}: {info["matched"]}/{info["total"]} ({info["frac"]:.1%})')

    if '--show' in sys.argv:
        data_dir = None
        if '--data-dir' in sys.argv:
            idx = sys.argv.index('--data-dir')
            data_dir = sys.argv[idx + 1]
        else:
            # Auto-detect: go up from Results/ to the data dir
            bin_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
            parent = os.path.dirname(bin_dir)
            if os.path.exists(os.path.join(parent, 'Parameters_pfhedm.txt')):
                data_dir = parent
            elif os.path.exists(os.path.join(bin_dir, 'Parameters_pfhedm.txt')):
                data_dir = bin_dir

        plotter = SpotDiagPlotter(diag, data_dir=data_dir)
        plotter.show()
