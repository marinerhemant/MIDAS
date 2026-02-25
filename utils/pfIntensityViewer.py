#!/usr/bin/env python
"""
PF-HEDM Interactive Sinogram & Peak Intensity Viewer

Displays two side-by-side plots per grain:
  Left:  Sinogram (scanNr × rotation angle)
  Right: 2D intensity distribution (Y × Z detector pixels) for the selected cell

Usage:
  python pfIntensityViewer.py -paramFile <paramFile> [-resultDir <dir>] [-portNr 8051]
"""

import os, sys, glob, argparse

import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, no_update, ctx
import dash_bootstrap_components as dbc

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
COMMON_LAYOUT = dict(margin=dict(l=10, r=10, b=10, t=50), height=800, template="plotly_dark")
DEFAULT_PATCH_HALF = 15
DEFAULT_REFRESH_MS = 500


# ──────────────────────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────────────────────
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n'); self.print_help(); sys.exit(2)


# ──────────────────────────────────────────────────────────────
# Parameter File Parsing (mirrors pf_MIDAS.py)
# ──────────────────────────────────────────────────────────────
def parse_param_file(paramFN):
    """Parse the PF parameter file and return a dict of relevant values."""
    params = {
        'FileStem': '',
        'StartFileNrFirstLayer': 0,
        'NrFilesPerSweep': 1440,
        'nScans': 1,
        'Padding': 6,
        'ImTransOpt': [],
        'StartNr': 1,
        'EndNr': 1440,
        'OmegaStart': 0,
        'OmegaStep': 0.25,
        'BC_Y': 0,
        'BC_Z': 0,
        'px': 200,
        'numPxY': 2048,
        'numPxZ': 2048,
    }

    with open(paramFN, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0]
            if key == 'FileStem':
                params['FileStem'] = parts[1]
            elif key == 'StartFileNrFirstLayer':
                params['StartFileNrFirstLayer'] = int(parts[1])
            elif key == 'NrFilesPerSweep':
                params['NrFilesPerSweep'] = int(parts[1])
            elif key == 'nScans':
                params['nScans'] = int(parts[1])
            elif key == 'Padding':
                params['Padding'] = int(parts[1])
            elif key == 'ImTransOpt':
                params['ImTransOpt'].append(int(parts[1]))
            elif key == 'StartNr':
                params['StartNr'] = int(parts[1])
            elif key == 'EndNr':
                params['EndNr'] = int(parts[1])
            elif key == 'OmegaStart':
                params['OmegaStart'] = float(parts[1])
            elif key == 'OmegaEnd':
                params['OmegaEnd'] = float(parts[1])
            elif key == 'OmegaStep':
                params['OmegaStep'] = float(parts[1])
            elif key == 'BC':
                params['BC_Y'] = float(parts[1])
                params['BC_Z'] = float(parts[2])
            elif key == 'px':
                params['px'] = float(parts[1])
            elif key == 'numPxY':
                params['numPxY'] = int(parts[1])
            elif key == 'numPxZ':
                params['numPxZ'] = int(parts[1])
    return params


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
def load_patches(topdir):
    """Load pre-extracted intensity patches from patches_*.bin.

    The patches are saved by findSingleSolutionPFRefactored as float32,
    shape (nGrs, maxNHKLs, nScans, 21, 21).
    Also loads spotPositions_*.bin (doubles, shape (nGrs, maxNHKLs, nScans, 2)).
    """
    PATCH_SIZE = 21

    # Find patches file
    pat = glob.glob(os.path.join(topdir, 'patches_*.bin'))
    if not pat:
        print("  WARNING: No patches_*.bin found — patch display disabled")
        return None, None

    fn = pat[0]
    # Parse nGrs, maxNHKLs, nScans from filename: patches_{nGrs}_{maxNHKLs}_{nScans}.bin
    base = os.path.splitext(os.path.basename(fn))[0]
    parts = base.split('_')
    nGrs = int(parts[1])
    maxNHKLs = int(parts[2])
    nScans = int(parts[3])

    patches = np.fromfile(fn, dtype=np.float32).reshape((nGrs, maxNHKLs, nScans, PATCH_SIZE, PATCH_SIZE))
    print(f"  Loaded patches: {fn}")
    print(f"    Shape: {patches.shape}, {patches.nbytes / 1024 / 1024:.1f} MB")
    # Diagnostic: how many patches have any non-zero data
    nonzero_count = np.count_nonzero(np.any(patches.reshape(nGrs, maxNHKLs, nScans, -1), axis=-1))
    print(f"    Non-zero patches: {nonzero_count} / {nGrs * maxNHKLs * nScans}")
    # Show a sample non-zero patch location
    for g in range(nGrs):
        for s in range(maxNHKLs):
            for sc in range(nScans):
                if np.any(patches[g, s, sc] != 0):
                    print(f"    Sample non-zero: grain={g}, spot={s}, scan={sc}, max={patches[g, s, sc].max():.1f}")
                    break
            else:
                continue
            break
        else:
            continue
        break

    # Load spot positions
    spotPos = None
    sp_pat = glob.glob(os.path.join(topdir, 'spotPositions_*.bin'))
    if sp_pat:
        spotPos = np.fromfile(sp_pat[0], dtype=np.float64).reshape((nGrs, maxNHKLs, nScans, 2))
        print(f"  Loaded spot positions: {sp_pat[0]}")

    return patches, spotPos


def load_spot_meta(topdir):
    """Load per-cell spot metadata: eta, 2theta, yCen, zCen.

    Saved by findSingleSolutionPFRefactored as doubles,
    shape (nGrs, maxNHKLs, nScans, 4).
    """
    SPOT_META_COLS = 4
    pat = glob.glob(os.path.join(topdir, 'spotMeta_*.bin'))
    if not pat:
        print("  No spotMeta_*.bin found — hover details limited")
        return None
    fn = pat[0]
    base = os.path.splitext(os.path.basename(fn))[0]
    parts = base.split('_')
    nGrs = int(parts[1])
    maxNHKLs = int(parts[2])
    nScans = int(parts[3])
    meta = np.fromfile(fn, dtype=np.float64).reshape((nGrs, maxNHKLs, nScans, SPOT_META_COLS))
    print(f"  Loaded spot metadata: {fn}")
    return meta


def load_sinogram_variants(topdir):
    """Load all 4 sinogram variant binary files."""
    variants = {}
    # Find the raw sinos file to get dimensions
    raw_files = glob.glob(os.path.join(topdir, 'sinos_raw_*.bin'))
    if not raw_files:
        # Try the default naming (without variant prefix)
        raw_files = glob.glob(os.path.join(topdir, 'sinos_[0-9]*.bin'))
        if raw_files:
            parts = os.path.basename(raw_files[0]).replace('sinos_', '').replace('.bin', '').split('_')
            nGrs, maxNHKLs, nScans = int(parts[0]), int(parts[1]), int(parts[2])
            variants['raw'] = np.fromfile(raw_files[0], dtype=np.double).reshape((nGrs, maxNHKLs, nScans))
            return variants, nGrs, maxNHKLs, nScans
        print("ERROR: No sinogram files found!", file=sys.stderr)
        sys.exit(1)

    # Parse dimensions from filename: sinos_raw_N_M_S.bin
    parts = os.path.basename(raw_files[0]).replace('sinos_raw_', '').replace('.bin', '').split('_')
    nGrs, maxNHKLs, nScans = int(parts[0]), int(parts[1]), int(parts[2])

    for variant in ['raw', 'norm', 'abs', 'normabs']:
        fn = os.path.join(topdir, f'sinos_{variant}_{nGrs}_{maxNHKLs}_{nScans}.bin')
        if os.path.exists(fn):
            variants[variant] = np.fromfile(fn, dtype=np.double).reshape((nGrs, maxNHKLs, nScans))
        else:
            print(f"  Warning: {fn} not found, skipping '{variant}'")

    return variants, nGrs, maxNHKLs, nScans


def load_omegas_and_hkls(topdir, nGrs, maxNHKLs):
    """Load omega angles and HKL counts per grain."""
    omegas = np.fromfile(os.path.join(topdir, f'omegas_{nGrs}_{maxNHKLs}.bin'),
                         dtype=np.double).reshape((nGrs, maxNHKLs))
    grainSpots = np.fromfile(os.path.join(topdir, f'nrHKLs_{nGrs}.bin'), dtype=np.int32)
    return omegas, grainSpots



# ──────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = MyParser(description='PF-HEDM Sinogram & Intensity Viewer',
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-paramFile', type=str, required=True,
                        help='Parameter file used for pf_MIDAS analysis')
    parser.add_argument('-resultDir', type=str, required=False, default='',
                        help='Results directory (defaults to cwd)')
    parser.add_argument('-portNr', type=int, required=False, default=8051,
                        help='Port number for the Dash server')
    parser.add_argument('-HostName', type=str, required=False, default='0.0.0.0',
                        help='Hostname for the Dash server')
    args, _ = parser.parse_known_args()

    topdir = args.resultDir if args.resultDir else os.getcwd()

    # --- Parse parameter file ---
    print("=" * 60)
    print("PF-HEDM Sinogram & Intensity Viewer")
    print("=" * 60)
    print(f"\nParameter file: {args.paramFile}")
    print(f"Result dir: {topdir}")
    print("\nParsing parameter file...")
    params = parse_param_file(args.paramFile)
    params['topdir'] = topdir

    # --- Load sinogram data ---
    print("\nLoading sinogram data...")
    sino_variants, nGrs, maxNHKLs, nScans = load_sinogram_variants(topdir)
    available_variants = list(sino_variants.keys())
    print(f"  {nGrs} grains, {maxNHKLs} max HKLs, {nScans} scans")
    print(f"  Variants: {available_variants}")

    # --- Load omegas and HKL counts ---
    print("\nLoading omega angles...")
    omegas, grainSpots = load_omegas_and_hkls(topdir, nGrs, maxNHKLs)

    # --- Load pre-extracted patches ---
    print("\nLoading pre-extracted patches...")
    patchesArr, spotPosArr = load_patches(topdir)

    # --- Load per-cell spot metadata ---
    print("\nLoading spot metadata...")
    spotMetaArr = load_spot_meta(topdir)

    # ── Dash App ─────────────────────────────────────────────
    external_stylesheets = [dbc.themes.CYBORG]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = "PF-HEDM Sinogram & Intensity Viewer"

    # --- Layout ---
    app.layout = dbc.Container([
        dcc.Store(id='store-sino-vmin', data=None),
        dcc.Store(id='store-sino-vmax', data=None),
        dcc.Store(id='store-patch-vmin', data=None),
        dcc.Store(id='store-patch-vmax', data=None),
        dcc.Interval(id='interval-play-row', interval=DEFAULT_REFRESH_MS,
                     n_intervals=0, disabled=True),
        dcc.Interval(id='interval-play-col', interval=DEFAULT_REFRESH_MS,
                     n_intervals=0, disabled=True),

        # Title
        dbc.Row([
            html.H3("PF-HEDM Sinogram & Intensity Viewer",
                     className="text-primary text-center mb-3 mt-2")
        ]),

        # Controls row
        dbc.Row([
            dbc.Col([
                dbc.Label("Grain Nr:"),
                dcc.Dropdown(
                    id='grain-dropdown',
                    options=[{'label': f'Grain {i} ({grainSpots[i]} spots)',
                              'value': i} for i in range(nGrs)],
                    value=0, clearable=False,
                    style={'color': '#000'}
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Sinogram Variant:"),
                dcc.Dropdown(
                    id='variant-dropdown',
                    options=[{'label': v, 'value': v} for v in available_variants],
                    value=available_variants[0] if available_variants else 'raw',
                    clearable=False,
                    style={'color': '#000'}
                ),
            ], width=2),
            dbc.Col([
                dbc.Label("Patch Half-Size (px):"),
                dcc.Slider(id='patch-size-slider', min=5, max=50, step=1,
                           value=DEFAULT_PATCH_HALF,
                           marks={5: '5', 15: '15', 25: '25', 50: '50'},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=3),
            dbc.Col([
                dbc.Label("HKL Nr:"),
                dcc.Slider(id='row-slider', min=0,
                           max=max(0, int(grainSpots[0]) - 1) if nGrs > 0 else 0,
                           step=1, value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=2),
            dbc.Col([
                dbc.Label("Scan Nr:"),
                dcc.Slider(id='col-slider', min=0, max=max(0, nScans - 1),
                           step=1, value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=2),
        ], className="mb-2"),

        # Plots
        dbc.Row([
            dbc.Col([
                dcc.Loading(type="circle", children=[
                    dcc.Graph(id='sinogram-plot', figure=go.Figure())
                ]),
            ], width=6),
            dbc.Col([
                dcc.Loading(type="circle", children=[
                    dcc.Graph(id='patch-plot', figure=go.Figure())
                ]),
            ], width=6),
        ]),

        html.Hr(),

        # Animation & Scale controls
        dbc.Row([
            # Left: Play controls
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Button("▶ Play Scans (fixed spot)",
                                   id='btn-play-row', color='success',
                                   size='sm', className='me-1'),
                        dbc.Button("■ Stop", id='btn-stop-row',
                                   color='danger', size='sm'),
                    ], width=6),
                    dbc.Col([
                        dbc.Button("▶ Play Spots (fixed scan)",
                                   id='btn-play-col', color='success',
                                   size='sm', className='me-1'),
                        dbc.Button("■ Stop", id='btn-stop-col',
                                   color='danger', size='sm'),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Refresh (ms):", className='mt-2'),
                        dcc.Input(id='refresh-ms-input', type='number',
                                  value=DEFAULT_REFRESH_MS, min=100, max=5000,
                                  step=50, style={'width': '100px', 'color': '#000'}),
                    ], width=4),
                ], className='mt-2'),
            ], width=6),

            # Right: Scale controls
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Sinogram Scale:"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Min"),
                            dbc.Input(id='sino-vmin', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.InputGroupText("Max"),
                            dbc.Input(id='sino-vmax', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.Button("Apply", id='btn-sino-scale',
                                       color='info', size='sm'),
                        ], size='sm'),
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Patch Scale:"),
                        dbc.InputGroup([
                            dbc.InputGroupText("Min"),
                            dbc.Input(id='patch-vmin', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.InputGroupText("Max"),
                            dbc.Input(id='patch-vmax', type='number',
                                      placeholder='auto', style={'color': '#000'}),
                            dbc.Button("Apply", id='btn-patch-scale',
                                       color='info', size='sm'),
                        ], size='sm'),
                    ], width=6),
                ]),
            ], width=6),
        ]),
    ], fluid=True)

    # ── Callbacks ────────────────────────────────────────────

    # --- Click on sinogram → update row/col sliders ---
    @callback(
        Output('row-slider', 'value', allow_duplicate=True),
        Output('col-slider', 'value', allow_duplicate=True),
        Input('sinogram-plot', 'clickData'),
        prevent_initial_call=True
    )
    def sinogram_click(clickData):
        if not clickData or not clickData.get('points'):
            return no_update, no_update
        pt = clickData['points'][0]
        # Heatmap with integer y-values: x=scanNr, y=spotNr (HKL index)
        col = int(pt.get('x', 0))   # scanNr
        row = int(pt.get('y', 0))   # spotNr (HKL index)
        print(f"  [CLICK] row(HKL)={row}, col(scan)={col}")
        return row, col

    # --- Scale apply buttons → update stores ---
    @callback(
        Output('store-sino-vmin', 'data'),
        Output('store-sino-vmax', 'data'),
        Input('btn-sino-scale', 'n_clicks'),
        State('sino-vmin', 'value'),
        State('sino-vmax', 'value'),
        prevent_initial_call=True
    )
    def apply_sino_scale(n, vmin, vmax):
        return vmin, vmax

    @callback(
        Output('store-patch-vmin', 'data'),
        Output('store-patch-vmax', 'data'),
        Input('btn-patch-scale', 'n_clicks'),
        State('patch-vmin', 'value'),
        State('patch-vmax', 'value'),
        prevent_initial_call=True
    )
    def apply_patch_scale(n, vmin, vmax):
        return vmin, vmax

    # --- Play/Stop controls ---
    @callback(
        Output('interval-play-row', 'disabled'),
        Output('interval-play-row', 'interval'),
        Input('btn-play-row', 'n_clicks'),
        Input('btn-stop-row', 'n_clicks'),
        State('refresh-ms-input', 'value'),
        prevent_initial_call=True
    )
    def toggle_play_scans(play, stop, ms):
        if ctx.triggered_id == 'btn-play-row':
            return False, max(100, ms or DEFAULT_REFRESH_MS)
        return True, max(100, ms or DEFAULT_REFRESH_MS)

    @callback(
        Output('interval-play-col', 'disabled'),
        Output('interval-play-col', 'interval'),
        Input('btn-play-col', 'n_clicks'),
        Input('btn-stop-col', 'n_clicks'),
        State('refresh-ms-input', 'value'),
        prevent_initial_call=True
    )
    def toggle_play_spots(play, stop, ms):
        if ctx.triggered_id == 'btn-play-col':
            return False, max(100, ms or DEFAULT_REFRESH_MS)
        return True, max(100, ms or DEFAULT_REFRESH_MS)

    # --- Interval ticks → advance col-slider (play scans for fixed spot) ---
    @callback(
        Output('col-slider', 'value', allow_duplicate=True),
        Input('interval-play-row', 'n_intervals'),
        State('col-slider', 'value'),
        prevent_initial_call=True
    )
    def advance_scan(n, current_col):
        return (current_col + 1) % nScans

    # --- Interval ticks → advance row-slider (play spots for fixed scan) ---
    @callback(
        Output('row-slider', 'value', allow_duplicate=True),
        Input('interval-play-col', 'n_intervals'),
        State('row-slider', 'value'),
        State('grain-dropdown', 'value'),
        prevent_initial_call=True
    )
    def advance_spot(n, current_row, grainNr):
        nSp = int(grainSpots[grainNr]) if grainNr is not None and grainNr < len(grainSpots) else 1
        return (current_row + 1) % max(1, nSp)

    # --- Reset row slider max when grain changes ---
    @callback(
        Output('row-slider', 'max'),
        Output('row-slider', 'value', allow_duplicate=True),
        Input('grain-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_row_slider_max(grainNr):
        if grainNr is None:
            return maxNHKLs - 1, 0
        nSp = int(grainSpots[grainNr]) if grainNr < len(grainSpots) else 1
        return max(0, nSp - 1), 0

    # --- Main sinogram plot ---
    @callback(
        Output('sinogram-plot', 'figure'),
        Input('grain-dropdown', 'value'),
        Input('variant-dropdown', 'value'),
        Input('row-slider', 'value'),
        Input('col-slider', 'value'),
        Input('store-sino-vmin', 'data'),
        Input('store-sino-vmax', 'data'),
    )
    def update_sinogram(grainNr, variant, row, col, vmin, vmax):
        fig = go.Figure()
        if grainNr is None or variant not in sino_variants:
            fig.update_layout(title="No data", **COMMON_LAYOUT)
            return fig

        nSp = int(grainSpots[grainNr]) if grainNr < len(grainSpots) else 0
        if nSp == 0:
            fig.update_layout(title=f"Grain {grainNr}: no spots", **COMMON_LAYOUT)
            return fig

        sino_data = sino_variants[variant]
        sino = sino_data[grainNr, :nSp, :]  # shape: (nSp, nScans)
        theta_vals = omegas[grainNr, :nSp]

        # Y-axis = integer spot indices; omega shown via custom tick labels + hover
        y_indices = list(range(nSp))
        # Build omega tick labels for a subset of y positions
        tick_step = max(1, nSp // 20)
        tick_vals = list(range(0, nSp, tick_step))
        tick_text = [f'{theta_vals[i]:.1f}°' for i in tick_vals]

        # Custom hover text with omega values
        hover_text = []
        for si in range(nSp):
            row_texts = []
            for sc in range(nScans):
                parts = [f'Scan: {sc}', f'HKL: {si}',
                         f'ω: {theta_vals[si]:.2f}°',
                         f'I: {sino[si, sc]:.2f}']
                if spotMetaArr is not None:
                    eta = spotMetaArr[grainNr, si, sc, 0]
                    tth = spotMetaArr[grainNr, si, sc, 1]
                    yp  = spotMetaArr[grainNr, si, sc, 2]
                    zp  = spotMetaArr[grainNr, si, sc, 3]
                    if not np.isnan(eta):
                        parts.append(f'η: {eta:.2f}°')
                        parts.append(f'2θ: {tth:.2f}°')
                        parts.append(f'Y: {yp:.1f} px')
                        parts.append(f'Z: {zp:.1f} px')
                row_texts.append('<br>'.join(parts))
            hover_text.append(row_texts)

        fig.add_trace(go.Heatmap(
            z=sino,
            x=list(range(nScans)),
            y=y_indices,
            colorscale='Viridis',
            zmin=vmin, zmax=vmax,
            colorbar=dict(title='Intensity'),
            hoverinfo='text',
            text=hover_text,
        ))

        # Crosshair at current selection
        if 0 <= row < nSp and 0 <= col < nScans:
            # Vertical line at col
            fig.add_shape(
                type='line', x0=col, x1=col,
                y0=-0.5, y1=nSp - 0.5,
                line=dict(color='red', width=1, dash='dash'),
            )
            # Horizontal line at row
            fig.add_shape(
                type='line', x0=-0.5, x1=nScans - 0.5,
                y0=row, y1=row,
                line=dict(color='red', width=1, dash='dash'),
            )
            # Marker at intersection
            fig.add_trace(go.Scatter(
                x=[col], y=[row],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.update_layout(
            title=f'Sinogram: Grain {grainNr} ({variant}) — {nSp} spots',
            xaxis_title='Scan Nr',
            yaxis_title='HKL Nr',
            yaxis=dict(tickvals=tick_vals, ticktext=tick_text),
            clickmode='event',
            **COMMON_LAYOUT,
        )
        return fig

    # --- Main intensity patch plot ---
    @callback(
        Output('patch-plot', 'figure'),
        Input('grain-dropdown', 'value'),
        Input('row-slider', 'value'),
        Input('col-slider', 'value'),
        Input('patch-size-slider', 'value'),
        Input('store-patch-vmin', 'data'),
        Input('store-patch-vmax', 'data'),
    )
    def update_patch(grainNr, row, col, patchHalf, vmin, vmax):
        fig = go.Figure()
        if grainNr is None:
            fig.update_layout(title="Select a grain", **COMMON_LAYOUT)
            return fig

        nSp = int(grainSpots[grainNr]) if grainNr < len(grainSpots) else 0
        if nSp == 0 or row >= nSp:
            fig.update_layout(
                title=f"Spot {row} out of range (grain has {nSp} spots)",
                **COMMON_LAYOUT)
            return fig

        theta_val = omegas[grainNr, row]

        # Direct lookup from pre-extracted patches
        if patchesArr is None:
            fig.update_layout(
                title='No patches data — run findSingleSolutionPFRefactored first',
                **COMMON_LAYOUT)
            return fig

        patch = patchesArr[grainNr, row, col]
        print(f"  [PATCH] grain={grainNr} row={row} col={col} max={patch.max():.2f} sum={patch.sum():.2f}")
        if np.all(patch == 0):
            fig.update_layout(
                title=f'No data: Grain {grainNr}, Spot {row}, Scan {col}',
                **COMMON_LAYOUT)
            return fig

        fig.add_trace(go.Heatmap(
            z=patch,
            colorscale='Viridis',
            zmin=vmin, zmax=vmax,
            colorbar=dict(title='Intensity'),
            hovertemplate=(
                'Y: %{x}<br>'
                'Z: %{y}<br>'
                'Intensity: %{z:.1f}<extra></extra>'
            )
        ))

        fig.update_layout(
            title=(f'Intensity: Grain {grainNr}, Spot {row} '
                   f'(θ={theta_val:.1f}°), Scan {col}'),
            xaxis_title='Y (pixels)',
            yaxis_title='Z (pixels)',
            yaxis=dict(scaleanchor='x'),
            **COMMON_LAYOUT,
        )
        return fig

    # ── Run ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Starting server: http://{args.HostName}:{args.portNr}")
    print(f"{'=' * 60}\n")
    app.run(port=args.portNr, host=args.HostName, debug=False)
