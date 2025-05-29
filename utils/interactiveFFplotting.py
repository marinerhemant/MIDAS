#!/usr/bin/env python

from dash import Dash, html, dcc, callback, Output, Input, State, no_update, dash_table
from zarr.storage import ZipStore
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import argparse
import sys
import zarr
from math import cos, sin, sqrt
import traceback

# --- Constants ---
deg2rad = np.pi / 180
rad2deg = 180 / np.pi

# Grain File Columns
GRAIN_ID_COL = 0
GRAIN_X_COL = 10; GRAIN_Y_COL = 11; GRAIN_Z_COL = 12
GRAIN_SIZE_COL = 22; GRAIN_COMPLETENESS_COL = 23; GRAIN_ERROR_COL = 19
GRAIN_EULER_0_COL = -3; GRAIN_EULER_1_COL = -2; GRAIN_EULER_2_COL = -1
GRAIN_STRAIN_ERROR_COL = -5

# SpotMatrix File Columns
SPOT_GRAIN_ID_COL = 0; SPOT_ID_COL = 1; SPOT_OMEGA_COL = 2
SPOT_DET_Y_PX_COL = 3; SPOT_DET_Z_PX_COL = 4; SPOT_OMEGA_RAW_COL = 5
SPOT_ETA_COL = 6; SPOT_RING_NR_COL = 7; SPOT_Y_M_COL = 8; SPOT_Z_M_COL = 9
SPOT_TTHETA_COL = 10; SPOT_STRAIN_COL = 11

# InputAll File Columns (Spots Original Info) - Assuming 0-based index
SPOT_ORIG_SIZE_COL = 3

# Volume Plot Settings
VOLUME_WINDOW_PX = 10
VOLUME_WINDOW_FRAME = 7
# Plotting Settings
COMMON_LAYOUT_SETTINGS = dict(margin=dict(l=0, r=0, b=0, t=50), height=700, template="plotly_white") # Use white template
DEFAULT_CONTINUOUS_COLOR_SCALE = 'Viridis'
MAX_MARKER_SIZE_VISUAL = 10
MIN_MARKER_SIZE_VISUAL = 2
DEFAULT_MARKER_SIZE = 4

SPOT_TABLE_COLUMNS = ['spotID', 'ringNr', 'omega', 'tTheta', 'eta', 'strain', 'ds', 'spotSize', 'y', 'z', 'omeRaw', 'detY', 'detZ']
DEFAULT_TABLE_PAGE_SIZE = 10

# --- Helper Functions ---
class MyParser(argparse.ArgumentParser):
    def error(self, message): sys.stderr.write('error: %s\n' % message); self.print_help(); sys.exit(2)

def rotateAroundZ(v, ome_rad):
    cos_o, sin_o = cos(ome_rad), sin(ome_rad); m = [[cos_o, -sin_o, 0], [sin_o,  cos_o, 0], [0, 0, 1]]
    r0 = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2]; r1 = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2]; r2 = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
    return r0, r1, r2

def spot2gv(det_dist, det_y_m, det_z_m, omega_deg):
    v = np.array([det_dist, det_y_m, det_z_m]); v_norm = np.linalg.norm(v)
    if v_norm < 1e-9: return 0, 0, 0
    vn = v / v_norm; g1r = vn[0] - 1; g2r = vn[1]; g3r = vn[2]; g_unrotated = [g1r, g2r, g3r]
    ome_rad = -omega_deg * deg2rad; return rotateAroundZ(g_unrotated, ome_rad)

# --- Data Loading Functions ---
def load_zarr_params(zarr_file):
    params = {}
    try:
        store = ZipStore(zarr_file, mode='r')
        zf = zarr.open_group(store=store, mode='r')
        params['pixSz'] = zf['/analysis/process/analysis_parameters/PixelSize'][0] # Assumed MICRONS/pixel
        params['wl'] = zf['/analysis/process/analysis_parameters/Wavelength'][0]
        params['Lsd'] = zf['/analysis/process/analysis_parameters/Lsd'][0]
        params['thresh'] = zf['/analysis/process/analysis_parameters/RingThresh'][0][1]
        params['omegaStep'] = zf['/measurement/process/scan_parameters/step'][0]
        params['omegaStart'] = zf['/measurement/process/scan_parameters/start'][0]
        params['rawDataLink'] = zf['exchange/data']
        params['ImTransOpt'] = zf['/analysis/process/analysis_parameters/ImTransOpt'][:]
        params['dark'] = zf['exchange/dark'][:]
        params['darkMean'] = np.mean(params['dark'], axis=0).astype(np.double)
        params['nFrames'], params['nPxZ'], params['nPxY'] = params['rawDataLink'].shape
        print(f"Zarr parameters loaded. Pixel Size: {params['pixSz']} microns/pixel (ASSUMED).")
        return params
    except Exception as e: print(f"Error loading Zarr parameters from {zarr_file}: {e}", file=sys.stderr); traceback.print_exc(); return None

def load_spot_data(spots_file, spots_orig_file, zarr_params):
    if zarr_params is None: return None
    try: spots = np.genfromtxt(spots_file, skip_header=1)
    except Exception as e: print(f"Error reading {spots_file}: {e}", file=sys.stderr); traceback.print_exc(); return None
    try: spotsOrig = np.genfromtxt(spots_orig_file, skip_header=1)
    except Exception as e: print(f"Warning/Error reading {spots_orig_file}: {e}. Spot size info may be missing.", file=sys.stderr); traceback.print_exc(); spotsOrig = None

    data = {k: [] for k in ['omega', 'y', 'z', 'g1', 'g2', 'g3', 'ringNr', 'ringNrInt', 'strain', 'ds', 'grainID', 'grainIDColor', 'spotID', 'detY', 'detZ', 'spotSize', 'detHor', 'detVer', 'omeRaw', 'eta', 'tTheta']}
    pixSz, Lsd, wl = zarr_params['pixSz'], zarr_params['Lsd'], zarr_params['wl']

    for i in range(spots.shape[0]):
        if spots[i][SPOT_OMEGA_RAW_COL] == 0.0 and spots[i][SPOT_OMEGA_COL] == 0.0: continue
        omega_val, tTheta_val_deg = spots[i][SPOT_OMEGA_COL], spots[i][SPOT_TTHETA_COL]
        tTheta_val_deg *= 2 # Note: Assuming input is Theta
        spot_y_microns, spot_z_microns = spots[i][SPOT_Y_M_COL], spots[i][SPOT_Z_M_COL]
        data['y'].append(spot_y_microns / pixSz); data['z'].append(spot_z_microns / pixSz); data['omega'].append(omega_val)
        ring_nr_int = int(spots[i][SPOT_RING_NR_COL]); data['ringNr'].append(str(ring_nr_int)); data['ringNrInt'].append(ring_nr_int)
        grain_id_val, spot_id_val = spots[i][SPOT_GRAIN_ID_COL], spots[i][SPOT_ID_COL]
        data['grainID'].append(grain_id_val); data['grainIDColor'].append(str(int(grain_id_val))); data['spotID'].append(spot_id_val)
        data['detY'].append(int(spots[i][SPOT_DET_Y_PX_COL])); data['detZ'].append(int(spots[i][SPOT_DET_Z_PX_COL]))
        data['detHor'].append(int(spots[i][SPOT_DET_Y_PX_COL])); data['detVer'].append(int(spots[i][SPOT_DET_Z_PX_COL]))
        data['omeRaw'].append(spots[i][SPOT_OMEGA_RAW_COL]); data['eta'].append(spots[i][SPOT_ETA_COL]); data['tTheta'].append(tTheta_val_deg)
        spot_size_val = np.nan
        if spotsOrig is not None:
            spot_id_int = int(spot_id_val)
            if 1 <= spot_id_int <= spotsOrig.shape[0]:
                try: raw_size = spotsOrig[spot_id_int - 1, SPOT_ORIG_SIZE_COL]; spot_size_val = raw_size if np.isfinite(raw_size) else np.nan
                except IndexError: pass
        data['spotSize'].append(spot_size_val)
        data['strain'].append(1e6 * np.abs(spots[i][SPOT_STRAIN_COL]))
        sin_tTheta = sin(tTheta_val_deg * deg2rad / 2.0)
        ds_val = wl / (2 * sin_tTheta) if abs(sin_tTheta) > 1e-9 else np.nan; data['ds'].append(ds_val)
        g1_raw, g2_raw, g3_raw = spot2gv(Lsd, spot_y_microns * 1e-6, spot_z_microns * 1e-6, omega_val)
        g_mag_raw = sqrt(g1_raw**2 + g2_raw**2 + g3_raw**2)
        if g_mag_raw > 1e-9 and not np.isnan(ds_val) and ds_val > 1e-9:
             scale_factor = (1.0 / ds_val) / g_mag_raw; g1, g2, g3 = g1_raw * scale_factor, g2_raw * scale_factor, g3_raw * scale_factor
        else: g1, g2, g3 = np.nan, np.nan, np.nan
        data['g1'].append(g1); data['g2'].append(g2); data['g3'].append(g3)

    try:
        spots_df = pd.DataFrame(data)
        spots_df.dropna(subset=['g1', 'g2', 'g3'], inplace=True)
        num_cols = ['spotSize', 'strain', 'ds', 'omega', 'y', 'z', 'g1', 'g2', 'g3', 'tTheta', 'eta', 'omeRaw']
        for col in num_cols: spots_df[col] = pd.to_numeric(spots_df[col], errors='coerce')
        id_cols = {'grainID': 'Int64', 'spotID': 'Int64', 'ringNrInt': 'Int64', 'detY':'Int64', 'detZ':'Int64', 'detHor':'Int64', 'detVer':'Int64'}
        for col, dtype in id_cols.items():
            if col in spots_df.columns: spots_df[col] = pd.to_numeric(spots_df[col], errors='coerce').astype(dtype)
        str_cols = ['grainIDColor', 'ringNr']; spots_df[str_cols] = spots_df[str_cols].astype(str)
        spots_df = spots_df.sort_values(by=['grainID', 'spotID'])
        print(f"Created DataFrame with {len(spots_df)} spots after processing.")
        return spots_df
    except Exception as e: print(f"Error creating Spot DataFrame: {e}", file=sys.stderr); traceback.print_exc(); return None

def load_grain_data(grains_file):
    """Loads grain data from CSV file, keeping numeric columns numeric."""
    try: grains = np.genfromtxt(grains_file, skip_header=9)
    except Exception as e: print(f"Error reading {grains_file}: {e}", file=sys.stderr); traceback.print_exc(); return None
    if grains.ndim == 1: grains = grains.reshape(1, -1)
    elif grains.shape[0] == 0: print("Warning: Grains file is empty."); return pd.DataFrame({'x': [], 'y': [], 'z': [], 'GrainSize': [], 'Confidence': [], 'ID': [], 'Euler0': [], 'Euler1': [], 'Euler2': [], 'StrainError': [], 'IDColor': [], 'Error': [], 'RawGrainSize': []})

    data2 = {k: [] for k in ['x', 'y', 'z', 'GrainSize', 'RawGrainSize', 'Confidence', 'ID', 'IDColor', 'Euler0', 'Euler1', 'Euler2', 'Error', 'StrainError']}
    valid_size_mask = np.full(grains.shape[0], False)
    if grains.shape[1] > GRAIN_SIZE_COL: valid_size_mask = np.isfinite(grains[:, GRAIN_SIZE_COL]) & (grains[:, GRAIN_SIZE_COL] > 0)
    largestSize = np.max(grains[valid_size_mask, GRAIN_SIZE_COL]) if np.any(valid_size_mask) else 1.0
    if largestSize <= 0: largestSize = 1.0

    for i in range(grains.shape[0]):
        data2['x'].append(grains[i, GRAIN_X_COL]); data2['y'].append(grains[i, GRAIN_Y_COL]); data2['z'].append(grains[i, GRAIN_Z_COL])
        raw_size = grains[i, GRAIN_SIZE_COL] if grains.shape[1] > GRAIN_SIZE_COL and np.isfinite(grains[i, GRAIN_SIZE_COL]) else 0
        data2['RawGrainSize'].append(raw_size)
        target_max_grain_visual_size = MAX_MARKER_SIZE_VISUAL * 2 # Grains can be larger
        scaled_size = max(MIN_MARKER_SIZE_VISUAL, target_max_grain_visual_size * raw_size / largestSize) if raw_size > 0 and largestSize > 0 else MIN_MARKER_SIZE_VISUAL
        data2['GrainSize'].append(scaled_size) # This 'GrainSize' is the scaled visual size
        data2['Confidence'].append(grains[i, GRAIN_COMPLETENESS_COL]) # Keep raw Confidence
        data2['Euler0'].append(grains[i, GRAIN_EULER_0_COL]); data2['Euler1'].append(grains[i, GRAIN_EULER_1_COL]); data2['Euler2'].append(grains[i, GRAIN_EULER_2_COL])
        data2['StrainError'].append(grains[i, GRAIN_STRAIN_ERROR_COL]); data2['Error'].append(grains[i, GRAIN_ERROR_COL])
        grain_id = grains[i, GRAIN_ID_COL]; data2['ID'].append(grain_id); data2['IDColor'].append(f'{int(grain_id)}')

    try:
        grains_df = pd.DataFrame(data2)
        num_cols = ['x', 'y', 'z', 'GrainSize', 'RawGrainSize', 'Confidence', 'Euler0', 'Euler1', 'Euler2', 'Error', 'StrainError']
        for col in num_cols: grains_df[col] = pd.to_numeric(grains_df[col], errors='coerce')
        grains_df['ID'] = pd.to_numeric(grains_df['ID'], errors='coerce').astype('Int64')
        grains_df['IDColor'] = grains_df['IDColor'].astype(str)
        grains_df = grains_df.sort_values(by=['ID'])
        print(f"Processed {len(data2['ID'])} grains.")
        return grains_df
    except Exception as e: print(f"Error creating Grain DataFrame: {e}", file=sys.stderr); traceback.print_exc(); return None

# --- Main Application Setup ---
if __name__ == '__main__':
    parser = MyParser(description='''MIDAS FF Interactive Plotter''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-resultFolder', type=str, required=True, help='Folder where the reconstruction exists')
    parser.add_argument('-dataFileName', type=str, required=True, help='Path to the input Zarr datafile')
    parser.add_argument('-HostName', type=str, required=False, default="0.0.0.0", help='HostName IP')
    parser.add_argument('-portNr', type=int, required=False, default=8050, help='Port number')
    args, unparsed = parser.parse_known_args()
    resultDir, dataFile, hn, portNr = args.resultFolder, args.dataFileName, args.HostName, args.portNr

    print("Loading Zarr parameters..."); zarr_params = load_zarr_params(dataFile)
    if zarr_params is None: sys.exit("Failed Zarr load.")
    print("Loading spot data..."); spots_df = load_spot_data(resultDir + '/SpotMatrix.csv', resultDir + '/InputAll.csv', zarr_params)
    if spots_df is None: sys.exit("Failed spot load.")
    print("Loading grain data..."); grains_df = load_grain_data(resultDir + '/Grains.csv')
    if grains_df is None: sys.exit("Failed grain load.")

    initial_grain_id = grains_df['ID'].iloc[0] if not grains_df.empty else None
    # Determine slider ranges from data
    eta_min, eta_max, tth_min, tth_max, ome_min, ome_max = -180.0, 180.0, 0.0, 90.0, -180.0, 180.0
    slider_step = 0.1 # Step for tTheta slider
    omega_step = 1.0 # Step for omega slider
    if not spots_df.empty:
        eta_min_data = spots_df['eta'].dropna().min(); eta_max_data = spots_df['eta'].dropna().max()
        tth_min_data = spots_df['tTheta'].dropna().min(); tth_max_data = spots_df['tTheta'].dropna().max()
        ome_min_data = spots_df['omega'].dropna().min(); ome_max_data = spots_df['omega'].dropna().max()

        if pd.notna(eta_min_data) and pd.notna(eta_max_data): eta_min, eta_max = np.floor(eta_min_data), np.ceil(eta_max_data); eta_max = max(eta_max, eta_min + 1)
        if pd.notna(tth_min_data) and pd.notna(tth_max_data): tth_min, tth_max = np.floor(tth_min_data * 10) / 10, np.ceil(tth_max_data * 10) / 10; tth_max = max(tth_max, tth_min + slider_step)
        if pd.notna(ome_min_data) and pd.notna(ome_max_data): ome_min, ome_max = np.floor(ome_min_data), np.ceil(ome_max_data); ome_max = max(ome_max, ome_min + omega_step)

        unique_ring_nrs_int = sorted(spots_df['ringNrInt'].dropna().astype(int).unique())
        ring_nr_options = [{"label": str(nr), "value": nr} for nr in unique_ring_nrs_int]
        initial_ring_nr_values = unique_ring_nrs_int
    else: ring_nr_options, initial_ring_nr_values = [], []

    external_stylesheets = [dbc.themes.CYBORG] # Use Cyborg theme
    app = Dash(__name__, external_stylesheets=external_stylesheets); app.title = "MIDAS FF-HEDM Interactive Viewer"

    # --- App Layout ---
    app.layout = dbc.Container([
        dcc.Store(id='selected-grain-id-store', data=initial_grain_id),
        dbc.Row([html.Div('MIDAS FF-HEDM Interactive Viewer', className="text-primary text-center fs-3 mb-2")]),
        dbc.Row([
            dbc.Col([dbc.Label("Spot Color (3D View):"), dbc.RadioItems(options=[{"label": x, "value": x} for x in ['ringNr', 'grainIDColor','strain','spotSize']], value='grainIDColor', inline=True, id='radio-buttons-spots')], width=6),
            dbc.Col([dbc.Label("G-Vector Color (Reciprocal Space):"), dbc.RadioItems(options=[{"label": x, "value": x} for x in ['ringNr', 'grainIDColor','strain','ds']], value='grainIDColor', inline=True, id='radio-buttons-spots_polar')], width=6),
        ]),
        dbc.Row([
             dbc.Col([dbc.Label("Select Rings (3D View):")],width=2), dbc.Col([dcc.Checklist(id="checklist_spots", options=ring_nr_options, value=initial_ring_nr_values, inline=True)],width=4),
             dbc.Col([dbc.Label("Select Rings (Reciprocal Space):")],width=2), dbc.Col([dcc.Checklist(id="checklist_spots_polar", options=ring_nr_options, value=initial_ring_nr_values, inline=True)],width=4),
        ]),
        dbc.Row([
            dbc.Col([dcc.Loading(id="loading-spots", type="circle", children=[dcc.Graph(figure=go.Figure(), id='spots')])], width=6),
            dbc.Col([dcc.Loading(id="loading-spots-polar", type="circle", children=[dcc.Graph(figure=go.Figure(), id='spots_polar')])], width=6),
        ]), html.Hr(),
        # --- Grain Filtering Controls ---
        dbc.Row([
            dbc.Col([
                dbc.Label("Filter Grains by Spot Properties:", className="fw-bold"),
                dbc.Row([dbc.Col(dbc.Label("Eta Range (°):"), width=3, className="text-end"), dbc.Col(dcc.RangeSlider(id='eta-range-slider', min=eta_min, max=eta_max, step=1, value=[eta_min, eta_max], marks=None, tooltip={"placement": "bottom", "always_visible": True}), width=9)]),
                dbc.Row([dbc.Col(dbc.Label("2θ Range (°):"), width=3, className="text-end"), dbc.Col(dcc.RangeSlider(id='tth-range-slider', min=tth_min, max=tth_max, step=slider_step, value=[tth_min, tth_max], marks=None, tooltip={"placement": "bottom", "always_visible": True}), width=9)]),
                dbc.Row([dbc.Col(dbc.Label("Omega Range (°):"), width=3, className="text-end"), dbc.Col(dcc.RangeSlider(id='omega-range-slider', min=ome_min, max=ome_max, step=omega_step, value=[ome_min, ome_max], marks=None, tooltip={"placement": "bottom", "always_visible": True}), width=9)]), # <-- NEW Omega Slider
                dbc.Row([dbc.Col(dbc.Label("Show Grain ID:"), width=3, className="text-end"), dbc.Col(dcc.Input(id='grain-id-input', type='number', placeholder="Enter Grain ID (optional)", debounce=True, style={'color': '#000'}), width=9)]), # <-- NEW Grain ID Input
                html.Hr(), dbc.Label("Grain Color (3D Map):"), dbc.RadioItems(options=[{"label": x, "value": x} for x in ['Confidence', 'GrainSize', 'IDColor', 'Error','Euler0','Euler1','Euler2','StrainError']], value='IDColor', inline=True, id='radio-buttons-grains')
            ], width=6),
            dbc.Col([html.Div(style={'height': '155px'}), html.Hr(), dbc.Label("Filtered Spot Color (per Grain):"), dbc.RadioItems(options=[{"label": x, "value": x} for x in ['ringNr', 'grainIDColor','strain','spotSize', 'tTheta', 'eta', 'ds']], value='strain', inline=True, id='radio-buttons-spots_filtered')], width=6), # Adjusted height for alignment
        ]),
        # --- Plots Below Filters ---
        dbc.Row([
            dbc.Col([dcc.Loading(id="loading-grains", type="circle", children=[dcc.Graph(figure=go.Figure(), id='grains', clickData={'points':[{'customdata': [initial_grain_id]}]} if initial_grain_id is not None else None)])], width=6),
            dbc.Col([dcc.Loading(id="loading-filtered-spots", type="circle", children=[dcc.Graph(figure=go.Figure(), id='filtered_spots', clickData=None)])], width=6),
        ]),
        dbc.Row([
            dbc.Col([dcc.Loading(id="loading-filtered-spots-2d", type="circle", children=[dcc.Graph(figure=go.Figure(), id='filtered_spots_2d', clickData=None)])], width=6),
            dbc.Col([dcc.Loading(id="loading-image-data", type="cube", children=[dcc.Graph(figure=go.Figure(), id='image_data')])], width=6),
        ]), html.Hr(),
        dbc.Row([
             dbc.Col([ dbc.Label("Table Rows:", html_for="page-size-dropdown"), dcc.Dropdown(id='page-size-dropdown', options=[{'label': str(s), 'value': s} for s in [10, 25, 50, 100]] + [{'label': 'All', 'value': 99999}], value=DEFAULT_TABLE_PAGE_SIZE, clearable=False, style={'color': '#000'}) ], width={"size": 2, "offset": 10}),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.H4("Spot Details for Selected Grain", className="text-center"),
                dcc.Loading(id="loading-spot-table", type="circle", children=[
                    dash_table.DataTable(id='spot-detail-table', columns=[{"name": i, "id": i} for i in SPOT_TABLE_COLUMNS], data=[], page_size=DEFAULT_TABLE_PAGE_SIZE, style_table={'overflowX': 'auto'}, style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold'}, style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}, style_cell={'textAlign': 'left', 'minWidth': '80px', 'width': '100px', 'maxWidth': '150px', 'border': '1px solid grey'}, style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(60, 60, 60)'}], sort_action="native", filter_action="native", filter_options={"case": "insensitive"}, page_action='native')
                ])
            ], width=12)
        ])
    ], fluid=True)


    # --- Callbacks ---

    # --- Grain Plot Callback --- Using go.Scatter3d --- Updated with Omega and Grain ID filters
    @callback(
        Output('grains', 'figure'),
        Input('radio-buttons-grains', 'value'),
        Input('eta-range-slider', 'value'),
        Input('tth-range-slider', 'value'),
        Input('omega-range-slider', 'value'), # <-- NEW Input
        Input('grain-id-input', 'value')      # <-- NEW Input
    )
    def update_grains_plot(grain_color_choice, eta_range, tth_range, omega_range, specific_grain_id):
        fig = go.Figure()
        if grains_df is None or grains_df.empty or spots_df is None or spots_df.empty: return fig.update_layout(title="Grains/Spots data not loaded", **COMMON_LAYOUT_SETTINGS)

        try:
            # --- Filter spots based on ALL range sliders ---
            spots_filtered_by_range = spots_df[
                (spots_df['eta'] >= eta_range[0]) & (spots_df['eta'] <= eta_range[1]) &
                (spots_df['tTheta'] >= tth_range[0]) & (spots_df['tTheta'] <= tth_range[1]) &
                (spots_df['omega'] >= omega_range[0]) & (spots_df['omega'] <= omega_range[1]) # <-- Added Omega filter
            ]

            valid_grain_ids = spots_filtered_by_range['grainID'].dropna().unique()
            if len(valid_grain_ids) == 0: return fig.update_layout(title=f"No grains with spots in selected ranges", **COMMON_LAYOUT_SETTINGS)

            grains_to_plot = grains_df[grains_df['ID'].isin(valid_grain_ids)].copy()

            # --- Apply Specific Grain ID Filter (if provided and valid) ---
            grain_id_to_show = None
            if specific_grain_id is not None:
                try:
                    grain_id_to_show = int(specific_grain_id)
                    # Filter the already range-filtered grains
                    grains_to_plot = grains_to_plot[grains_to_plot['ID'] == grain_id_to_show]
                    if grains_to_plot.empty:
                         return fig.update_layout(title=f"Grain ID {grain_id_to_show} has no spots in selected ranges", **COMMON_LAYOUT_SETTINGS)
                except (ValueError, TypeError):
                    print(f"Invalid Grain ID input: {specific_grain_id}. Ignoring ID filter.")
                    grain_id_to_show = None # Reset if invalid input

            # --- Proceed with plotting the filtered grains ---
            if grains_to_plot.empty:
                return fig.update_layout(title="No grains found matching all filters", **COMMON_LAYOUT_SETTINGS)

            numeric_cols_grain = ['x', 'y', 'z', 'GrainSize', 'RawGrainSize', 'Confidence', 'Euler0', 'Euler1', 'Euler2', 'Error', 'StrainError']
            for col in numeric_cols_grain:
                if col in grains_to_plot.columns: grains_to_plot.loc[:, col] = pd.to_numeric(grains_to_plot[col], errors='coerce')

            if grain_color_choice not in grains_to_plot.columns: grain_color_choice = 'IDColor'
            if 'IDColor' not in grains_to_plot.columns: return fig.update_layout(title="Grains data missing IDColor", **COMMON_LAYOUT_SETTINGS)

            required_cols = ['x', 'y', 'z', 'ID', 'IDColor', 'GrainSize', 'RawGrainSize', 'Confidence', 'Euler0', 'Euler1', 'Euler2', 'Error', 'StrainError']
            if grain_color_choice != 'IDColor' and grain_color_choice in grains_to_plot.columns: required_cols.append(grain_color_choice)

            grains_df_cleaned = grains_to_plot.dropna(subset=['x','y','z','ID','IDColor','GrainSize']).copy()
            if grains_df_cleaned.empty: return fig.update_layout(title="No valid grains to plot (filtered)", **COMMON_LAYOUT_SETTINGS)

            # --- Update Title based on filters ---
            title_prefix = "Filtered Grains"
            if grain_id_to_show is not None:
                title_prefix = f"Grain ID: {grain_id_to_show}"
            plot_title = f'{title_prefix} ({len(grains_df_cleaned)} shown, Color: {grain_color_choice})'

            # --- Plotting Logic (same as before) ---
            marker_props = dict(size=grains_df_cleaned['GrainSize'], sizemode='diameter', opacity=0.8) # Use pre-scaled visual size
            layout_updates = {'title': plot_title, 'scene': dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), 'showlegend': False, **COMMON_LAYOUT_SETTINGS}
            is_continuous_color = False # Initialize before the check
            color_col_actual = grain_color_choice

            if color_col_actual in numeric_cols_grain and color_col_actual in grains_df_cleaned.columns:
                 if not grains_df_cleaned[color_col_actual].isnull().all():
                      is_continuous_color = True; marker_props['color'] = grains_df_cleaned[color_col_actual]; marker_props['colorscale'] = DEFAULT_CONTINUOUS_COLOR_SCALE
                      marker_props['colorbar'] = dict(title=color_col_actual)
                 else: color_col_actual = 'IDColor' # Fallback if all NaN
            elif color_col_actual not in ['IDColor']: color_col_actual = 'IDColor' # Fallback if col missing/invalid

            if not is_continuous_color:
                 unique_cats = grains_df_cleaned[color_col_actual].astype(str).unique(); cmap = {c: px.colors.qualitative.Plotly[i%len(px.colors.qualitative.Plotly)] for i,c in enumerate(unique_cats)}; marker_props['color'] = grains_df_cleaned[color_col_actual].astype(str).map(cmap)

            hover_cols_go = ['ID', 'RawGrainSize', 'Confidence', 'Euler0', 'Euler1', 'Euler2', 'Error', 'StrainError']
            customdata_list = []; hovertemplate_parts = ["<b>Grain Info</b><br>"]
            for i, col in enumerate(hover_cols_go):
                 if col in grains_df_cleaned.columns:
                      col_data = grains_df_cleaned[col]; customdata_list.append(col_data)
                      fmt = ".1f" if col=='RawGrainSize' else (".3f" if col=='Confidence' else (".0f" if col=='ID' else (".1f" if col=='StrainError' else (".3f" if col=='Error' else (".2f" if col in ['Euler0','Euler1','Euler2'] else "")))))
                      hovertemplate_parts.append(f"{col}: %{{customdata[{i}]:{fmt}}}<br>")
                 else: customdata_list.append(pd.Series(["N/A"] * len(grains_df_cleaned), index=grains_df_cleaned.index)); hovertemplate_parts.append(f"{col}: N/A<br>")
            if is_continuous_color and grain_color_choice not in hover_cols_go:
                 if grain_color_choice in grains_df_cleaned.columns: col_data = grains_df_cleaned[grain_color_choice]; customdata_list.append(col_data); fmt = ".2f"; hovertemplate_parts.append(f"{grain_color_choice}: %{{customdata[{len(customdata_list)-1}]:{fmt}}}<br>")

            customdata_stacked = np.stack(customdata_list, axis=-1) if customdata_list else None
            hovertemplate_go = "".join(hovertemplate_parts) + "<extra></extra>"

            trace = go.Scatter3d( x=grains_df_cleaned['x'], y=grains_df_cleaned['y'], z=grains_df_cleaned['z'], mode='markers', marker=marker_props, customdata=customdata_stacked, hovertemplate=hovertemplate_go, ids=grains_df_cleaned['ID'].astype(str), name='')
            fig.add_trace(trace)
            fig.update_layout(clickmode='event+select')
            fig.update_layout(**layout_updates)

        except Exception as e: print(f"Error generating grains plot: {e}"); traceback.print_exc(); fig.update_layout(title='Error generating Grains plot', **COMMON_LAYOUT_SETTINGS)
        return fig


    # --- Filtered Spots Callbacks --- (No changes needed here for grain filtering)
    @callback(
        Output('filtered_spots', 'figure'), Output('selected-grain-id-store', 'data'),
        Input('grains', 'clickData'), Input('radio-buttons-spots_filtered', 'value'),
        prevent_initial_call=True
    )
    def update_filtered_spots_3d(clickData, spots_color_choice):
        fig = go.Figure(); selected_id = no_update
        if not (clickData and clickData['points']): return fig.update_layout(title="Click a grain", **COMMON_LAYOUT_SETTINGS), no_update
        try:
            clicked_id = None; point_data = clickData['points'][0]
            # Prioritize 'id' if available (set in go.Scatter3d trace)
            if 'id' in point_data and point_data['id'] is not None:
                 try: clicked_id = int(point_data['id'])
                 except (ValueError, TypeError): pass
            # Fallback to customdata if 'id' is missing or invalid
            if clicked_id is None and 'customdata' in point_data and isinstance(point_data['customdata'], (list, np.ndarray)) and len(point_data['customdata']) > 0:
                 try: clicked_id = int(point_data['customdata'][0])
                 except (ValueError, TypeError): pass

            if clicked_id is None: return fig.update_layout(title="Could not get Grain ID from click", **COMMON_LAYOUT_SETTINGS), no_update

            if grains_df['ID'].isin([clicked_id]).any():
                selected_id = clicked_id; dff_grain = spots_df[spots_df['grainID'] == selected_id].copy()
                numeric_cols = ['omega', 'y', 'z', 'spotSize', 'strain', 'tTheta', 'eta', 'ds', 'omeRaw']
                for col in numeric_cols:
                    if col in dff_grain.columns: dff_grain.loc[:, col] = pd.to_numeric(dff_grain[col], errors='coerce')
                required_cols = ['omega', 'y', 'z', 'spotID'] # Base required
                if spots_color_choice in numeric_cols and spots_color_choice in dff_grain.columns: required_cols.append(spots_color_choice)
                if 'spotSize' in dff_grain.columns: required_cols.append('spotSize')
                # Also include categorical color columns if chosen
                if spots_color_choice in ['grainIDColor', 'ringNr'] and spots_color_choice in dff_grain.columns: required_cols.append(spots_color_choice)

                dff_grain_cleaned = dff_grain.dropna(subset=[c for c in required_cols if c in dff_grain.columns]).copy()

                size_col = None # Handle size scaling
                if 'spotSize' in dff_grain_cleaned.columns and dff_grain_cleaned['spotSize'].notna().any():
                    valid_sizes = dff_grain_cleaned['spotSize'].dropna(); max_sz = valid_sizes.max() if not valid_sizes.empty else 0
                    if max_sz > 0: size_range = MAX_MARKER_SIZE_VISUAL - MIN_MARKER_SIZE_VISUAL; dff_grain_cleaned.loc[:,'sz_scaled'] = (dff_grain_cleaned['spotSize']/max_sz)*size_range+MIN_MARKER_SIZE_VISUAL; dff_grain_cleaned.loc[:,'sz_scaled']=dff_grain_cleaned['sz_scaled'].clip(lower=MIN_MARKER_SIZE_VISUAL, upper=MAX_MARKER_SIZE_VISUAL).fillna(DEFAULT_MARKER_SIZE); size_col='sz_scaled'

                if not dff_grain_cleaned.empty:
                    mS = np.nanmean(np.abs(dff_grain_cleaned['strain'])) if 'strain' in dff_grain_cleaned else 0; medS = np.nanmedian(np.abs(dff_grain_cleaned['strain'])) if 'strain' in dff_grain_cleaned else 0
                    title=f'Spots ID:{int(selected_id)} (Mean:{mS:.1f}, Med:{medS:.1f} µstrain)'
                    hover_cols = ['strain', 'tTheta', 'eta', 'ds', 'spotSize', 'ringNr', 'omeRaw']; valid_hover = [c for c in hover_cols if c in dff_grain_cleaned.columns]

                    color_arg = None; manual_color = None; is_discrete_color = False

                    if spots_color_choice == 'grainIDColor':
                         color_arg = None; grain_id_str = str(int(selected_id)); color_index = int(grain_id_str) % len(px.colors.qualitative.Plotly); manual_color = px.colors.qualitative.Plotly[color_index]
                    elif spots_color_choice == 'ringNr':
                        unique_rings = dff_grain_cleaned['ringNr'].unique()
                        if len(unique_rings) == 1:
                             ring_str = unique_rings[0]; ring_int = dff_grain_cleaned['ringNrInt'].iloc[0] # Get corresponding int
                             color_index = int(ring_int) % len(px.colors.qualitative.Plotly); manual_color = px.colors.qualitative.Plotly[color_index]; color_arg = None
                        else: # Multiple rings, let px handle it
                            color_arg = 'ringNr'; is_discrete_color = True
                    else: # Continuous or fallback
                        if spots_color_choice not in dff_grain_cleaned.columns: color_col = 'strain' if 'strain' in dff_grain_cleaned.columns else 'ringNr'
                        elif spots_color_choice in numeric_cols and dff_grain_cleaned[spots_color_choice].isnull().all(): color_col = 'ringNr'
                        else: color_col = spots_color_choice
                        color_arg = color_col # Pass column name for continuous or fallback ringNr

                    fig = px.scatter_3d(dff_grain_cleaned, x='omega', y='y', z='z',
                                        color=color_arg, # Pass column name or None
                                        size=size_col, size_max=MAX_MARKER_SIZE_VISUAL+2, title=title,
                                        hover_name='spotID', hover_data=valid_hover, custom_data=['spotID'],
                                        color_continuous_scale=DEFAULT_CONTINUOUS_COLOR_SCALE,
                                        color_discrete_map="identity" if color_arg == 'grainIDColor' else None, # Identity only for grainIDColor if we map it
                                        color_discrete_sequence=px.colors.qualitative.Plotly if is_discrete_color else None # Sequence for multi-ringNr
                                        )
                    if manual_color:
                        fig.update_traces(marker_color=manual_color)

                    fig.update_layout(**COMMON_LAYOUT_SETTINGS)
                else: fig.update_layout(title=f"No valid spots for Grain {int(selected_id)}", **COMMON_LAYOUT_SETTINGS)
            else: print(f"Clicked ID {clicked_id} invalid."); selected_id = None; fig.update_layout(title="Invalid grain selected", **COMMON_LAYOUT_SETTINGS)
        except Exception as e: print(f"Error update_filtered_spots_3d: {e}"); traceback.print_exc(); selected_id = None; fig.update_layout(title="Error updating 3D spots", **COMMON_LAYOUT_SETTINGS)
        return fig, selected_id if selected_id is not no_update else None

    # --- Filtered Spots 2D Callback ---
    @callback( Output('filtered_spots_2d', 'figure'), Input('selected-grain-id-store', 'data'), Input('radio-buttons-spots_filtered', 'value'), prevent_initial_call=True)
    def update_filtered_spots_2d(selected_grain_id, spots_color_choice):
        fig = go.Figure()
        if selected_grain_id is None: return fig.update_layout(title="Select a grain", **COMMON_LAYOUT_SETTINGS)
        try:
            dff_grain = spots_df[spots_df['grainID'] == selected_grain_id].copy()
            numeric_cols = ['y', 'z', 'spotSize', 'strain', 'tTheta', 'eta', 'ds', 'detY', 'detZ', 'detHor', 'detVer', 'omeRaw']
            for col in numeric_cols:
                 if col in dff_grain.columns: dff_grain.loc[:, col] = pd.to_numeric(dff_grain[col], errors='coerce')
            required_cols = ['y', 'z', 'spotID']
            if spots_color_choice in numeric_cols and spots_color_choice in dff_grain.columns: required_cols.append(spots_color_choice)
            if 'spotSize' in dff_grain.columns: required_cols.append('spotSize')
            if spots_color_choice in ['grainIDColor', 'ringNr'] and spots_color_choice in dff_grain.columns: required_cols.append(spots_color_choice)

            dff_grain_cleaned = dff_grain.dropna(subset=[c for c in required_cols if c in dff_grain.columns]).copy()

            size_col = None # Handle size scaling
            if 'spotSize' in dff_grain_cleaned.columns and dff_grain_cleaned['spotSize'].notna().any():
                valid_sizes = dff_grain_cleaned['spotSize'].dropna(); max_sz = valid_sizes.max() if not valid_sizes.empty else 0
                if max_sz > 0: size_range = MAX_MARKER_SIZE_VISUAL - MIN_MARKER_SIZE_VISUAL; dff_grain_cleaned.loc[:,'sz_scaled'] = (dff_grain_cleaned['spotSize']/max_sz)*size_range+MIN_MARKER_SIZE_VISUAL; dff_grain_cleaned.loc[:,'sz_scaled']=dff_grain_cleaned['sz_scaled'].clip(lower=MIN_MARKER_SIZE_VISUAL, upper=MAX_MARKER_SIZE_VISUAL).fillna(DEFAULT_MARKER_SIZE); size_col='sz_scaled'

            if not dff_grain_cleaned.empty:
                mS = np.nanmean(np.abs(dff_grain_cleaned['strain'])) if 'strain' in dff_grain_cleaned else 0
                title=f'2D Spots ID:{int(selected_grain_id)} (Mean Strain:{mS:.1f} µstrain)'
                hover_cols = ['detHor', 'detVer', 'omeRaw', 'eta', 'tTheta', 'strain', 'ds', 'spotSize', 'ringNr']; valid_hover = [c for c in hover_cols if c in dff_grain_cleaned.columns]

                color_arg = None; manual_color = None; is_discrete_color = False
                if spots_color_choice == 'grainIDColor':
                    grain_id_str = str(int(selected_grain_id)); color_index = int(grain_id_str) % len(px.colors.qualitative.Plotly); manual_color = px.colors.qualitative.Plotly[color_index]; color_arg = None
                elif spots_color_choice == 'ringNr':
                    unique_rings = dff_grain_cleaned['ringNr'].unique()
                    if len(unique_rings) == 1:
                         ring_str = unique_rings[0]; ring_int = dff_grain_cleaned['ringNrInt'].iloc[0]
                         color_index = int(ring_int) % len(px.colors.qualitative.Plotly); manual_color = px.colors.qualitative.Plotly[color_index]; color_arg = None
                    else: color_arg = 'ringNr'; is_discrete_color = True
                else: # Continuous or fallback
                    if spots_color_choice not in dff_grain_cleaned.columns: color_col = 'strain' if 'strain' in dff_grain_cleaned.columns else 'ringNr'
                    elif spots_color_choice in numeric_cols and dff_grain_cleaned[spots_color_choice].isnull().all(): color_col = 'ringNr'
                    else: color_col = spots_color_choice
                    color_arg = color_col

                fig = px.scatter(dff_grain_cleaned, x='y', y='z',
                                color=color_arg, # Pass column name or None
                                size=size_col, size_max=MAX_MARKER_SIZE_VISUAL+2, title=title,
                                hover_name='spotID', hover_data=valid_hover, custom_data=['spotID'],
                                color_continuous_scale=DEFAULT_CONTINUOUS_COLOR_SCALE,
                                color_discrete_map="identity" if color_arg == 'grainIDColor' else None, # Identity only for grainIDColor if we map it
                                color_discrete_sequence=px.colors.qualitative.Plotly if is_discrete_color else None)

                if manual_color: fig.update_traces(marker_color=manual_color) # Apply manual color if needed
                fig.update_layout(xaxis_title="Det Y (pix)", yaxis_title="Det Z (pix)", xaxis_autorange="reversed", yaxis_scaleanchor="x", **COMMON_LAYOUT_SETTINGS)
            else: fig.update_layout(title=f"No valid spots for Grain {int(selected_grain_id)}", **COMMON_LAYOUT_SETTINGS)
        except Exception as e: print(f"Error 2D spots plot: {e}"); traceback.print_exc(); fig.update_layout(title="Error displaying 2D spots", **COMMON_LAYOUT_SETTINGS)
        return fig

    # --- Volume Plot Callback ---
    @callback( Output('image_data', 'figure'), Input('filtered_spots_2d', 'clickData'), State('selected-grain-id-store', 'data'), prevent_initial_call=True)
    def update_volume_plot(spot_clickData, selected_grain_id):
        fig = go.Figure()
        if selected_grain_id is None: return fig.update_layout(title="Select a grain", **COMMON_LAYOUT_SETTINGS)
        if not (spot_clickData and spot_clickData['points'] and 'customdata' in spot_clickData['points'][0]): return fig.update_layout(title="Click a spot", **COMMON_LAYOUT_SETTINGS)
        try: clicked_spot_id_val = spot_clickData['points'][0]['customdata'][0]; clicked_spot_id = int(clicked_spot_id_val) if clicked_spot_id_val is not None else None
        except (ValueError, TypeError, IndexError): print(f"Invalid spot ID: {clicked_spot_id_val}"); return fig.update_layout(title="Invalid spot ID clicked", **COMMON_LAYOUT_SETTINGS)
        if clicked_spot_id is None: return fig.update_layout(title="Click on a spot", **COMMON_LAYOUT_SETTINGS)
        spot_info_df = spots_df[(spots_df['spotID'] == clicked_spot_id) & (spots_df['grainID'] == selected_grain_id)]
        if spot_info_df.empty: return fig.update_layout(title=f"Spot {clicked_spot_id}/Grain {int(selected_grain_id)} not found", **COMMON_LAYOUT_SETTINGS)
        spot_info = spot_info_df.iloc[0]
        try: detY_px, detZ_px = int(spot_info['detY']), int(spot_info['detZ'])
        except (ValueError, TypeError): return fig.update_layout(title=f"Invalid coords Spot {clicked_spot_id}", **COMMON_LAYOUT_SETTINGS)
        omega_deg = spot_info['omeRaw']
        if pd.isna(omega_deg): return fig.update_layout(title=f"Missing OmegaRaw for Spot {clicked_spot_id}", **COMMON_LAYOUT_SETTINGS)
        frameNrMid = int(round((omega_deg - zarr_params['omegaStart']) / zarr_params['omegaStep']))
        frameMin, frameMax = max(0, frameNrMid - VOLUME_WINDOW_FRAME), min(zarr_params['nFrames'], frameNrMid + VOLUME_WINDOW_FRAME + 1)
        yMin, yMax = max(0, detY_px - VOLUME_WINDOW_PX), min(zarr_params['nPxY'], detY_px + VOLUME_WINDOW_PX + 1)
        zMin, zMax = max(0, detZ_px - VOLUME_WINDOW_PX), min(zarr_params['nPxZ'], detZ_px + VOLUME_WINDOW_PX + 1)
        yMin_trans, yMax_trans, zMin_trans, zMax_trans = yMin, yMax, zMin, zMax
        nPxY_eff, nPxZ_eff = zarr_params['nPxY'], zarr_params['nPxZ']
        for transOpt in zarr_params.get('ImTransOpt', []):
            if transOpt == 1: yMin_trans, yMax_trans = nPxY_eff - yMax_trans, nPxY_eff - yMin_trans
            elif transOpt == 2: zMin_trans, zMax_trans = nPxZ_eff - zMax_trans, nPxZ_eff - zMin_trans
            elif transOpt == 3: yMin_trans, zMin_trans, yMax_trans, zMax_trans, nPxY_eff, nPxZ_eff = zMin_trans, yMin_trans, zMax_trans, yMax_trans, nPxZ_eff, nPxY_eff
        yMin_trans, yMax_trans = max(0, yMin_trans), min(nPxY_eff, yMax_trans); zMin_trans, zMax_trans = max(0, zMin_trans), min(nPxZ_eff, zMax_trans)
        if frameMin >= frameMax or zMin_trans >= zMax_trans or yMin_trans >= yMax_trans: return fig.update_layout(title=f"Volume indices invalid", **COMMON_LAYOUT_SETTINGS)
        try: # Zarr extraction
            dark_slice = zarr_params['darkMean'][zMin_trans:zMax_trans, yMin_trans:yMax_trans]
            extracted_data = zarr_params['rawDataLink'][frameMin:frameMax, zMin_trans:zMax_trans, yMin_trans:yMax_trans].astype(np.double)
            if extracted_data.shape[1:] == dark_slice.shape: extracted_data -= dark_slice
            else: print(f"Warn: Dark ({dark_slice.shape})/data ({extracted_data.shape[1:]}) mismatch.")
            extracted_data[extracted_data < zarr_params['thresh']] = 0; extracted_data = extracted_data.astype(np.uint16)
            if extracted_data.size == 0 or np.max(extracted_data) < zarr_params['thresh']: return fig.update_layout(title=f"No data > thresh (Spot {clicked_spot_id})", **COMMON_LAYOUT_SETTINGS)
            F, Z, Y = np.mgrid[frameMin:frameMax, zMin_trans:zMax_trans, yMin_trans:yMax_trans]
            fig = go.Figure(data=go.Volume(x=F.flatten(), y=Y.flatten(), z=Z.flatten(), value=extracted_data.flatten(), isomin=zarr_params['thresh'], isomax=np.max(extracted_data), opacity=0.1, surface_count=17, caps=dict(x_show=False, y_show=False, z_show=False)))
            fig.update_layout(title=f'Volume: Spot {clicked_spot_id} (Grain {int(selected_grain_id)})', scene=dict(xaxis_title='Frame', yaxis_title='Det Y', zaxis_title='Det Z'), **COMMON_LAYOUT_SETTINGS)
        except IndexError as e: print(f"Error Zarr slicing: {e}"); traceback.print_exc(); return fig.update_layout(title=f"Error extracting (Indices)", **COMMON_LAYOUT_SETTINGS)
        except Exception as e: print(f"Error Zarr processing: {e}"); traceback.print_exc(); return fig.update_layout(title=f"Error extracting data", **COMMON_LAYOUT_SETTINGS)
        return fig

    # --- All Spots 3D Callback ---
    @callback( Output('spots', 'figure'), Input('radio-buttons-spots', 'value'), Input("checklist_spots", "value"))
    def update_all_spots_3d(spots_color_choice, selected_ring_nrs):
        fig = go.Figure()
        if spots_df is None or spots_df.empty: return fig.update_layout(title='Spots data empty', **COMMON_LAYOUT_SETTINGS)
        if selected_ring_nrs is None or not isinstance(selected_ring_nrs, list) or len(selected_ring_nrs) == 0: return fig.update_layout(title='Select rings', **COMMON_LAYOUT_SETTINGS)
        try: selected_ring_nrs_int = [int(nr) for nr in selected_ring_nrs]
        except (ValueError, TypeError): selected_ring_nrs_int = []
        if not selected_ring_nrs_int: return fig.update_layout(title='Invalid ring selection', **COMMON_LAYOUT_SETTINGS)

        dff = spots_df[spots_df['ringNrInt'].isin(selected_ring_nrs_int)].copy()
        numeric_cols = ['omega', 'y', 'z', 'spotSize', 'strain']
        for col in numeric_cols:
            if col in dff.columns: dff.loc[:, col] = pd.to_numeric(dff[col], errors='coerce')
        required_cols = ['omega', 'y', 'z', 'spotID', 'grainID', 'ringNr', 'strain', 'spotSize']
        if spots_color_choice in numeric_cols and spots_color_choice in dff.columns: required_cols.append(spots_color_choice)
        elif spots_color_choice not in ['grainIDColor', 'ringNr'] and spots_color_choice not in dff.columns: print(f"Warn: Color '{spots_color_choice}' missing, defaulting."); spots_color_choice = 'ringNr'
        dff_cleaned = dff.dropna(subset=['omega', 'y', 'z', 'spotID']).copy()
        if dff_cleaned.empty: return fig.update_layout(title='No valid spots for selection', **COMMON_LAYOUT_SETTINGS)

        marker_size = DEFAULT_MARKER_SIZE
        if 'spotSize' in dff_cleaned.columns and dff_cleaned['spotSize'].notna().any():
            valid_sizes = dff_cleaned['spotSize'].dropna(); max_sz = valid_sizes.max() if not valid_sizes.empty else 0
            if max_sz > 0: size_range = MAX_MARKER_SIZE_VISUAL - MIN_MARKER_SIZE_VISUAL; scaled_sizes = (dff_cleaned['spotSize']/max_sz)*size_range+MIN_MARKER_SIZE_VISUAL; marker_size = scaled_sizes.fillna(DEFAULT_MARKER_SIZE).clip(lower=MIN_MARKER_SIZE_VISUAL, upper=MAX_MARKER_SIZE_VISUAL)

        try:
            title = f'All Spots 3D (Color: {spots_color_choice})'; marker_props = dict(size=marker_size, sizemode='diameter', opacity=0.8); layout_updates = {'title': title, 'scene': dict(xaxis_title='Omega', yaxis_title='Det Y', zaxis_title='Det Z'), 'showlegend': False, **COMMON_LAYOUT_SETTINGS}
            is_cont_color = False; color_col = spots_color_choice # Use color_col for logic

            if color_col in numeric_cols and color_col in dff_cleaned.columns:
                 if not dff_cleaned[color_col].isnull().all():
                      is_cont_color = True
                      marker_props['color'] = dff_cleaned[color_col]
                      marker_props['colorscale'] = DEFAULT_CONTINUOUS_COLOR_SCALE
                      marker_props['colorbar'] = dict(title=color_col)
                 else: color_col = 'ringNr' # Fallback if all NaN
            elif color_col not in ['grainIDColor', 'ringNr']: color_col = 'ringNr' # Fallback if col missing

            if not is_cont_color:
                 if color_col in dff_cleaned.columns:
                      unique_cats = dff_cleaned[color_col].astype(str).unique()
                      cmap = {c: px.colors.qualitative.Plotly[i%len(px.colors.qualitative.Plotly)] for i,c in enumerate(unique_cats)}
                      marker_props['color'] = dff_cleaned[color_col].astype(str).map(cmap)
                 else: # Ultimate fallback if even ringNr is somehow missing
                     marker_props['color'] = 'grey'


            hover_cols = ['spotID', 'grainID', 'ringNr', 'strain', 'spotSize']; customdata_list = []; hover_parts = ["<b>Spot</b><br>"]
            for i, col in enumerate(hover_cols):
                if col in dff_cleaned.columns: data = dff_cleaned[col].fillna("N/A"); customdata_list.append(data); fmt=".1f" if col in ['strain','spotSize'] else (".0f" if col in ['spotID','grainID','ringNrInt'] or (col=='ringNr' and pd.api.types.is_integer_dtype(dff_cleaned['ringNrInt'].dtype)) else ""); hover_parts.append(f"{col}:%{{customdata[{i}]:{fmt}}}<br>")
                else: customdata_list.append(pd.Series(["N/A"]*len(dff_cleaned), index=dff_cleaned.index)); hover_parts.append(f"{col}: N/A<br>")
            customdata = np.stack(customdata_list, axis=-1) if customdata_list else None; hovertemplate = "".join(hover_parts) + "<extra></extra>"

            trace = go.Scatter3d(x=dff_cleaned['omega'], y=dff_cleaned['y'], z=dff_cleaned['z'], mode='markers', marker=marker_props, customdata=customdata, hovertemplate=hovertemplate, name='')
            fig.add_trace(trace)
            fig.update_layout(**layout_updates)
        except Exception as e: print(f"Error generating all spots 3D: {e}"); traceback.print_exc(); fig.update_layout(title='Error generating Spots 3D plot', **COMMON_LAYOUT_SETTINGS)
        return fig

    # --- G-Vector Plot Callback ---
    @callback( Output('spots_polar', 'figure'), Input('radio-buttons-spots_polar', 'value'), Input("checklist_spots_polar", "value"))
    def update_g_vector_plot(gvector_color_choice, selected_ring_nrs):
        fig = go.Figure(); scene_settings = dict(aspectmode='cube')
        if spots_df is None or spots_df.empty: return fig.update_layout(title='Spots data empty', scene=scene_settings, **COMMON_LAYOUT_SETTINGS)
        if selected_ring_nrs is None or not isinstance(selected_ring_nrs, list) or len(selected_ring_nrs) == 0: return fig.update_layout(title='Select rings', scene=scene_settings, **COMMON_LAYOUT_SETTINGS)
        try: selected_ring_nrs_int = [int(nr) for nr in selected_ring_nrs]
        except (ValueError, TypeError): selected_ring_nrs_int = []
        if not selected_ring_nrs_int: return fig.update_layout(title='Invalid ring selection', scene=scene_settings, **COMMON_LAYOUT_SETTINGS)

        dff = spots_df[spots_df['ringNrInt'].isin(selected_ring_nrs_int)].copy()
        numeric_cols = ['g1', 'g2', 'g3', 'ds', 'strain']
        for col in numeric_cols:
             if col in dff.columns: dff.loc[:, col] = pd.to_numeric(dff[col], errors='coerce')
        required_cols = ['g1', 'g2', 'g3', 'spotID', 'grainID', 'ringNr', 'strain', 'ds'] # Base + hover
        size_data = DEFAULT_MARKER_SIZE # Default size

        # --- Color Logic ---
        color_arg = None; manual_color_map = None; is_discrete_color = False
        color_col_actual = gvector_color_choice # Start with user choice

        if color_col_actual == 'grainIDColor':
            if 'grainIDColor' in dff.columns:
                 is_discrete_color = True; required_cols.append('grainIDColor')
                 unique_cats = dff['grainIDColor'].dropna().astype(str).unique()
                 manual_color_map = {cat: px.colors.qualitative.Plotly[i%len(px.colors.qualitative.Plotly)] for i, cat in enumerate(unique_cats)}
                 color_arg = 'grainIDColor' # Use color_discrete_map with px
            else: color_col_actual = 'ringNr' # Fallback
        elif color_col_actual == 'ringNr':
            if 'ringNr' in dff.columns:
                is_discrete_color = True; required_cols.append('ringNr'); color_arg = 'ringNr' # Let px handle multi-ring case
                # Check if only one ring after potential filtering/dropna
                dff_temp_clean = dff.dropna(subset=['ringNr']) # Check on potentially filtered data
                if not dff_temp_clean.empty:
                    unique_rings = dff_temp_clean['ringNr'].unique()
                    if len(unique_rings) == 1:
                        ring_int = dff_temp_clean['ringNrInt'].iloc[0] # Get corresponding int
                        color_index = int(ring_int) % len(px.colors.qualitative.Plotly);
                        # Use manual_color_map to force the single color
                        manual_color_map = {unique_rings[0]: px.colors.qualitative.Plotly[color_index]}
                        color_arg = 'ringNr' # Still use ringNr for the map key
            else: color_col_actual = 'strain' # Fallback
        # Check for continuous/numeric case AFTER categorical checks
        if color_arg is None and manual_color_map is None: # If not handled as discrete yet
            if color_col_actual in numeric_cols:
                if color_col_actual in dff.columns and not dff[color_col_actual].isnull().all():
                     required_cols.append(color_col_actual); color_arg = color_col_actual
                else: color_col_actual = 'ringNr'; color_arg = 'ringNr'; is_discrete_color=True; required_cols.append('ringNr') # Fallback to ringNr
            elif color_col_actual not in dff.columns: # Handle case where choice is invalid string
                 color_col_actual = 'ringNr'; color_arg = 'ringNr'; is_discrete_color=True; required_cols.append('ringNr')

        # Setup size aesthetic using 'ds'
        size_arg = None
        if color_col_actual != 'ds' and 'ds' in dff.columns and dff['ds'].notna().any():
             required_cols.append('ds'); ds_clean = dff['ds'].dropna()
             if not ds_clean.empty: max_ds, min_ds = ds_clean.max(), ds_clean.min()
             else: max_ds, min_ds = 1, 0 # Avoid division by zero if empty
             if max_ds > min_ds: size_range = MAX_MARKER_SIZE_VISUAL - MIN_MARKER_SIZE_VISUAL; norm = (dff['ds'] - min_ds) / (max_ds - min_ds); scaled = norm*size_range+MIN_MARKER_SIZE_VISUAL; size_data = scaled.fillna(DEFAULT_MARKER_SIZE).clip(lower=MIN_MARKER_SIZE_VISUAL, upper=MAX_MARKER_SIZE_VISUAL)
             else: size_data = DEFAULT_MARKER_SIZE # Handle case where max_ds == min_ds
             size_arg = size_data if isinstance(size_data, pd.Series) else None


        # Final check on required columns before dropna
        final_required = [c for c in required_cols if c in dff.columns]
        dff_cleaned = dff.dropna(subset=final_required).copy()

        if isinstance(size_arg, pd.Series): size_arg = size_arg.reindex(dff_cleaned.index).fillna(DEFAULT_MARKER_SIZE) # Align size series
        if dff_cleaned.empty: return fig.update_layout(title='No valid G-vectors for selection', scene=scene_settings, **COMMON_LAYOUT_SETTINGS)

        try: # Axis range calc
             g_mag = np.sqrt(dff_cleaned['g1']**2 + dff_cleaned['g2']**2 + dff_cleaned['g3']**2); g_mag_finite = g_mag.dropna(); max_lim = 0.1
             if not g_mag_finite.empty: max_lim = max(np.percentile(g_mag_finite, 99.5) * 1.1, 0.01)
             axis_range = [-max_lim, max_lim]; scene_settings = dict(xaxis=dict(range=axis_range, title='G1 (1/Å)'), yaxis=dict(range=axis_range, title='G2 (1/Å)'), zaxis=dict(range=axis_range, title='G3 (1/Å)'), aspectmode='cube')
        except Exception as e: print(f"Error G-vector range: {e}")

        try: # Plotting with px
            plot_title = f'G-vectors (Color: {gvector_color_choice})'
            hover_cols = ['grainID', 'ringNr', 'strain', 'ds']; valid_hover = [c for c in hover_cols if c in dff_cleaned.columns]

            fig = px.scatter_3d(dff_cleaned, x='g1', y='g2', z='g3',
                                color=color_arg, # Pass column name or None
                                size=size_arg, size_max=MAX_MARKER_SIZE_VISUAL+2,
                                title=plot_title, hover_name='spotID', hover_data=valid_hover, custom_data=['spotID'],
                                color_continuous_scale=DEFAULT_CONTINUOUS_COLOR_SCALE if not is_discrete_color and manual_color_map is None else None,
                                color_discrete_sequence=px.colors.qualitative.Plotly if (is_discrete_color and manual_color_map is None and color_arg != 'grainIDColor') else None,
                                color_discrete_map=manual_color_map if manual_color_map else None
                               )

            # Apply fixed size if size was not data-driven
            if size_arg is None: fig.update_traces(marker_size=DEFAULT_MARKER_SIZE)


            fig.update_layout(scene=scene_settings, **COMMON_LAYOUT_SETTINGS)
        except Exception as e: print(f"Error px G-vector plot: {e}"); traceback.print_exc(); fig.update_layout(title='Error generating G-vector plot', scene=scene_settings, **COMMON_LAYOUT_SETTINGS)
        return fig


    # --- Spot Detail Table Callback ---
    @callback( Output('spot-detail-table', 'data'), Output('spot-detail-table', 'columns'), Output('spot-detail-table', 'page_size'), Input('selected-grain-id-store', 'data'), Input('page-size-dropdown', 'value'))
    def update_spot_table(selected_grain_id, selected_page_size):
        cols = [{"name": i, "id": i} for i in SPOT_TABLE_COLUMNS]; page_size = selected_page_size if selected_page_size != 99999 else DEFAULT_TABLE_PAGE_SIZE
        if selected_grain_id is None or spots_df is None or spots_df.empty: return [], cols, page_size
        try:
            dff_table_filtered = spots_df[spots_df['grainID'] == selected_grain_id]
            table_cols_to_show = [col for col in SPOT_TABLE_COLUMNS if col in dff_table_filtered.columns]
            dff_table_display = dff_table_filtered[table_cols_to_show].copy()
            float_cols = dff_table_display.select_dtypes(include=['float']).columns
            if not float_cols.empty: dff_table_display.loc[:, float_cols] = dff_table_display.loc[:, float_cols].round(4)
            int_cols = dff_table_display.select_dtypes(include=['Int64']).columns
            if not int_cols.empty:
                for col in int_cols:
                     if dff_table_display[col].isnull().any():
                          dff_table_display.loc[:, col] = dff_table_display.loc[:, col].astype(object)
            table_columns = [{"name": i, "id": i} for i in table_cols_to_show]
            if selected_page_size == 99999: page_size = max(len(dff_table_display), DEFAULT_TABLE_PAGE_SIZE)
            else: page_size = selected_page_size
            table_data = dff_table_display.astype(object).where(pd.notnull(dff_table_display), None).to_dict('records')
            return table_data, table_columns, page_size
        except Exception as e: print(f"Error updating spot table: {e}"); traceback.print_exc(); return [], cols, DEFAULT_TABLE_PAGE_SIZE

    # --- Run the App ---
    print(f"Starting Dash server on http://{hn}:{portNr}")
    app.run(port=portNr, host=hn, debug=False) # Set debug=True for development if needed