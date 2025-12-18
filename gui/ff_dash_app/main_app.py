# main_app.py
import dash
from dash import dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import base64
import io
import os
import time
import traceback
from pathlib import Path
import shutil
import json
import csv

import data_processing_dash as dpd
import utils_dash as utils

# --- Configuration ---
UPLOAD_DIRECTORY = "uploads_viewer_h5_zarr"
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], suppress_callback_exceptions=True)
server = app.server

# --- Helper Functions (for HKL upload) ---
def parse_upload_contents(contents, filename):
    if contents is None or ',' not in contents:
        raise ValueError("File content is malformed or missing.")
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded, filename

def parse_hkls_csv_content(csv_content_decoded_str):
    parsed_rings = []
    ring_data_list_unique = []
    seen_ring_numbers = set()
    try:
        csvfile = io.StringIO(csv_content_decoded_str)
        lines = [line for line in csvfile if line.strip() and not line.strip().startswith('#')]
        header_found_heuristic = False
        if lines and any(c.isalpha() for c in lines[0]):
            lines = lines[1:]
            header_found_heuristic = True
        
        reader = csv.reader(lines, delimiter=' ', skipinitialspace=True)

        for i, row_parts_raw in enumerate(reader):
            row_parts = [part for part in row_parts_raw if part]
            if not row_parts:
                continue
            if not header_found_heuristic and i == 0 and any(c.isalpha() for c in "".join(row_parts)):
                continue

            if len(row_parts) >= 11: # Expected: H K L Mult RingNr ... Radius_um (idx 10)
                try:
                    h, k, l = int(row_parts[0]), int(row_parts[1]), int(row_parts[2])
                    ring_id = int(row_parts[4])
                    radius_um = float(row_parts[10])
                    parsed_rings.append({
                        'hkl': [h, k, l],
                        'radius_um': radius_um,
                        'id': ring_id,
                    })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping HKL line (parse error): {row_parts_raw} -> {row_parts} (Error: {e})")
            else:
                print(f"Warning: Skipping short HKL line (<11 parts): {row_parts_raw} -> {row_parts}")

        for ring_entry in parsed_rings:
            if ring_entry['id'] not in seen_ring_numbers:
                hkl_str = f"{ring_entry['hkl'][0]},{ring_entry['hkl'][1]},{ring_entry['hkl'][2]}"
                ring_entry['display_str'] = f"Ring {ring_entry['id']} ({hkl_str}) R: {ring_entry['radius_um']:.1f}µm"
                ring_data_list_unique.append(ring_entry)
                seen_ring_numbers.add(ring_entry['id'])
    except Exception as e:
        print(f"ERROR parsing HKL CSV content: {traceback.format_exc()}")
    
    print(f"DEBUG (parse_hkls_csv): Parsed {len(ring_data_list_unique)} unique rings from {len(parsed_rings)} total valid entries.")
    return ring_data_list_unique

# --- App Layout ---
app.layout = dbc.Container([
    dcc.Store(id='store-params', data={}),
    dcc.Store(id='store-current-image-data-json', data=None),
    dcc.Store(id='store-current-file-info', data={}),
    # REMOVED: store-data-file-path and store-params-file-content. Paths will come from Inputs directly.
    dcc.Store(id='store-available-ring-data', data=[]),
    dcc.Store(id='store-selected-ring-ids', data=[]),

    dbc.Row(dbc.Col(html.H1("Far-Field Viewer (HDF5/Zarr)"), width=12), className="mb-3"),
    dbc.Alert("Important: This app reads files directly from the server. Please provide the full, absolute file path.", color="info"),
    
    dbc.Row([ # File Path Inputs
        dbc.Col([
            dbc.Label("Data File Path (.zip or .h5):"),
            dbc.Input(id='input-data-file-path', type='text', placeholder='e.g., /path/to/your/data.h5', debounce=True),
        ], md=6),
        dbc.Col([
            dbc.Label("Parameters File Path (required for .h5):"),
            dbc.Input(id='input-params-file-path', type='text', placeholder='e.g., /path/to/your/params.txt', debounce=True),
        ], md=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div("HKL Data File (hkls.csv, optional):"),
            dcc.Upload(id='upload-hkls-csv', children=html.Div(['Drag & Drop or ', html.A('Select hkls.csv')]),
                       style={'width':'98%','height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px','textAlign':'center','margin':'10px'}),
            html.Div(id='output-hkls-upload-status')
        ], md=6),
    ], className="mt-3"),

    dbc.Row([ # Frame/Max Proj Controls - Top Row
        dbc.Col(dbc.Button("Load Display", id="btn-load-frame", color="primary", className="mt-2 mb-2"), width="auto"),
        dbc.Col([
            dbc.Label("Frame Nr:", html_for="input-frame-nr", className="mr-2"), 
            dcc.Input(id="input-frame-nr", type="number", value=0, min=0, step=1, style={'width': '80px'})
        ], width="auto", className="mt-2 mb-2 d-flex align-items-center"),
        dbc.Col(html.Div(id="output-total-frames", children="/ 0"), width="auto", className="mt-2 mb-2 align-self-center"),
        dbc.Col(html.Div(id="output-omega-display", children="Omega: N/A"), width="auto", className="mt-2 mb-2 align-self-center"),
    ], className="align-items-center"),

    dbc.Row([ # Frame/Max Proj Controls - Second Row for Max Proj
         dbc.Col([
            dbc.Checkbox(id="cb-use-max-proj", label="Use Max Projection", value=False, className="mt-1 mb-2"),
        ], width="auto"),
        dbc.Col([
            dbc.Label("Num Frames:", html_for="input-max-frames", className="mr-1"),
            dcc.Input(id="input-max-frames", type="number", value=10, min=1, step=1, style={'width': '70px'}, disabled=True)
        ], width="auto", className="d-flex align-items-center mt-1 mb-2"),
        dbc.Col([
            dbc.Label("Start Frame:", html_for="input-max-start-frame", className="mr-1"),
            dcc.Input(id="input-max-start-frame", type="number", value=0, min=0, step=1, style={'width': '70px'}, disabled=True)
        ], width="auto", className="d-flex align-items-center mt-1 mb-2")
    ], className="align-items-center"),

    dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id='main-plot', style={'height': '1000px'})), width=12)),
    dbc.Row(dbc.Col(html.Div(id='plot-status-info', style={'textAlign': 'center', 'padding': '5px'}), width=12)),

    # --- Bottom Controls ---
    dbc.Row([
        dbc.Col([ # Display Options
            html.H4("Display Options"),
            dbc.Label("Min Thresh:"), dbc.Input(id="input-min-thresh", type="number", value=0),
            dbc.Label("Max Thresh:"), dbc.Input(id="input-max-thresh", type="number", value=2000),
            dbc.Checkbox(id="cb-log-scale", label="Log Scale", value=False, className="mt-2"),
            dbc.Checkbox(id="cb-hflip", label="HFlip Image", value=False),
            dbc.Checkbox(id="cb-vflip", label="VFlip Image", value=False),
            dbc.Checkbox(id="cb-transpose", label="Transpose Image (Final View)", value=False),
            dbc.Checkbox(id="cb-dark-correct-zarr", label="Apply Dark (from Data File)", value=True, className="mt-2"),
        ], md=6),
        dbc.Col([ # Ring Analysis
            html.H4("Ring Analysis"),
            dbc.Button("Select Rings to Plot...", id="btn-open-ring-select-modal", color="info", size="sm", className="mb-2"),
            dbc.Checkbox(id="cb-plot-rings", label="Plot Selected Rings", value=False),
            html.Div(id="output-ring-info", children="Upload hkls.csv & select rings.", className="mt-1 small"),
            html.H5("Parameters from File:", className="mt-3"),
            dbc.Row([dbc.Col(dbc.Label("Detector Ycen (px):"),width=5), dbc.Col(dbc.Input(id="display-bcx-zarr",type="text",readonly=True,size="sm"),width=7)],className="mb-1"),
            dbc.Row([dbc.Col(dbc.Label("Detector Zcen (px):"),width=5), dbc.Col(dbc.Input(id="display-bcy-zarr",type="text",readonly=True,size="sm"),width=7)],className="mb-1"),
            dbc.Row([dbc.Col(dbc.Label("Pixel Size (µm):"),width=5), dbc.Col(dbc.Input(id="display-px-zarr",type="text",readonly=True,size="sm"),width=7)],className="mb-1"),
            dbc.Row([dbc.Col(dbc.Label("LSD (µm):"),width=5), dbc.Col(dbc.Input(id="display-lsd-zarr",type="text",readonly=True,size="sm"),width=7)]),
        ], md=6),
    ], className="mt-4"),

    html.Hr(),
    dbc.Row(dbc.Col(html.Div(id="app-status-bar", children="Ready. Enter file paths to begin."), width=12)),

    # --- Ring Selection Modal ---
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Select Rings to Plot")),
        dbc.ModalBody([
            html.P("Select one or more rings from the list:"),
            dcc.Checklist(
                id='checklist-rings-selection',
                options=[], # Populated by callback
                value=[],   # Populated by callback
                labelStyle={'display': 'block', 'marginBottom': '5px'} 
            )
        ]),
        dbc.ModalFooter(
            dbc.Button("Apply Selection", id="btn-apply-ring-selection", className="ml-auto", n_clicks=0)
        ),
    ], id="modal-ring-selection", is_open=False, scrollable=True, size="lg"),

], fluid=True)

# --- Callbacks ---

@app.callback(
    Output('store-params', 'data'),
    Output('display-lsd-zarr', 'value'),
    Output('display-bcx-zarr', 'value'),
    Output('display-bcy-zarr', 'value'),
    Output('display-px-zarr', 'value'),
    Output('app-status-bar', 'children', allow_duplicate=True),
    Input('input-data-file-path', 'value'),
    Input('input-params-file-path', 'value'),
    prevent_initial_call=True
)
def generate_parameters(data_file_path, params_file_path):
    # This callback now triggers when file path inputs change.
    triggered_id = ctx.triggered_id
    if not triggered_id:
        return no_update

    # Basic validation
    if not data_file_path:
        return {}, "N/A", "N/A", "N/A", "N/A", "Please provide a path to a data file."

    data_file_path = data_file_path.strip()
    if not Path(data_file_path).exists():
        return {}, "N/A", "N/A", "N/A", "N/A", f"Data file not found at: {data_file_path}"
    
    params = {}
    app_status_msg = "Error processing paths."
    
    try:
        if data_file_path.endswith('.zip'):
            params = dpd.extract_parameters(data_file_path)
            app_status_msg = f"Extracted parameters from {Path(data_file_path).name}. Ready to load."
        
        elif data_file_path.endswith('.h5'):
            if not params_file_path:
                return no_update # Wait for user to enter param file path
            
            params_file_path = params_file_path.strip()
            if not Path(params_file_path).exists():
                 return {}, "N/A", "N/A", "N/A", "N/A", f"Parameters file not found at: {params_file_path}"
            
            params = dpd.extract_parameters(data_file_path, params_file_path)
            app_status_msg = f"Extracted parameters for {Path(data_file_path).name}. Ready to load."
        
        else:
            return {}, "N/A", "N/A", "N/A", "N/A", "Unsupported file type. Use .zip or .h5"

        if params:
            det_p = params.get('DetParams')[0]
            lsd_val = det_p.get('lsd')
            lsd_disp = f"{lsd_val:.1f}" if isinstance(lsd_val, (int, float)) else str(lsd_val)
            
            bc_vals = det_p.get('bc', [None, None])
            bcx_val, bcy_val = bc_vals[0], bc_vals[1]
            bcx_disp = f"{bcx_val:.2f}" if isinstance(bcx_val, (int, float)) else str(bcx_val)
            bcy_disp = f"{bcy_val:.2f}" if isinstance(bcy_val, (int, float)) else str(bcy_val)
            
            px_val = params.get('px')
            px_disp = f"{px_val:.2f}" if isinstance(px_val, (int, float)) else str(px_val)
        else:
            app_status_msg = "Failed to extract parameters."
            return {}, "N/A", "N/A", "N/A", "N/A", app_status_msg

    except Exception as e:
        app_status_msg = f"Error generating parameters: {e}"
        print(f"ERROR in generate_parameters: {traceback.format_exc()}")
        return {}, "Err", "Err", "Err", "Err", app_status_msg

    return params, lsd_disp, bcx_disp, bcy_disp, px_disp, app_status_msg

@app.callback(
    [Output('store-available-ring-data', 'data'),
     Output('output-hkls-upload-status', 'children'),
     Output('output-ring-info', 'children', allow_duplicate=True),
     Output('checklist-rings-selection', 'options'),
     Output('checklist-rings-selection', 'value')],
    [Input('upload-hkls-csv', 'contents')],
    [State('upload-hkls-csv', 'filename')],
    prevent_initial_call=True
)
def handle_hkls_upload_and_populate_checklist(contents, filename):
    if contents is None:
        return no_update, "Upload hkls.csv for ring plotting.", "Upload hkls.csv to see rings.", [], []

    available_rings = []
    upload_msg = "Error parsing HKLs file."
    ring_info_msg = "Error."
    checklist_options = []
    checklist_default_value = []
    try:
        decoded_content, _ = parse_upload_contents(contents, filename)
        hkl_content_str = decoded_content.decode('utf-8', errors='replace')
        available_rings = parse_hkls_csv_content(hkl_content_str)
        
        if not available_rings:
            upload_msg = f"No valid rings parsed from {filename}."
            ring_info_msg = upload_msg
        else:
            upload_msg = f"{len(available_rings)} unique rings parsed from {filename}."
            checklist_options = [{'label': ring['display_str'], 'value': ring['id']} for ring in available_rings]
            checklist_default_value = [ring['id'] for ring in available_rings[:min(5, len(available_rings))]]
            ring_info_msg = f"{len(available_rings)} rings available. Default selection: {len(checklist_default_value)}."
    except Exception as e:
        upload_msg = f"Error processing hkls.csv: {e}"
        print(f"ERROR in handle_hkls_upload: {traceback.format_exc()}")
        ring_info_msg = upload_msg
    
    return available_rings, upload_msg, ring_info_msg, checklist_options, checklist_default_value

@app.callback(
    Output("modal-ring-selection", "is_open"),
    [Input("btn-open-ring-select-modal", "n_clicks"), Input("btn-apply-ring-selection", "n_clicks")],
    [State("modal-ring-selection", "is_open")],
    prevent_initial_call=True,
)
def toggle_ring_modal(n_open, n_apply, is_open):
    if ctx.triggered_id == "btn-open-ring-select-modal" or ctx.triggered_id == "btn-apply-ring-selection":
        return not is_open
    return is_open

@app.callback(
    [Output('store-selected-ring-ids', 'data'),
     Output('output-ring-info', 'children', allow_duplicate=True)],
    [Input('btn-apply-ring-selection', 'n_clicks')],
    [State('checklist-rings-selection', 'value'),
     State('store-available-ring-data', 'data')],
    prevent_initial_call=True
)
def store_selected_rings(n_clicks_apply, selected_ids_from_checklist, all_available_rings_data):
    if not n_clicks_apply or selected_ids_from_checklist is None:
        return no_update, no_update

    num_total_available = len(all_available_rings_data) if all_available_rings_data else 0
    num_selected = len(selected_ids_from_checklist)
    
    info_text = f"{num_total_available} rings avail. Selected: {num_selected}."
    print(f"DEBUG: Storing selected ring IDs: {selected_ids_from_checklist}")
    return selected_ids_from_checklist, info_text


@app.callback(
    [Output('input-frame-nr', 'disabled'),
     Output('input-max-frames', 'disabled'),
     Output('input-max-start-frame', 'disabled')],
    [Input('cb-use-max-proj', 'value')]
)
def toggle_max_proj_inputs(use_max_proj_checked):
    return use_max_proj_checked, not use_max_proj_checked, not use_max_proj_checked

@app.callback(
    [Output('main-plot', 'figure'),
     Output('store-current-image-data-json', 'data'),
     Output('store-current-file-info', 'data'),
     Output('app-status-bar', 'children', allow_duplicate=True),
     Output('output-total-frames', 'children'),
     Output('output-omega-display', 'children')],
    [Input('btn-load-frame', 'n_clicks')],
    [State('store-params', 'data'), # This now contains the file path
     State('input-frame-nr', 'value'), State('cb-hflip', 'value'),
     State('cb-vflip', 'value'), State('cb-transpose', 'value'),
     State('cb-dark-correct-zarr', 'value'), State('input-min-thresh', 'value'),
     State('input-max-thresh', 'value'), State('cb-log-scale', 'value'),
     State('cb-use-max-proj', 'value'), State('input-max-frames', 'value'),
     State('input-max-start-frame', 'value'),
     State('cb-plot-rings', 'value'),
     State('store-selected-ring-ids', 'data'),
     State('store-available-ring-data', 'data')],
    prevent_initial_call=True
)
def load_frame_update_plot_and_rings(n_clicks, params, frame_nr_single,
                        hflip, vflip, transpose_display, dark_correct_cb,
                        min_thresh, max_thresh, log_scale,
                        use_max_proj_checked, num_frames_for_max, start_frame_for_max,
                        plot_rings_checked, selected_ring_ids, all_available_rings_data):
    
    if not n_clicks:
        return go.Figure(), None, {}, "Enter file paths and click Load Display.", "/ 0", "Omega: N/A"

    data_file_path = params.get('paramFN_path')
    if not data_file_path or not params or not Path(data_file_path).exists():
        return go.Figure(), None, {}, "File path not found in parameters or file does not exist.", "/ 0", "Omega: N/A"

    fig = go.Figure()
    json_serializable_image_data = None
    file_info_dict_for_store = {}
    status_msg = "Error during processing."
    total_frames_text = "/ 0"
    omega_text = "Omega: N/A"
    current_title_frame_id = frame_nr_single

    try:
        data_array = None
        if use_max_proj_checked:
            status_msg = f"Processing Max Projection..."
            data_array, file_info_dict_for_store = dpd.get_max_projection(
                params, data_file_path,
                num_frames_for_max, start_frame_for_max,
                hflip, vflip, transpose_display
            )
            current_title_frame_id = f"Max ({start_frame_for_max}-{start_frame_for_max+num_frames_for_max-1})"
            if file_info_dict_for_store.get('total_frames') and params.get('omegaStart') is not None and params.get('omegaStep') is not None:
                file_info_dict_for_store['omega'] = params['omegaStart'] + start_frame_for_max * params['omegaStep']
        else:
            status_msg = f"Processing Frame {frame_nr_single}..."
            data_array, file_info_dict_for_store = dpd.load_image_frame(
                params, data_file_path, frame_nr_single, hflip, vflip, transpose_display
            )
        
        if data_array is None:
            status_msg = f"Failed to load data. Check file structure and paths."
            return fig, None, file_info_dict_for_store, status_msg, total_frames_text, omega_text

        # Dark Correction
        processed_data_np = data_array.copy()
        if dark_correct_cb:
            dark_frame = dpd.load_dark_frame(params, data_file_path, hflip, vflip, transpose_display)
            if dark_frame is not None:
                if dark_frame.shape == processed_data_np.shape:
                    processed_data_np = processed_data_np - dark_frame
                    processed_data_np.clip(min=0, out=processed_data_np)
                    file_info_dict_for_store['dark_corrected_in_app'] = True
                    status_msg += " Dark Corrected."
                else:
                    status_msg += " Dark shape mismatch."
            else:
                status_msg += " No dark data found for correction."
        
        img_list = processed_data_np.tolist()
        img_shape = processed_data_np.shape
        img_dtype = str(processed_data_np.dtype)
        json_serializable_image_data = {'data': img_list, 'shape': img_shape, 'dtype': img_dtype}

        plot_disp_data = processed_data_np.copy()
        if log_scale:
            plot_disp_data = np.log1p(plot_disp_data.clip(min=0))
        
        fig = px.imshow(plot_disp_data, color_continuous_scale=('viridis' if log_scale else 'gray'),
                        aspect="equal", origin='lower', zmin=min_thresh, zmax=max_thresh)
        
        title_parts = [f"Display: {current_title_frame_id}"]
        if file_info_dict_for_store.get('dark_corrected_in_app'): title_parts.append("Dark Corrected")

        # Ring Plotting
        if plot_rings_checked and selected_ring_ids and all_available_rings_data and params.get('px'):
            px_size_um = params['px']
            det_p_rings = params.get('DetParams', [{}])[0]
            
            bc_from_params = det_p_rings.get('bc')
            if isinstance(bc_from_params, list) and len(bc_from_params) == 2 and \
               isinstance(bc_from_params[0], (int, float)) and \
               isinstance(bc_from_params[1], (int, float)):
                bc_plot_H_cols, bc_plot_V_rows = bc_from_params[0], bc_from_params[1]
            else:
                bc_plot_H_cols = plot_disp_data.shape[1] / 2.0
                bc_plot_V_rows = plot_disp_data.shape[0] / 2.0

            if px_size_um > 0 :
                etas_deg_ring = np.linspace(-180, 180, 181)
                rings_to_plot_info = [r for r in all_available_rings_data if r['id'] in selected_ring_ids]
                
                for i, ring_info in enumerate(rings_to_plot_info):
                    Y_offsets_um_ring, Z_offsets_um_ring = utils.YZ4mREta(ring_info['radius_um'], etas_deg_ring)
                    Y_offsets_px_ring = Y_offsets_um_ring / px_size_um
                    Z_offsets_px_ring = Z_offsets_um_ring / px_size_um
                    
                    plot_x_coords = bc_plot_H_cols - Y_offsets_px_ring
                    plot_y_coords = bc_plot_V_rows + Z_offsets_px_ring
                    
                    fig.add_trace(go.Scatter(x=plot_x_coords, y=plot_y_coords, mode='lines',
                                             line=dict(color=utils.colors[i % len(utils.colors)], dash='dot'),
                                             name=ring_info.get('display_str', f"Ring {ring_info['id']}")))
                title_parts.append(f"{len(rings_to_plot_info)} Rings Plotted")
        
        fig.update_layout(title=" | ".join(title_parts),
                          xaxis_title="Detector Y (cols, px)", yaxis_title="Detector Z (rows, px)")

        total_f = file_info_dict_for_store.get('total_frames', 0)
        total_frames_text = f"/ {total_f - 1 if total_f > 0 else 0}"
        if use_max_proj_checked:
            total_frames_text += " (Max Projection)"
        
        omega_val = file_info_dict_for_store.get('omega')
        omega_text = f"Omega: {omega_val:.4f}°" if omega_val is not None else "Omega: N/A"
        status_msg = f"Displayed: {current_title_frame_id} from {Path(data_file_path).name}"

    except Exception as e:
        print(f"ERROR in load_frame_update_plot_and_rings: {traceback.format_exc()}")
        status_msg = f"Error during display: {e}"
    
    return fig, json_serializable_image_data, file_info_dict_for_store, status_msg, total_frames_text, omega_text


@app.callback(
    Output('plot-status-info', 'children'),
    [Input('main-plot', 'hoverData')],
    [State('store-params', 'data')]
)
def display_hover_data(hoverData, params):
    if hoverData is None or not params or not params.get('DetParams'):
        return "Hover over the plot for details."
    try:
        point_data = hoverData['points'][0]
        plot_col_coord, plot_row_coord = point_data['x'], point_data['y']
        intensity_val = point_data.get('z')

        det_p = params['DetParams'][0]
        bc_param = det_p.get('bc') 
        
        if isinstance(bc_param, list) and len(bc_param) == 2 and all(isinstance(c, (int,float)) for c in bc_param):
            bc_plot_H_cols, bc_plot_V_rows = bc_param
        else:
            img_cols = params.get('NrPixelsY_detector', 0)
            img_rows = params.get('NrPixelsZ_detector', 0)
            bc_plot_H_cols = img_cols / 2.0 if img_cols else 0
            bc_plot_V_rows = img_rows / 2.0 if img_rows else 0

        rel_x_plot = plot_col_coord - bc_plot_H_cols
        rel_y_plot = plot_row_coord - bc_plot_V_rows
        
        y_physical_offset = -rel_x_plot
        z_physical_offset =  rel_y_plot
        
        eta_deg, radius_px = utils.CalcEtaAngleRad(y_physical_offset, z_physical_offset)

        status_text = f"Plot Col: {plot_col_coord:.1f}, Row: {plot_row_coord:.1f}"
        if intensity_val is not None:
            status_text += f" | I: {intensity_val:.1f}"
        status_text += f" | η: {eta_deg:.2f}° | R_px: {radius_px:.2f}"
        return status_text
    except Exception as e:
        return f"Hover error: {e}"


@app.callback(
    Output('main-plot', 'figure', allow_duplicate=True),
    [Input('input-min-thresh', 'value'), Input('input-max-thresh', 'value'),
     Input('cb-log-scale', 'value'), Input('cb-plot-rings', 'value')],
    [State('store-current-image-data-json', 'data'), State('input-frame-nr', 'value'),
     State('store-current-file-info', 'data'), State('store-params', 'data'),
     State('store-selected-ring-ids', 'data'), State('store-available-ring-data', 'data'),
     State('cb-use-max-proj', 'value'), State('input-max-start-frame', 'value'),
     State('input-max-frames', 'value')],
    prevent_initial_call=True
)
def update_plot_display_only(min_thresh, max_thresh, log_scale, plot_rings_checked,
                            current_image_json_obj, frame_nr_single, file_info, params,
                            selected_ring_ids, all_available_rings_data,
                            use_max_proj_checked, start_frame_for_max, num_frames_for_max):
    
    if current_image_json_obj is None or not params:
        return no_update

    try:
        img_json = current_image_json_obj
        img_data_np = np.array(img_json['data'], dtype=img_json['dtype']).reshape(img_json['shape'])
        
        plot_disp_data = img_data_np.copy()
        if log_scale:
            plot_disp_data = np.log1p(plot_disp_data.clip(min=0))
            
        fig = px.imshow(plot_disp_data, color_continuous_scale=('viridis' if log_scale else 'gray'),
                        aspect="equal", origin='lower', zmin=min_thresh, zmax=max_thresh)
        
        current_title_frame_id = f"Max ({start_frame_for_max}-{start_frame_for_max+num_frames_for_max-1})" if use_max_proj_checked else frame_nr_single
        title_parts = [f"Display: {current_title_frame_id}"]
        if file_info and file_info.get('dark_corrected_in_app'):
            title_parts.append("Dark Corrected")

        if plot_rings_checked and selected_ring_ids and all_available_rings_data and params.get('px'):
            px_size_um = params['px']
            det_p_rings = params.get('DetParams', [{}])[0]
            
            bc_from_params = det_p_rings.get('bc')
            if isinstance(bc_from_params, list) and len(bc_from_params) == 2 and all(isinstance(c, (int,float)) for c in bc_from_params):
                bc_plot_H_cols, bc_plot_V_rows = bc_from_params
            else:
                bc_plot_H_cols = img_data_np.shape[1] / 2.0
                bc_plot_V_rows = img_data_np.shape[0] / 2.0

            if px_size_um > 0:
                etas_deg_ring = np.linspace(-180, 180, 181)
                rings_to_plot_info = [r for r in all_available_rings_data if r['id'] in selected_ring_ids]
                
                for i, ring_info in enumerate(rings_to_plot_info):
                    Y_offsets_um_ring, Z_offsets_um_ring = utils.YZ4mREta(ring_info['radius_um'], etas_deg_ring)
                    Y_offsets_px_ring = Y_offsets_um_ring / px_size_um
                    Z_offsets_px_ring = Z_offsets_um_ring / px_size_um
                    plot_x_coords = bc_plot_H_cols - Y_offsets_px_ring
                    plot_y_coords = bc_plot_V_rows + Z_offsets_px_ring
                    
                    fig.add_trace(go.Scatter(x=plot_x_coords, y=plot_y_coords, mode='lines',
                                             line=dict(color=utils.colors[i % len(utils.colors)], dash='dot'),
                                             name=ring_info.get('display_str', f"Ring {ring_info['id']}")))
                title_parts.append(f"{len(rings_to_plot_info)} Rings Plotted")

        fig.update_layout(title=" | ".join(title_parts),
                          xaxis_title="Detector Y (cols, px)", yaxis_title="Detector Z (rows, px)")
        return fig
    except Exception as e:
        print(f"ERROR in update_plot_display_only: {traceback.format_exc()}")
        return no_update

if __name__ == '__main__':
    # No cleanup function needed as we are not creating temporary files anymore
    app.run(debug=False, host='0.0.0.0', port=1200)