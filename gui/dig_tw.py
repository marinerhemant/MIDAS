"""
MIDAS Digital Twin - Interactive visualization tool for microstructure simulation.
Supports both FF-HEDM (ForwardSimulationCompressed) and NF-HEDM (simulateNF) modes.
"""

import os
import sys
import subprocess
import glob
from typing import Tuple, List, Dict, Optional, Any, Union
from dataclasses import dataclass
from functools import lru_cache
import argparse
from math import cos, sin

# Set up paths before importing midas_config
file_path = os.path.abspath(__file__)
install_path = os.path.dirname(os.path.dirname(file_path))
utils_dir = os.path.join(install_path, 'utils/')
sys.path.insert(0, utils_dir)

import midas_config
midas_config.run_startup_checks()
import numpy as np
import pandas as pd
import zarr
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State, ctx, DiskcacheManager
import dash_bootstrap_components as dbc
import diskcache

# Set up caching
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# Constants
DEG2RAD = np.pi/180
RAD2DEG = 180/np.pi
DEFAULT_BEAM_SIZE = 2000.0
DEFAULT_DETECTOR_PIXELS = 2048
DEFAULT_OMEGA_RANGE = (-10, 10)
DEFAULT_OMEGA_STEP = 0.25



try:
    from calcMiso import OrientMat2Euler
except ImportError:
    def OrientMat2Euler(orient):
        """Fallback function if calcMiso module is not available."""
        print("Warning: calcMiso module not found. Using fallback OrientMat2Euler function.")
        # Placeholder implementation
        return [0, 0, 0]

# Environment setup
env = dict(os.environ)
if hasattr(midas_config, 'get_env'):
    env.update(midas_config.get_env())

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class SimulationParams:
    """Simulation parameters data structure."""
    lsd: float = 1000000
    xbc: float = 1024.0
    ybc: float = 1024.0
    energy: float = 71.676
    sg: int = 194
    a: float = 2.925390
    b: float = 2.925390
    c: float = 4.673990
    alpha: float = 90
    beta: float = 90
    gamma: float = 120
    beam_size: float = DEFAULT_BEAM_SIZE
    min_ome: float = DEFAULT_OMEGA_RANGE[0]
    max_ome: float = DEFAULT_OMEGA_RANGE[1]
    ome_step: float = DEFAULT_OMEGA_STEP
    pixel_size: float = 200.0   # µm
    n_pixels: int = DEFAULT_DETECTOR_PIXELS
    sim_type: str = 'FF-HEDM'

    @property
    def n_frames(self) -> int:
        """Calculate number of frames from omega range and step."""
        return int((self.max_ome - self.min_ome) / self.ome_step)

    @property
    def px_mm(self) -> float:
        """Pixel size in mm."""
        return self.pixel_size / 1000.0


@dataclass
class MicrostructureData:
    """Microstructure data structure."""
    positions: np.ndarray
    eulers: np.ndarray
    plot_type: int
    mic_file: str


class DetectorViewState:
    """Detector view state data structure."""
    def __init__(self, params: SimulationParams):
        px_mm = params.px_mm
        self.ranges: List[float] = [
            -int(params.xbc * px_mm) - 2,
            int((params.n_pixels - params.xbc) * px_mm) + 2,
            -int(params.ybc * px_mm) - 2,
            int((params.n_pixels - params.ybc) * px_mm) + 2
        ]

    def update(self, relayout_data: Dict) -> None:
        """Update ranges from relayout data."""
        if relayout_data is not None and len(relayout_data) == 4:
            self.ranges = [
                relayout_data['xaxis.range[0]'],
                relayout_data['xaxis.range[1]'],
                relayout_data['yaxis.range[0]'],
                relayout_data['yaxis.range[1]']
            ]

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

class MyParser(argparse.ArgumentParser):
    """Custom argument parser that prints help on error."""
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = MyParser(
        description='MIDAS DIGITAL TWIN - Interactive visualization tool for microstructure simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-mic', type=str, required=True, help='File name of microstructure file')
    return parser.parse_args()


def run_subprocess(cmd: List[str]) -> int:
    """Run a subprocess with improved error handling. cmd must be a list."""
    try:
        result = subprocess.run(cmd, check=True,
                              stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                              env=env)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout.decode('utf-8')}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode('utf-8')}")
        return e.returncode


def _write_common_params(pf, params: SimulationParams) -> None:
    """Write parameters common to both FF and NF simulations."""
    pf.write(f"Lsd {params.lsd}\n")
    pf.write(f"BC {params.xbc} {params.ybc}\n")
    pf.write(f"SpaceGroup {params.sg}\n")
    pf.write(f"LatticeParameter {params.a} {params.b} {params.c} {params.alpha} {params.beta} {params.gamma}\n")
    pf.write(f"tx 0\nty 0\ntz 0\n")
    pf.write(f"p0 0\np1 0\np2 0\np3 0\n")
    pf.write(f"Wedge 0\n")
    pf.write(f"OmegaStep {params.ome_step}\n")
    pf.write(f"OmegaStart {params.min_ome}\n")
    pf.write(f"OmegaEnd {params.max_ome}\n")
    pf.write(f"Wavelength {12.398 / params.energy}\n")
    pf.write(f"NrPixelsY {params.n_pixels}\n")
    pf.write(f"NrPixelsZ {params.n_pixels}\n")
    pf.write(f"px {params.pixel_size}\n")
    pf.write(f"ExcludePoleAngle 10\n")


def write_parameter_file(params: SimulationParams, mic_fn: str, param_fn: str = 'ps_midas_dtw.txt') -> None:
    """Write FF-HEDM parameters to file for ForwardSimulationCompressed."""
    with open(param_fn, 'w') as pf:
        _write_common_params(pf, params)
        pf.write(f"RhoD 404800\n")
        pf.write(f"GaussWidth 2\n")
        pf.write(f"PeakIntensity 5000\n")
        for i in range(1, 31):
            pf.write(f"RingThresh {i} 50\n")
        pf.write(f"WriteSpots 1\n")
        pf.write(f"InFileName {mic_fn}\n")
        pf.write(f"OutFileName {mic_fn}.sim\n")
        pf.write(f"BeamSize {float(params.beam_size)}\n")


def write_nf_parameter_file(params: SimulationParams, mic_fn: str, param_fn: str = 'ps_midas_dtw.txt') -> None:
    """Write NF-HEDM parameters to file for simulateNF."""
    # Read existing param file, keep only geometry lines, then add NF-specific
    with open(param_fn, 'w') as pf:
        _write_common_params(pf, params)
        pf.write(f"MaxRingRad 404800\n")
        pf.write(f"nDistances 1\n")
        pf.write(f"NrFilesPerDistance {params.n_frames}\n")
        pf.write(f"EndNr {params.n_frames}\n")
        pf.write(f"StartNr 1\n")
        pf.write(f"BoxSize -1000000 1000000 -1000000 1000000\n")
        pf.write(f"OmegaRange {min(params.min_ome, params.max_ome)} {max(params.min_ome, params.max_ome)}\n")


def update_parameter_file_for_sequence(param_fn: str, mic_fn: str, seq: int) -> str:
    """Update parameter file for sequence processing."""
    mic_st = '.'.join(mic_fn.split('.')[:-1])
    mic_fn_loc = mic_st + f'.{int(seq)}'
    
    with open(param_fn, 'r') as f:
        lines = f.readlines()
    
    with open(param_fn, 'w') as pf:
        for line in lines:
            if line.startswith('InFileName'):
                pf.write(f"InFileName {mic_fn_loc}\n")
            elif line.startswith('OutFileName'):
                pf.write(f"OutFileName {mic_fn_loc}.sim\n")
            else:
                pf.write(line)
    
    return mic_fn_loc


# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

@lru_cache(maxsize=8)
def load_microstructure_data(mic_fn: str, filter_beam_size: Optional[float] = None) -> MicrostructureData:
    """Load microstructure data from file with optional beam size filtering."""
    try:
        with open(mic_fn, 'r') as f:
            mic_h = f.readline()
        
        positions = None
        eulers = None
        plot_type = 1
        
        if mic_h.startswith('%NumGrains'):
            # Grains.csv
            mic = np.genfromtxt(mic_fn, skip_header=9)
            
            # Apply beam size filter if specified
            if filter_beam_size is not None:
                rads = np.linalg.norm(mic[:,10:12], axis=1)
                mic = mic[rads < filter_beam_size]
            
            positions = mic[:,10:13]
            orients = mic[:,1:10]
            eulers = np.zeros((mic.shape[0], 3))
            
            for row_nr in range(orients.shape[0]):
                orient = orients[row_nr].reshape(3, 3)
                det = np.linalg.det(orient)
                if abs(det) > 1e-10:  # Avoid division by zero
                    orient = orient / det
                eulers[row_nr] = RAD2DEG * OrientMat2Euler(orient.flatten())
            
            plot_type = 3
            
        elif mic_h.startswith('%TriEdgeSize'):
            # NF Mic, 2D
            mic = np.genfromtxt(mic_fn, skip_header=4)
            
            # Apply beam size filter if specified
            if filter_beam_size is not None:
                rads = np.linalg.norm(mic[:,3:5], axis=1)
                mic = mic[rads < filter_beam_size]
            
            positions = mic[:,3:6]
            positions[:,2] = 0
            eulers = mic[:,7:10]
            
            # Convert radians to degrees if needed
            if np.max(np.abs(eulers)) < 10:  # Assuming radians
                eulers = RAD2DEG * eulers
                
            plot_type = 2
            
        elif mic_h.startswith('# SpotID'):
            # pf-Mic, 2D
            mic = np.genfromtxt(mic_fn, skip_header=1, delimiter=',')
            
            # Apply beam size filter if specified
            if filter_beam_size is not None:
                rads = np.linalg.norm(mic[:,11:13], axis=1)
                mic = mic[rads < filter_beam_size]
            
            positions = mic[:,11:14]
            eulers = mic[:,-7:-4]
            plot_type = 2
            
        elif mic_h.startswith('#EBSD'):
            # EBSD CSV format
            mic = np.genfromtxt(mic_fn, skip_header=1, delimiter=',')
            
            # Apply beam size filter if specified
            if filter_beam_size is not None:
                rads = np.linalg.norm(mic[:,0:2], axis=1)
                mic = mic[rads < filter_beam_size]
            
            positions = mic[:,:2]
            eulers = mic[:,3:]  # Already in degrees
            plot_type = 2
            
        else:
            raise ValueError(f"Unrecognized microstructure file format: {mic_h}")
        
        return MicrostructureData(positions, eulers, plot_type, mic_fn)
        
    except Exception as e:
        print(f"Error loading microstructure data from {mic_fn}: {str(e)}")
        raise


@lru_cache(maxsize=8)
def load_sequence_data(seq_nr: int) -> pd.DataFrame:
    """Load sequence data from Grains.csv files."""
    try:
        all_files = glob.glob('Grains.csv.*')
        all_files = [f for f in all_files if not 'zip' in f]
        
        if not all_files:
            raise FileNotFoundError("No Grains.csv.* files found")
        
        all_df = []
        for file_n in all_files:
            try:
                df = pd.read_csv(file_n, header=8, delimiter='\t', index_col=False)
                load_nr = int(file_n.split('.')[-1])
                df.insert(0, "LoadNr", np.ones(df.shape[0]).astype(int) * load_nr, True)
                all_df.append(df)
            except Exception as e:
                print(f"Error loading {file_n}: {str(e)}")
        
        if not all_df:
            raise ValueError("No valid data loaded from sequence files")
            
        combined_df = pd.concat(all_df)
        filtered_df = combined_df.loc[combined_df['LoadNr'] == seq_nr]
        
        if filtered_df.empty:
            raise ValueError(f"No data found for sequence number {seq_nr}")
            
        return filtered_df
        
    except Exception as e:
        print(f"Error loading sequence data: {str(e)}")
        raise


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_setup_figure(hkls_data: np.ndarray, params: SimulationParams) -> go.Figure:
    """Create setup visualization figure."""
    fig = go.Figure()
    
    # Extract necessary data
    rads = hkls_data[:,-1] / 1000
    rads = rads.astype(int)
    hkls = hkls_data[:,:3].astype(int)
    unique_rads = np.unique(rads)
    
    # Add traces for each unique radius
    lsd_km = int(params.lsd / 1000)
    y_limit_lower = int(params.ybc * 0.2)
    y_limit_upper = int((DEFAULT_DETECTOR_PIXELS - params.ybc) * 0.2)
    
    for rad in unique_rads:
        found = 0
        for radnr in range(len(rads)):
            if rads[radnr] == rad and found == 0:
                hkl = np.abs(hkls[radnr])
                found = 1
        
        label = f'HKL {hkl[0]} {hkl[1]} {hkl[2]}'
        
        if rad < y_limit_lower and rad < y_limit_upper:
            fig.add_trace(go.Scatter(
                x=[lsd_km, 0, lsd_km], 
                y=[-rad, 0, rad],
                name=label
            ))
        elif rad > y_limit_lower and rad < y_limit_upper:
            fig.add_trace(go.Scatter(
                x=[0, lsd_km], 
                y=[0, rad],
                name=label
            ))
        elif rad < y_limit_lower and rad > y_limit_upper:
            fig.add_trace(go.Scatter(
                x=[lsd_km, 0], 
                y=[-rad, 0],
                name=label
            ))
    
    # Set trace mode to lines
    fig.update_traces(mode='lines')
    
    # Add beam trace
    fig.add_trace(go.Scatter(
        x=[-500], y=[-10],
        mode='text',
        text="X-ray Beam",
        textfont=dict(size=18),
        textposition='bottom center',
        name='beam'
    ))
    
    fig.add_trace(go.Scatter(
        x=[-1000, 0], y=[0, 0],
        line=dict(width=10),
        marker=dict(size=10, symbol="arrow-bar-up", angleref="previous"),
        name='beam'
    ))
    
    # Add detector trace
    fig.add_trace(go.Scatter(
        x=[lsd_km, lsd_km], 
        y=[-y_limit_lower, y_limit_upper],
        mode='lines',
        line=dict(width=10),
        name='Detector'
    ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        width=1000,
        height=695,
        title_text='Setup simulation'
    )
    
    return fig


def create_detector_figure(frame_data: np.ndarray, params: SimulationParams,
                          show_rings: bool, view_state: DetectorViewState,
                          hkl_data: Optional[np.ndarray] = None,
                          zmax: float = 8000) -> go.Figure:
    """Create detector visualization figure."""
    fig = go.Figure()
    px_mm = params.px_mm

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=frame_data,
        x0=-int(params.xbc * px_mm),
        y0=-int(params.ybc * px_mm),
        dx=px_mm,
        dy=px_mm,
        zmin=0,
        zmax=zmax,
        zauto=False,
        colorscale='gray_r'
    ))

    # Add rings if requested
    if show_rings and hkl_data is not None:
        try:
            rads = hkl_data[:, -1] / 1000  # µm → mm
            hkls_int = hkl_data[:, :3].astype(int)
            unique_rads = np.unique(rads)

            max_rad = np.max([
                1.414 * params.xbc,
                1.414 * params.ybc,
                1.414 * (params.n_pixels - params.xbc),
                1.414 * (params.n_pixels - params.ybc)
            ]) * px_mm

            for rad in unique_rads:
                if rad > max_rad:
                    continue
                idx = np.where(rads == rad)[0][0]
                hkl = np.abs(hkls_int[idx])
                label = f'HKL {hkl[0]} {hkl[1]} {hkl[2]}'
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=-rad + 0.4, y0=-rad + 0.4,
                    x1=rad + 0.4, y1=rad + 0.4,
                    name=label
                )
        except Exception as e:
            print(f"Error adding rings to detector figure: {e}")

    # Update layout
    fig.update_layout(
        xaxis_range=[view_state.ranges[0], view_state.ranges[1]],
        yaxis_range=[view_state.ranges[2], view_state.ranges[3]],
        margin=dict(l=0, r=0, b=0, t=50),
        width=750,
        height=700,
        title_text='2D Detector'
    )

    return fig


def create_microstructure_figure(mic_data: MicrostructureData, euler_val: int, 
                               title: str) -> go.Figure:
    """Create microstructure visualization figure."""
    fig = go.Figure()
    
    # Create appropriate trace based on plot type
    if mic_data.plot_type == 3:
        fig.add_trace(go.Scatter3d(
            x=mic_data.positions[:,0],
            y=mic_data.positions[:,1],
            z=mic_data.positions[:,2],
            mode='markers',
            marker=dict(
                color=mic_data.eulers[:,euler_val],
                showscale=True
            )
        ))
    else:
        fig.add_trace(go.Scatter(
            x=mic_data.positions[:,0],
            y=mic_data.positions[:,1],
            mode='markers',
            marker=dict(
                color=mic_data.eulers[:,euler_val],
                showscale=True
            )
        ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        width=700,
        height=700,
        title_text=title
    )
    
    return fig


def create_sequence_figure(seq_data: pd.DataFrame, seq_nr: int) -> go.Figure:
    """Create microstructure visualization for sequence data."""
    fig = go.Figure()
    
    # Extract data
    x = seq_data['X']
    y = seq_data['Y']
    z = seq_data['Z']
    color = seq_data['eKen33']
    
    # Create 3D scatter plot
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            color=color,
            colorscale='Jet',
            showscale=True
        )
    ))
    
    # Update layout
    median_strain = np.median(color)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        width=700,
        height=700,
        title_text=f'Microstructure SequenceNr {seq_nr} coloring by Strain (loading direction) Median: {median_strain:.6f}'
    )
    
    return fig


# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------

def create_app_layout(mic_fn: str, params: SimulationParams) -> html.Div:
    """Create the Dash app layout."""
    return dbc.Container([
        dbc.Row([
            html.Div('MIDAS Digital Twin', className="text-primary text-center fs-3")
        ]),
        
        # First row with microstructure and setup
        dbc.Row([
            # Left column - Input Microstructure
            dbc.Col([
                dbc.Row([html.H6("Input Microstructure")]),
                dbc.Row([
                    html.Div([
                        'Euler Component to plot: ',
                        dcc.RadioItems([0, 1, 2], 0, id='EulerComponent', inline=True),
                    ]),
                ]),
                dbc.Row([dcc.Graph(figure={}, id='microstructure')]),
            ]),
            
            # Right column - Setup Simulation
            dbc.Col([
                dbc.Row([html.H6('Setup Simulation')]),
                dbc.Row([html.H6('This will show the setup according to parameters chosen below.')]),
                dbc.Row([html.H6('Click Show Setup to generate plot of experimental setup.')]),
                dbc.Row([dcc.Graph(figure={}, id='setup')]),
            ]),
        ]),
        
        # Separator
        dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),
        
        # Button to show setup
        dbc.Row([
            dbc.Col([
                html.Button("Show Setup", id='button_run_sim', className="btn btn-primary")
            ], width={"size": 2}),
            dbc.Col([
                html.Div(id='setup_status', children="")
            ])
        ]),
        
        # Separator
        dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),
        
        # Parameter inputs
        dbc.Row([html.H6("Input parameters: ")]),
        
        # Parameter rows
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Detector distance [um]: ",
                    dcc.Input(id='lsd', value=params.lsd, type='number', style={'width': 100})
                ])
            ], width={"size": 6}),
            dbc.Col([
                html.Div(id='lsd_validation', className="text-danger")
            ], width={"size": 6})
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Beam Center (H,V) [px]: ",
                    dcc.Input(id='xbc', value=params.xbc, type='number', style={'width': 90}),
                    dcc.Input(id='ybc', value=params.ybc, type='number', style={'width': 90})
                ])
            ], width={"size": 6}),
            dbc.Col([
                html.Div(id='bc_validation', className="text-danger")
            ], width={"size": 6})
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    "X-Ray Energy [keV]: ",
                    dcc.Input(id='energy', value=params.energy, type='number', style={'width': 100})
                ])
            ], width={"size": 6}),
            dbc.Col([
                html.Div(id='energy_validation', className="text-danger")
            ], width={"size": 6})
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Space Group Nr: ",
                    dcc.Input(id='sg', value=params.sg, type='number', style={'width': 50})
                ])
            ], width={"size": 6}),
            dbc.Col([
                html.Div(id='sg_validation', className="text-danger")
            ], width={"size": 6})
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Lattice Constants [Å,°]: ",
                    dcc.Input(id='a', value=params.a, type='number', style={'width': 100}),
                    dcc.Input(id='b', value=params.b, type='number', style={'width': 100}),
                    dcc.Input(id='c', value=params.c, type='number', style={'width': 100}),
                    dcc.Input(id='alpha', value=params.alpha, type='number', style={'width': 60}),
                    dcc.Input(id='beta', value=params.beta, type='number', style={'width': 60}),
                    dcc.Input(id='gamma', value=params.gamma, type='number', style={'width': 60})
                ])
            ], width={"size": 10}),
            dbc.Col([
                html.Div(id='lattice_validation', className="text-danger")
            ], width={"size": 2})
        ]),
        
        # Separator
        dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),

        # Additional detector parameters
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Pixel Size [µm]: ",
                    dcc.Input(id='pixelsz', value=params.pixel_size, type='number', style={'width': 100}),
                ])
            ], width={"size": 4}),
            dbc.Col([
                html.Div([
                    "NrPixels: ",
                    dcc.Input(id='NrPixels', value=params.n_pixels, type='number', style={'width': 100}),
                ])
            ], width={"size": 4}),
        ]),

        # Separator
        dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),

        # Simulation type selector
        dbc.Row([
            html.Div([
                html.H6('Simulation type', className="text-primary text-left fs-3"),
                dcc.RadioItems([
                    {'label': html.Span('Far-Field HEDM [point-cloud maps] or Powder Diffraction',
                                       style={'font-size': 15, 'padding-left': 10, 'padding-right': 20}),
                     'value': 'FF-HEDM'},
                    {'label': html.Span('Near-Field HEDM [space-filling maps]',
                                       style={'font-size': 15, 'padding-left': 10, 'padding-right': 20}),
                     'value': 'NF-HEDM'}
                ], 'FF-HEDM', id='SimType', inline=True),
            ]),
        ]),

        # Separator
        dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),

        # Beam size slider
        dbc.Row([
            dbc.Col([
                html.Div(['BeamSize [µm] (ONLY for FF-HEDM; click Show Setup and Run Simulation again to update)'])
            ]),
            dbc.Col([
                dcc.Slider(
                    1.0,
                    5000.0,
                    value=params.beam_size,
                    step=1,
                    marks={
                        1: {'label': '1.0'},
                        1000: {'label': '1000.0'},
                        2000: {'label': '2000.0'},
                        3000: {'label': '3000.0'},
                        4000: {'label': '4000.0'},
                        5000: {'label': '5000.0'},
                    },
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='BeamSize'
                )
            ]),
        ]),

        # Separator
        dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),

        # Button to run simulation
        dbc.Row([
            dbc.Col([
                html.Button("Run Microstructure Simulation", id='button_run_2d', className="btn btn-primary")
            ], width={"size": 3}),
            dbc.Col([
                html.Div(id='sim_status', children="")
            ])
        ]),
        
        # Frame selectors
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H6("Frame Number Selector")]),
                dbc.Row([
                    dcc.Slider(
                        0,
                        params.n_frames - 1,
                        step=1,
                        value=0,
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='FrameNr'
                    )
                ]),
            ]),
            dbc.Col([
                dbc.Row([html.H6("nFramesSum Selector")]),
                dbc.Row([
                    dcc.Slider(
                        1,
                        params.n_frames,
                        step=1,
                        value=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='nFramesSum'
                    )
                ]),
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Checklist(
                        options=['ShowRings'],
                        value=['ShowRings'],
                        id='Rings',
                    ),
                ]),
                dbc.Row([
                    dcc.Checklist(
                        options=['Sequence'],
                        value=[],
                        id='Multiple',
                    ),
                ]),
                dbc.Row([
                    dcc.Slider(
                        1,
                        2,  # Initial max value, will be updated dynamically
                        step=1,
                        value=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                        id='SequenceNr'
                    )
                ]),
            ]),
        ]),
        
        # Result displays
        dbc.Row([
            dbc.Col([
                dbc.Row([html.H6("Detector")]),
                dbc.Row([dcc.Graph(figure={}, id='detector')]),
            ]),
            dbc.Col([
                dbc.Row([html.H6("Microstructure")]),
                dbc.Row([dcc.Graph(figure={}, id='microstructure2')]),
            ]),
        ]),
        
        # Export options
        dbc.Row([
            dbc.Col([
                html.Button("Export Current View", id='button_export', className="btn btn-secondary mt-3")
            ], width={"size": 2}),
            dbc.Col([
                html.Div(id='export_status')
            ])
        ])
        
    ], fluid=True)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@callback(
    [Output('lsd_validation', 'children'),
     Output('bc_validation', 'children'),
     Output('energy_validation', 'children'),
     Output('sg_validation', 'children'),
     Output('lattice_validation', 'children')],
    [Input('lsd', 'value'),
     Input('xbc', 'value'),
     Input('ybc', 'value'),
     Input('energy', 'value'),
     Input('sg', 'value'),
     Input('a', 'value'),
     Input('b', 'value'),
     Input('c', 'value'),
     Input('alpha', 'value'),
     Input('beta', 'value'),
     Input('gamma', 'value')]
)
def validate_inputs(lsd, xbc, ybc, energy, sg, a, b, c, alpha, beta, gamma):
    """Validate user inputs and provide feedback."""
    lsd_error = "" if lsd > 0 else "Detector distance must be positive"
    bc_error = "" if 0 <= xbc <= DEFAULT_DETECTOR_PIXELS and 0 <= ybc <= DEFAULT_DETECTOR_PIXELS else "Beam center must be within detector"
    energy_error = "" if energy > 0 else "Energy must be positive"
    sg_error = "" if 1 <= sg <= 230 else "Space group must be between 1 and 230"
    lattice_error = "" if a > 0 and b > 0 and c > 0 else "Lattice constants must be positive"
    
    return lsd_error, bc_error, energy_error, sg_error, lattice_error


@callback(
    [Output('setup', 'figure'),
     Output('setup_status', 'children')],
    [Input('button_run_sim', 'n_clicks')],
    [State('lsd', 'value'),
     State('xbc', 'value'),
     State('ybc', 'value'),
     State('energy', 'value'),
     State('sg', 'value'),
     State('a', 'value'),
     State('b', 'value'),
     State('c', 'value'),
     State('alpha', 'value'),
     State('beta', 'value'),
     State('gamma', 'value'),
     State('BeamSize', 'value'),
     State('pixelsz', 'value'),
     State('NrPixels', 'value'),
     State('SimType', 'value')]
)
def update_setup(n_clicks, lsd, xbc, ybc, energy, sg, a, b, c, alpha, beta, gamma,
                beam_size, pixel_size, n_pixels, sim_type):
    """Update the setup visualization based on input parameters."""
    fig = go.Figure()
    status = ""

    if n_clicks is not None and n_clicks > 0:
        try:
            wl = 12.398 / energy
            params = SimulationParams(
                lsd=lsd, xbc=xbc, ybc=ybc, energy=energy, sg=sg,
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                beam_size=beam_size, pixel_size=pixel_size, n_pixels=n_pixels,
                sim_type=sim_type
            )

            # Write parameter file (FF for setup — GetHKLList needs it)
            param_fn = 'ps_midas_dtw.txt'
            write_parameter_file(params, mic_fn, param_fn)

            # Run GetHKLList in CLI stdout mode
            hkl_bin = os.path.join(install_path, 'FF_HEDM/bin/GetHKLList')
            lp = [str(v) for v in [params.a, params.b, params.c, params.alpha, params.beta, params.gamma]]
            cmd = [
                hkl_bin,
                '--sg', str(params.sg),
                '--lp'] + lp + [
                '--wl', str(wl),
                '--lsd', str(params.lsd),
                '--maxR', '404800',
                '--stdout',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Parse HKL data from stdout
                    hkl_rows = []
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 11:
                            hkl_rows.append([float(x) for x in parts])
                    hkls = np.array(hkl_rows)
                    # Save for ring overlay use
                    np.savetxt('hkls.csv', hkls, header='h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius',
                              fmt='%.0f %.0f %.0f %f %.0f %f %f %f %f %f %f', comments='')
                    fig = create_setup_figure(hkls, params)
                    status = html.Span("Setup successfully generated", className="text-success")
                else:
                    status = html.Span("GetHKLList produced no output", className="text-danger")
            else:
                status = html.Span("Error running GetHKLList", className="text-danger")
        except Exception as e:
            status = html.Span(f"Error: {str(e)}", className="text-danger")

    return fig, status


@callback(
    [Output('detector', 'figure'),
     Output('sim_status', 'children')],
    [Input('button_run_2d', 'n_clicks'),
     Input('FrameNr', 'value'),
     Input('nFramesSum', 'value'),
     Input('Rings', 'value')],
    [State('xbc', 'value'),
     State('ybc', 'value'),
     State('BeamSize', 'value'),
     State('Multiple', 'value'),
     State('SequenceNr', 'value'),
     State('SimType', 'value'),
     State('pixelsz', 'value'),
     State('NrPixels', 'value'),
     State('lsd', 'value'),
     State('energy', 'value'),
     State('sg', 'value'),
     State('a', 'value'),
     State('b', 'value'),
     State('c', 'value'),
     State('alpha', 'value'),
     State('beta', 'value'),
     State('gamma', 'value')]
)
def update_detector(n_clicks, frame_nr, n_frames_sum, show_rings, xbc, ybc,
                   beam_size, multiple_data, seq_nr, sim_type,
                   pixel_size, n_pixels, lsd, energy, sg, a, b, c, alpha, beta, gamma):
    """Update the 2D detector visualization."""
    fig = go.Figure()
    status = ""

    params = SimulationParams(
        lsd=lsd, xbc=xbc, ybc=ybc, energy=energy, sg=sg,
        a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
        beam_size=beam_size, pixel_size=pixel_size, n_pixels=n_pixels,
        sim_type=sim_type
    )
    view_state = DetectorViewState(params)
    px_mm = params.px_mm

    if n_clicks is None or n_clicks <= 0:
        # Show detector outline
        fig.add_trace(go.Scatter(
            x=[-(xbc * px_mm), ((n_pixels - xbc) * px_mm),
               ((n_pixels - xbc) * px_mm), -(xbc * px_mm), -(xbc * px_mm)],
            y=[-(ybc * px_mm), -(ybc * px_mm),
               ((n_pixels - ybc) * px_mm), ((n_pixels - ybc) * px_mm), -(ybc * px_mm)],
            mode='lines', line=dict(width=10), name='Detector'
        ))
        fig.update_layout(
            xaxis_range=[view_state.ranges[0], view_state.ranges[1]],
            yaxis_range=[view_state.ranges[2], view_state.ranges[3]],
            margin=dict(l=0, r=0, b=0, t=50), width=750, height=700,
            title_text='2D Detector'
        )
        return fig, status

    try:
        param_fn = 'ps_midas_dtw.txt'

        # Handle multiple sequences
        if multiple_data and 'Sequence' in multiple_data:
            mic_fn_loc = update_parameter_file_for_sequence(param_fn, mic_fn, seq_nr)
        else:
            mic_fn_loc = mic_fn

        # Load HKL data for ring overlay
        hkl_data = None
        try:
            hkl_data = np.genfromtxt('hkls.csv', skip_header=1, delimiter=' ')
        except Exception:
            pass

        show_rings_list = show_rings if isinstance(show_rings, list) else []
        do_show_rings = 'ShowRings' in show_rings_list

        # ── NF-HEDM mode ──────────────────────────────────────────
        if sim_type == 'NF-HEDM':
            if ctx.triggered_id == 'button_run_2d':
                write_nf_parameter_file(params, mic_fn_loc, param_fn)
                if hasattr(midas_config, 'MIDAS_NF_BIN_DIR') and midas_config.MIDAS_NF_BIN_DIR:
                    sim_bin = os.path.join(midas_config.MIDAS_NF_BIN_DIR, 'simulateNF')
                else:
                    sim_bin = os.path.join(install_path, 'NF_HEDM/bin/simulateNF')
                cmd = [sim_bin, param_fn, mic_fn_loc, f'{mic_fn_loc}.result']
                if run_subprocess(cmd) != 0:
                    return fig, html.Span("Error running simulateNF", className="text-danger")
                status = html.Span("NF simulation completed", className="text-success")

            # Read NF result (uint16 binary with 8192-byte header)
            result_file = f'{mic_fn_loc}.result'
            if not os.path.exists(result_file):
                return fig, html.Span("No NF result file found. Click Run first.", className="text-warning")
            n_frames = params.n_frames
            frame_nr = int(frame_nr)
            n_frames_sum = int(n_frames_sum)
            zf = np.fromfile(result_file, offset=8192, dtype=np.uint16,
                            count=n_frames * n_pixels * n_pixels).reshape((n_frames, n_pixels, n_pixels))
            last_frame = min(frame_nr + n_frames_sum, n_frames)
            frame = np.max(zf[frame_nr:last_frame, :, :], axis=0)
            frame = np.transpose(frame)

            fig = create_detector_figure(
                frame, params, do_show_rings, view_state,
                hkl_data=hkl_data, zmax=1
            )

        # ── FF-HEDM mode ──────────────────────────────────────────
        else:
            if ctx.triggered_id == 'button_run_2d':
                cmd = [os.path.join(install_path, 'FF_HEDM/bin/ForwardSimulationCompressed'), param_fn]
                if run_subprocess(cmd) != 0:
                    return fig, html.Span("Error running simulation", className="text-danger")
                status = html.Span("FF simulation completed", className="text-success")

            sim_zip = f'{mic_fn_loc}.sim_scanNr_0.zip'
            if not os.path.exists(sim_zip):
                return fig, html.Span("No FF result file found. Click Run first.", className="text-warning")
            zf = zarr.open(sim_zip, 'r')['exchange/data']
            frame_nr = int(frame_nr)
            n_frames_sum = int(n_frames_sum)
            n_frames = params.n_frames
            last_frame = min(frame_nr + n_frames_sum, n_frames)
            frame = np.max(zf[frame_nr:last_frame, :, :], axis=0)

            fig = create_detector_figure(
                frame, params, do_show_rings, view_state,
                hkl_data=hkl_data, zmax=8000
            )

    except Exception as e:
        return fig, html.Span(f"Error: {str(e)}", className="text-danger")

    return fig, status


@callback(
    [Output('microstructure', 'figure'),
     Output('SequenceNr', 'max')],
    [Input('EulerComponent', 'value'),
     Input('Multiple', 'value'),
     Input('SequenceNr', 'value')]
)
def update_microstructure(euler_val, multiple_data, seq_nr):
    """Update microstructure visualization."""
    # Default return values
    fig = go.Figure()
    seq_max = 2  # Default max
    
    try:
        # Handle sequence data if multiple data is enabled
        if multiple_data and 'Sequence' in multiple_data:
            try:
                # Load sequence data
                seq_nr = int(seq_nr)
                seq_data = load_sequence_data(seq_nr)
                
                # Count available sequences
                all_files = glob.glob('Grains.csv.*')
                all_files = [f for f in all_files if not 'zip' in f]
                seq_max = len(all_files) + 1 if all_files else 2
                
                # Create figure
                fig = create_sequence_figure(seq_data, seq_nr)
                
            except Exception as e:
                print(f"Error loading sequence data: {str(e)}")
                # Create empty figure with error message
                fig.update_layout(
                    annotations=[{
                        'text': f"Error loading sequence data: {str(e)}",
                        'showarrow': False,
                        'font': {'color': 'red'}
                    }]
                )
        else:
            # Load regular microstructure data
            try:
                mic_data = load_microstructure_data(mic_fn)
                euler_val = int(euler_val)  # Ensure euler_val is an integer
                fig = create_microstructure_figure(
                    mic_data, 
                    euler_val,
                    f'Microstructure coloring by Euler{euler_val}'
                )
            except Exception as e:
                print(f"Error loading microstructure data: {str(e)}")
                # Create empty figure with error message
                fig.update_layout(
                    annotations=[{
                        'text': f"Error loading microstructure data: {str(e)}",
                        'showarrow': False,
                        'font': {'color': 'red'}
                    }]
                )
    except Exception as e:
        print(f"Error in update_microstructure: {str(e)}")
        # Create empty figure with error message
        fig.update_layout(
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'font': {'color': 'red'}
            }]
        )
    
    return fig, seq_max


@callback(
    Output('microstructure2', 'figure'),
    [Input('EulerComponent', 'value'),
     Input('button_run_2d', 'n_clicks')],
    [State('Multiple', 'value'),
     State('SequenceNr', 'value'),
     State('BeamSize', 'value')]
)
def update_filtered_microstructure(euler_val, n_clicks, multiple_data, seq_nr, beam_size):
    """Update filtered microstructure visualization based on beam size."""
    # Initialize with empty figure
    fig = go.Figure()
    
    if n_clicks is None or n_clicks <= 0:
        return fig
    
    try:
        # Handle sequence data if multiple data is enabled
        if multiple_data and 'Sequence' in multiple_data:
            try:
                # Load sequence data
                seq_nr = int(seq_nr)
                seq_data = load_sequence_data(seq_nr)
                
                # Filter by beam size if necessary
                # This is a simplified version; for real implementation, 
                # you would need to filter the DataFrame based on position
                if beam_size < float('inf'):
                    x = seq_data['X']
                    y = seq_data['Y']
                    rad = np.sqrt(x**2 + y**2)
                    seq_data = seq_data[rad < beam_size]
                
                # Create figure
                fig = create_sequence_figure(seq_data, seq_nr)
                
            except Exception as e:
                print(f"Error loading sequence data: {str(e)}")
                # Create empty figure with error message
                fig.update_layout(
                    annotations=[{
                        'text': f"Error loading sequence data: {str(e)}",
                        'showarrow': False,
                        'font': {'color': 'red'}
                    }]
                )
        else:
            # Load filtered microstructure data
            try:
                mic_data = load_microstructure_data(mic_fn, beam_size)
                euler_val = int(euler_val)  # Ensure euler_val is an integer
                fig = create_microstructure_figure(
                    mic_data, 
                    euler_val,
                    f'Filtered Microstructure (beam size: {beam_size} µm) coloring by Euler{euler_val}'
                )
            except Exception as e:
                print(f"Error loading microstructure data: {str(e)}")
                # Create empty figure with error message
                fig.update_layout(
                    annotations=[{
                        'text': f"Error loading microstructure data: {str(e)}",
                        'showarrow': False,
                        'font': {'color': 'red'}
                    }]
                )
    except Exception as e:
        print(f"Error in update_filtered_microstructure: {str(e)}")
        # Create empty figure with error message
        fig.update_layout(
            annotations=[{
                'text': f"Error: {str(e)}",
                'showarrow': False,
                'font': {'color': 'red'}
            }]
        )
    
    return fig


@callback(
    Output('detector', 'relayoutData'),
    [Input('detector', 'relayoutData')],
    prevent_initial_call=True
)
def save_relayout_data(relayout_data):
    """Save detector layout data for maintaining view state."""
    return relayout_data


@callback(
    Output('export_status', 'children'),
    [Input('button_export', 'n_clicks')],
    [State('detector', 'figure'),
     State('microstructure2', 'figure')],
    prevent_initial_call=True
)
def export_results(n_clicks, detector_fig, microstructure_fig):
    """Export current visualization results."""
    if n_clicks is None or n_clicks <= 0:
        return ""
    
    try:
        # Create export directory if it doesn't exist
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detector figure
        if detector_fig:
            detector_path = os.path.join(export_dir, f"detector_{timestamp}.png")
            import plotly.io as pio
            pio.write_image(detector_fig, detector_path)
        
        # Export microstructure figure
        if microstructure_fig:
            microstructure_path = os.path.join(export_dir, f"microstructure_{timestamp}.png")
            import plotly.io as pio
            pio.write_image(microstructure_fig, microstructure_path)
        
        # Export parameters
        param_path = os.path.join(export_dir, f"params_{timestamp}.txt")
        with open('ps_midas_dtw.txt', 'r') as src_file:
            with open(param_path, 'w') as dst_file:
                dst_file.write(src_file.read())
        
        return html.Span(f"Exported to {export_dir} directory", className="text-success")
    except Exception as e:
        return html.Span(f"Export error: {str(e)}", className="text-danger")


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    global mic_fn
    mic_fn = args.mic
    
    # Check if microstructure file exists
    if not os.path.isfile(mic_fn):
        print(f"Error: Microstructure file {mic_fn} does not exist")
        sys.exit(1)
    
    # Create default parameters
    params = SimulationParams()
    
    # Check if omega range is valid
    if params.n_frames <= 0:
        print("Error: Invalid omega range or step size")
        sys.exit(1)
    
    # Initialize Dash app
    external_stylesheets = [dbc.themes.CYBORG]
    app = Dash(
        __name__, 
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True
    )
    app.title = "MIDAS DIGITAL TWIN"
    
    # Set up app layout
    app.layout = create_app_layout(mic_fn, params)
    
    # Run the app
    app.run(debug=False, host='0.0.0.0')


# MIDAS version banner
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'utils'))
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == '__main__':
    main()