#!/usr/bin/env python

from dash import Dash, html, dcc, callback, Output, Input, State
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import argparse
import sys
import subprocess
import zarr
import os, sys
from math import cos, sin
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from calcMiso import *
import glob

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''MIDAS Visualization of caking data''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-fn', type=str, required=True, help='File name of caking file')
args, unparsed = parser.parse_known_args()
fn = args.fn

external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MIDAS CAKING VISUALIZATION"

zf = zarr.open('/Users/hsharma/Desktop/analysis/test_calibrant/2024_08_21/shade_LSHR_voi_ff_000304.ge3.analysis.MIDAS.zip.caked.hdf.zarr.zip','r')
nDsets = len(zf['OmegaSumFrame'])
dSetNrs = np.sort(np.array([int(i.split('_')[1]) for i in zf['OmegaSumFrame']]))

app.layout = dbc.Container([
    dbc.Row([
        html.Div('MIDAS CAKING VISUALIZATION', className="text-primary text-center fs-3")
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([f'OmegaSumFrame number to plot: (choose between {0} to {nDsets-1})',dcc.Input(id='frameNr',value=0,type='number',style={'width':100})])
        ])
    ]),
    dbc.Row([
        dcc.Graph(figure={},id='ShowFrames')
    ])
])

@callback(
    Output(component_id='ShowFrames',component_property='figure'),
    Input('frameNr','value')
)
def plotFrames(frameNr):
    fig = go.Figure()
    frameNr = int(frameNr)
    if frameNr<0: frameNr=0
    if frameNr>nDsets-1: frameNr=nDsets-1
    thisFrame = zf[f'OmegaSumFrame/LastFrameNumber_{dSetNrs[frameNr]}'][:]
    fig.add_trace(go.Heatmap(z=thisFrame,colorscale='gray_r'))
    fig.update_layout(width=1000,height=1000)
    return fig

app.run(debug=False,host='0.0.0.0')
