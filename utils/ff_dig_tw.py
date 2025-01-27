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

##### TODO:
###### READ Z-values from EBSD, then plotType2 = 3....
###### Select beam height, beam width, beam offsets, beam projection according to angle.....
###### ForwardSimulationCompressed read EBSD by detecting EBSD files.
###### Check which simulation to do for pf-HEDM in ForwardSimulationCompressed



deg2rad = np.pi/180
rad2deg = 180/np.pi

env = dict(os.environ)
midas_path = os.path.expanduser("~/.MIDAS")
libpth = os.environ.get('LD_LIBRARY_PATH','')
env['LD_LIBRARY_PATH'] = f'{midas_path}/BLOSC/lib64:{midas_path}/FFTW/lib:{midas_path}/HDF5/lib:{midas_path}/LIBTIFF/lib:{midas_path}/LIBZIP/lib64:{midas_path}/NLOPT/lib:{midas_path}/ZLIB/lib:{libpth}'

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(description='''MIDAS DIGITAL TWIN''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-mic', type=str, required=True, help='File name of microstructure file')
args, unparsed = parser.parse_known_args()
micfn = args.mic
norig = 0
norig2 = 0
seqOrig = 0
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MIDAS DIGITAL TWIN"
paramFN = 'ps_midas_dtw.txt'
minOme = -10
maxOme = 10
omeStep = 0.25
nFrames = (maxOme - minOme) // omeStep
pos = None
eulers = None
plotType = 1
pos2 = None
eulers2 = None
plotType2 = 1
a = 2.925390
b = 2.925390
c = 4.673990
alpha = 90
beta = 90
gamma = 120
BC_or = -1
sg = 194
maxSeq = 2
minSeq = 1
ranges = [0,0,0,0]

if (nFrames<0):
    sys.exit()

app.layout = dbc.Container([
    dbc.Row([
        html.Div('MIDAS Digital Twin', className="text-primary text-center fs-3")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([html.H6("Input Microstructure"),]),
            dbc.Row([
                html.Div([
                    'Euler Component to plot: ',
                    dcc.RadioItems([0, 1,2], 0,id='EulerComponent',inline=True),
                ]),
            ]),
            dbc.Row([dcc.Graph(figure={}, id='microstructure'),]),
        ]),
        dbc.Col([
            dbc.Row([html.H6('Setup Simulation')]),
            dbc.Row([html.H6('This will show the setup according to parameters chosen below.')]),
            dbc.Row([html.H6('Click Show Setup to generate plot of experimental setup.')]),
            dbc.Row([
                dcc.Graph(figure={}, id='setup')
            ]),
        ]),
    ]),
    dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),
    dbc.Row([
        html.Button("Show Setup",id='button_run_sim')
    ]),
    dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),
    dbc.Row([
        html.H6("Input parameters: "),
    ]),
    dbc.Row([
        html.Div([
            "Detector distance [um]: ",
            dcc.Input(id='lsd', value=1000000, type='number',style={'width':100}),
        ]),
    ]),
    dbc.Row([
        html.Div([
            "Beam Center (H,V) [px]: ",
            dcc.Input(id='xbc', value=1024.0, type='number',style={'width':90}),
            dcc.Input(id='ybc', value=1024.0, type='number',style={'width':90}),
        ]),
    ]),
    dbc.Row([
        html.Div([
            ".......X-Ray Energy [keV]: ",
            dcc.Input(id='energy', value=71.676, type='number',style={'width':100}),
        ]),
    ]),
    dbc.Row([
        html.Div([
            "............Space Group Nr: ",
            dcc.Input(id='sg', value=sg, type='number',style={'width':50}),
        ]),
    ]),
    dbc.Row([
        html.Div([
            "...Lattice Constant [Å,°]: ",
            dcc.Input(id='a', value=a, type='number',style={'width':100}),
            dcc.Input(id='b', value=b, type='number',style={'width':100}),
            dcc.Input(id='c', value=c, type='number',style={'width':100}),
            dcc.Input(id='alpha', value=alpha, type='number',style={'width':60}),
            dcc.Input(id='beta', value=beta, type='number',style={'width':60}),
            dcc.Input(id='gamma', value=gamma, type='number',style={'width':60}),
        ]),
    ]),
    dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),
    dbc.Row([
        dbc.Col([
            html.Div(['BeamSize [um] (click Show Setup and Run Microstructure Simulation again to update)',])
        ]),
        dbc.Col([
            dcc.Slider(
                1.0,
                5000.0,
                value = 2000.0,
                step=1,
                marks={1:{'label':'1.0'},
                       1000:{'label':'1000.0'},
                       2000:{'label':'2000.0'},
                       3000:{'label':'3000.0'},
                       4000:{'label':'4000.0'},
                       5000:{'label':'5000.0'},},
                tooltip={"placement": "bottom", "always_visible": True},
                id='BeamSize'
            )
        ]),
    ]),
    dbc.Row([html.Hr(style={'borderWidth': "1.3vh"})]),
    dbc.Row([
        html.Button("Run Microstructure Simulation",id='button_run_2d')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.H6("Frame Number Selector"),
            ]),
            dbc.Row([
                dcc.Slider(
                    0,
                    nFrames-1,
                    step=1,
                    value = 0,
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='FrameNr'
                )
            ]),
        ]),
        dbc.Col([
            dbc.Row([
                html.H6("nFramesSum Selector"),
            ]),
            dbc.Row([
                dcc.Slider(
                    1,
                    nFrames,
                    step=1,
                    value = 1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='nFramesSum'
                )
            ]),
        ]),
        dbc.Col([
            dbc.Row([
                dcc.Checklist(
                    options=['ShowRings'],
                    value = ['ShowRings'],
                    id='Rings',
                ),
            ]),
            dbc.Row([
                dcc.Checklist(
                        options=['Sequence'],
                        value = [],
                        id='Multiple',
                ),
            ]),
            dbc.Row([
                dcc.Slider(
                    minSeq,
                    maxSeq,
                    step=1,
                    value = 1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='SequenceNr'
                )
            ]),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.H6("Detector"),
            ]),
            dbc.Row([
                dcc.Graph(figure={}, id='detector')
            ]),
        ]),
        dbc.Col([
            dbc.Row([
                html.H6("Microstructure"),
            ]),
            dbc.Row([
                dcc.Graph(figure={}, id='microstructure2')
            ]),
        ]),
    ]),

],fluid=True)

@callback(
    Output(component_id='setup', component_property='figure'),
    Input("button_run_sim","n_clicks"),
    Input("lsd", "value"),
    Input("xbc", "value"),
    Input("ybc", "value"),
    Input("energy", "value"),
    Input("sg", "value"),
    Input("a", "value"),
    Input("b", "value"),
    Input("c", "value"),
    Input("alpha", "value"),
    Input("beta", "value"),
    Input("gamma", "value"),
    Input("BeamSize", "value"),
)
def update_setup(nclk,lsd,xbc,ybc,energy,sg,a,b,c,alpha,beta,gamma,bsz):
    global norig
    if nclk is not None:
        if nclk > norig:
            norig = nclk
            pF = open(paramFN,'w')
            pF.write(f"Lsd {lsd}\n")
            pF.write(f"BC {xbc} {ybc}\n")
            pF.write(f"SpaceGroup {sg}\n")
            pF.write(f"LatticeParameter {a} {b} {c} {alpha} {beta} {gamma}\n")
            pF.write(f"tx 0\n")
            pF.write(f"ty 0\n")
            pF.write(f"tz 0\n")
            pF.write(f"p0 0\n")
            pF.write(f"p1 0\n")
            pF.write(f"p2 0\n")
            pF.write(f"p3 0\n")
            pF.write(f"Wedge 0\n")
            pF.write(f"RhoD 404800\n")
            pF.write(f"OmegaStep {omeStep}\n")
            pF.write(f"OmegaStart {minOme}\n")
            pF.write(f"OmegaEnd {maxOme}\n")
            print(minOme)
            pF.write(f"Wavelength {12.398/energy}\n")
            pF.write(f"NrPixels 2048\n")
            pF.write(f"px 200\n")
            pF.write(f"GaussWidth 2\n")
            pF.write(f"PeakIntensity 5000\n")
            pF.write(f"RingThresh 1 50\n")
            pF.write(f"RingThresh 2 50\n")
            pF.write(f"RingThresh 3 50\n")
            pF.write(f"RingThresh 4 50\n")
            pF.write(f"RingThresh 5 50\n")
            pF.write(f"RingThresh 6 50\n")
            pF.write(f"RingThresh 7 50\n")
            pF.write(f"RingThresh 8 50\n")
            pF.write(f"RingThresh 9 50\n")
            pF.write(f"RingThresh 10 50\n")
            pF.write(f"RingThresh 11 50\n")
            pF.write(f"RingThresh 12 50\n")
            pF.write(f"RingThresh 13 50\n")
            pF.write(f"RingThresh 14 50\n")
            pF.write(f"RingThresh 15 50\n")
            pF.write(f"RingThresh 16 50\n")
            pF.write(f"RingThresh 17 50\n")
            pF.write(f"RingThresh 18 50\n")
            pF.write(f"RingThresh 19 50\n")
            pF.write(f"RingThresh 20 50\n")
            pF.write(f"RingThresh 21 50\n")
            pF.write(f"RingThresh 22 50\n")
            pF.write(f"RingThresh 23 50\n")
            pF.write(f"RingThresh 24 50\n")
            pF.write(f"RingThresh 25 50\n")
            pF.write(f"RingThresh 26 50\n")
            pF.write(f"RingThresh 27 50\n")
            pF.write(f"RingThresh 28 50\n")
            pF.write(f"RingThresh 29 50\n")
            pF.write(f"RingThresh 30 50\n")
            pF.write(f"WriteSpots 1\n")
            pF.write(f"InFileName {micfn}\n")
            pF.write(f"OutFileName {micfn}.sim\n")
            pF.write(f"BeamSize {float(bsz)}\n")
            pF.close()
            cmmd = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/GetHKLList')+f' {paramFN}'
            subprocess.call(cmmd,shell=True)
            hkls = np.genfromtxt('hkls.csv',skip_header=1,delimiter=' ')
            rads = hkls [:,-1] / 1000
            rads = rads.astype(int)
            hkls = hkls[:,:3].astype(int)
            unique_rads = np.unique(rads)
            fig = go.Figure()
            for rad in unique_rads:
                found = 0
                for radnr in range(len(rads)):
                    if rads[radnr] == rad and found == 0:
                        hkl = np.abs(hkls[radnr])
                        found = 1
                label = f'HKL {hkl[0]} {hkl[1]} {hkl[2]}'
                if rad < int(ybc*0.2) and rad < int((2048-ybc)*0.2):
                    fig.add_trace(go.Scatter(x=[int(lsd/1000),0,int(lsd/1000)], y=[-rad,0,rad],name=label))
                elif rad > int(ybc*0.2) and rad < int((2048-ybc)*0.2):
                    fig.add_trace(go.Scatter(x=[0,int(lsd/1000)], y=[0,rad],name=label))
                elif rad < int(ybc*0.2) and rad > int((2048-ybc)*0.2):
                    fig.add_trace(go.Scatter(x=[int(lsd/1000),0], y=[-rad,0],name=label))
            fig.update_traces(mode='lines')
            fig.add_trace(go.Scatter(x=[-500],y=[-10],mode='text',text="X-ray Beam",textfont=dict(size=18),textposition='bottom center',name='beam'))
            fig.add_trace(go.Scatter(x=[-1000,0], y=[0,0],line=dict(width=10),marker=dict(size=10,symbol= "arrow-bar-up", angleref="previous"),name='beam'))
            fig.add_trace(go.Scatter(x=[int(lsd/1000),int(lsd/1000)], y=[-int(ybc*0.2),int((2048-ybc)*0.2)],mode='lines',line=dict(width=10),name='Detector'))
        else:
            fig = go.Figure()
    else:
        fig = go.Figure()
    fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),width=1000,height=695,
            title_text='Setup simulation')
    return fig

@callback(
    Output(component_id='detector', component_property='figure'),
    Input("button_run_2d","n_clicks"),
    Input("xbc", "value"),
    Input("ybc", "value"),
    Input("FrameNr", "value"),
    Input("nFramesSum", "value"),
    Input("Rings", "value"),
    Input('Multiple','value'),
    Input('SequenceNr','value'),
)
def update_2d(nclk,xbc,ybc,frameNr,nFramesSum,showRings,mult,seq):
    nFramesSum = int(nFramesSum)
    frameNr = int(frameNr)
    global norig2, seqOrig, ranges
    if mult:
        micst = '.'.join(micfn.split('.')[:-1])
        micfnLoc = micst + f'.{int(seq)}'
        lines = open(paramFN,'r').readlines()
        pF = open(paramFN,'w')
        for line in lines:
            if line.startswith('InFileName'):
                pF.write(f"InFileName {micfnLoc}\n")
            elif line.startswith('OutFileName'):
                pF.write(f"OutFileName {micfnLoc}.sim\n")
            else:
                pF.write(line)
        pF.close()
    else:
        micfnLoc = micfn
    maxRad = np.max([1.414*(xbc),1.414*ybc,1.141*(2048-xbc),1.141*(2048-ybc)])*0.2
    if nclk is not None:
        if nclk > 0:
            if nclk > norig2 or seqOrig!=int(seq):
                seqOrig = int(seq)
                norig2 = nclk
                cmmd = os.path.expanduser('~/opt/MIDAS/FF_HEDM/bin/ForwardSimulationCompressed')+f' {paramFN}'
                subprocess.call(cmmd,shell=True,env=env)
            zf = zarr.open(f'{micfnLoc}.sim_scanNr_0.zip','r')['exchange/data']
            lastFrame = frameNr+ nFramesSum
            # print(frameNr,nFramesSum,lastFrame)
            if lastFrame > nFrames:
                lastFrame = int(nFrames)
            print(frameNr,nFramesSum,lastFrame)
            frame = zf[frameNr:lastFrame,:,:]
            frame = np.max(frame,axis=0)
            hkls = np.genfromtxt('hkls.csv',skip_header=1,delimiter=' ')
            rads = hkls [:,-1] / 1000
            hkls = hkls[:,:3].astype(int)
            unique_rads = np.unique(rads)
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=frame,x0=-int(xbc*0.2), y0=-int(ybc*0.2),dx=0.2,dy=0.2,zmin=0,zmax=8000,zauto=False,colorscale='gray_r'))
            if showRings:
                for rad in unique_rads:
                    found = 0
                    if rad > maxRad:
                        continue
                    for radnr in range(len(rads)):
                        if rads[radnr] == rad and found == 0:
                            hkl = np.abs(hkls[radnr])
                            found = 1
                    label = f'HKL {hkl[0]} {hkl[1]} {hkl[2]}'
                    fig.add_shape(type="circle",
                        xref="x", yref="y",
                        x0=-rad+0.4, y0=-rad+0.4, x1=rad+0.4, y1=rad+0.4,name=label
                    )
            if ranges[0] == 0:
                ranges = [-int(xbc*0.2)-2,int((2048-xbc)*0.2)+2,-int(ybc*0.2)-2,int((2048-ybc)*0.2)+2]
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[-int(xbc*0.2),int((2048-xbc)*0.2),int((2048-xbc)*0.2),-int(xbc*0.2),-int(xbc*0.2)], y=[-int(ybc*0.2),-int(ybc*0.2),int((2048-ybc)*0.2),int((2048-ybc)*0.2),-int(ybc*0.2)],mode='lines',line=dict(width=10),name='Detector'))
    fig.update_layout(
            xaxis_range=[ranges[0], ranges[1]],
            yaxis_range = [ranges[2],ranges[3]],
            margin=dict(l=0, r=0, b=0, t=50),width=750,height=700,
            title_text='2D Detector')
    return fig

@callback(
    Output(component_id='microstructure', component_property='figure'),
    Output('SequenceNr','max'),
    Input('EulerComponent','value'),
    Input('Multiple','value'),
    Input('SequenceNr','value')
)
def show_mic(eulerVal,multipleData,selLoadNr):
    global pos, eulers, plotType
    mic_h = open(micfn,'r').readline()
    fig = go.Figure()
    seqMax = 0
    if not multipleData:
        if pos is None:
            if mic_h.startswith('%NumGrains'):
                # Grains.csv
                mic = np.genfromtxt(micfn,skip_header=9)
                pos = mic[:,10:13]
                orients = mic[:,1:10]
                eulers = np.zeros((mic.shape[0],3))
                for rowNr in range(orients.shape[0]):
                    orient = orients[rowNr]
                    det = np.linalg.det(orient.reshape(3,3))
                    orient = orient / det
                    eulers[rowNr] = rad2deg * OrientMat2Euler(orient)
                plotType = 3
            elif mic_h.startswith('%TriEdgeSize'):
                # NF Mic, 2D
                mic = np.genfromtxt(micfn,skip_header=4)
                pos = mic[:,3:6]
                pos[:,2] = 0
                eulers = mic[:,7:10]
                plotType = 2
            elif mic_h.startswith('# SpotID'):
                # pf-Mic, 2D
                mic = np.genfromtxt(micfn,skip_header=1,delimiter=',')
                pos = mic[:,11:14]
                eulers = mic[:,-7:-4]
                plotType = 2
            elif mic_h.startswith('#EBSD'):
                # This is an EBSD CSV file written by Nicolas
                # Which column has what....
                # #EBSD X, Y, Z, Euler Angle1, Euler Angle2, Euler Angle3
                mic = np.genfromtxt(micfn,skip_header=1,delimiter=',')
                rads = np.linalg.norm(mic[:,0:2],axis=1)
                # mic = mic[:,:]
                pos = mic[:,:2]
                eulers = mic[:,3:] # already in degrees
                plotType2 = 2
        if plotType == 3:
            fig.add_trace(go.Scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2],marker=dict(color=eulers[:,eulerVal],showscale=True)))
        else:
            fig.add_trace(go.Scatter(x=pos[:,0],y=pos[:,1],mode='markers',marker=dict(color=eulers[:,eulerVal],showscale=True)))
        fig.update_traces(mode='markers')
        titleText = f'Microstructure coloring by Euler{eulerVal}'
    else:
        allFiles = glob.glob('Grains.csv.*')
        allFiles = [f for f in allFiles if not 'zip' in f]
        all_df = []
        for fileN in allFiles:
            df = pd.read_csv(fileN,header=8,delimiter='\t',index_col=False)
            loadNr = int(fileN.split('.')[-1])
            df.insert(0,"LoadNr",np.ones(df.shape[0]).astype(int)*loadNr,True)
            all_df.append(df)
        all_df = pd.concat(all_df)
        seqMax = len(allFiles)+1
        # print(all_df.shape)
        selLoadNr = int(selLoadNr)
        subDF = all_df.loc[all_df['LoadNr']==selLoadNr]
        x = subDF['X']
        y = subDF['Y']
        z = subDF['Z']
        color = subDF['eKen33']
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,marker=dict(color=color,colorscale='Jet',showscale=True)))
        fig.update_traces(mode='markers')
        titleText = f'Microstructure SequenceNr {selLoadNr} coloring by Strain (loading direction)'
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),width=700,height=700,
        title_text=titleText)
    return fig,seqMax

@callback(
    Output(component_id='microstructure2', component_property='figure'),
    Input('EulerComponent','value'),
    Input('Multiple','value'),
    Input('SequenceNr','value'),
    Input("button_run_2d","n_clicks"),
    Input("BeamSize", "value"),
)
def show_mic2(eulerVal,multipleData,selLoadNr,clicks,bsz):
    fig = go.Figure()
    if clicks is not None:
        global pos2, eulers2, plotType2, BC_or
        mic_h = open(micfn,'r').readline()
        seqMax = 0
        if not multipleData:
            if pos2 is None or BC_or != bsz:
                BC_or = bsz
                if mic_h.startswith('%NumGrains'):
                    # Grains.csv
                    mic = np.genfromtxt(micfn,skip_header=9)
                    rads = np.linalg.norm(mic[:,10:12],axis=1)
                    mic = mic[rads<bsz,:]
                    pos2 = mic[:,10:13]
                    orients = mic[:,1:10]
                    eulers2 = np.zeros((mic.shape[0],3))
                    for rowNr in range(orients.shape[0]):
                        orient = orients[rowNr]
                        eulers2[rowNr] = rad2deg * OrientMat2Euler(orient)
                    plotType2 = 3
                elif mic_h.startswith('%TriEdgeSize'):
                    # NF Mic, 2D
                    mic = np.genfromtxt(micfn,skip_header=4)
                    rads = np.linalg.norm(mic[:,3:5],axis=1)
                    mic = mic[rads<bsz,:]
                    pos2 = mic[:,3:6]
                    pos2[:,2] = 0
                    eulers2 = rad2deg * mic[:,7:10]
                    plotType2 = 2
                elif mic_h.startswith('# SpotID'):
                    # pf-Mic, 2D
                    mic = np.genfromtxt(micfn,skip_header=1,delimiter=',')
                    rads = np.linalg.norm(mic[:,11:13],axis=1)
                    mic = mic[rads<bsz,:]
                    pos2 = mic[:,11:14]
                    eulers2 = mic[:,-7:-4]
                    plotType2 = 2
                elif mic_h.startswith('#EBSD'):
                    # This is an EBSD CSV file written by Nicolas
                    # Which column has what....
                    # #EBSD X, Y, Z, Euler Angle1, Euler Angle2, Euler Angle3
                    mic = np.genfromtxt(micfn,skip_header=1,delimiter=',')
                    rads = np.linalg.norm(mic[:,0:2],axis=1)
                    mic = mic[rads<bsz,:]
                    pos2 = mic[:,:2]
                    eulers2 = mic[:,3:] # already in degrees
                    plotType2 = 2
            if plotType2 == 3:
                fig.add_trace(go.Scatter3d(x=pos2[:,0],y=pos2[:,1],z=pos2[:,2],marker=dict(color=eulers2[:,eulerVal],showscale=True)))
            else:
                fig.add_trace(go.Scatter(x=pos2[:,0],y=pos2[:,1],mode='markers',marker=dict(color=eulers2[:,eulerVal],showscale=True)))
            fig.update_traces(mode='markers')
            titleText = f'Microstructure coloring by Euler{eulerVal}'
        else:
            allFiles = glob.glob('Grains.csv.*')
            allFiles = [f for f in allFiles if not 'zip' in f]
            all_df = []
            for fileN in allFiles:
                df = pd.read_csv(fileN,header=8,delimiter='\t',index_col=False)
                loadNr = int(fileN.split('.')[-1])
                df.insert(0,"LoadNr",np.ones(df.shape[0]).astype(int)*loadNr,True)
                all_df.append(df)
            all_df = pd.concat(all_df)
            seqMax = len(allFiles)+1
            # print(all_df.shape)
            selLoadNr = int(selLoadNr)
            subDF = all_df.loc[all_df['LoadNr']==selLoadNr]
            x = subDF['X']
            y = subDF['Y']
            z = subDF['Z']
            color = subDF['eKen33']
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,marker=dict(color=color,colorscale='Jet',showscale=True)))
            fig.update_traces(mode='markers')
            titleText = f'SequenceNr {selLoadNr} StrainColor (loading direction) Median: {np.median(color)}'
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),width=700,height=700,
            title_text=titleText)
    return fig

@callback(
    Output('detector', 'relayoutData'),
    Input('detector', 'relayoutData'))
def save_relayout_data(relayoutData):
    global ranges
    if relayoutData is not None and len(relayoutData) ==4:
        print(len(relayoutData))
        ranges = [relayoutData['xaxis.range[0]'],relayoutData['xaxis.range[1]'],relayoutData['yaxis.range[0]'],relayoutData['yaxis.range[1]'],]


app.run(debug=False,host='0.0.0.0')
