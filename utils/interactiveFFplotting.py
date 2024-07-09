#!/usr/bin/env python

from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import argparse
import sys
import zarr
from math import cos, sin

deg2rad = np.pi/180
rad2deg = 180/np.pi

id_first = 0
id_spot_first = 0
values = []
options = []
window = 10
windowFrame = 7
selectedID = 0
xRowNr = 10
yRowNr = 11
zRowNr = 12
sizeRowNr = 22
completenessRowNr = 23

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

def rotateAroundZ(v,ome):
    m = [[cos(ome),-sin(ome),0],[sin(ome),cos(ome),0],[0,0,1]]
    r0 = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2]
    r1 = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2]
    r2 = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2]
    return r0,r1,r2

def spot2gv(x,y,z,ome):
    v = np.array([x,y,z])
    vn = v/np.linalg.norm(v)
    g1r = -1 + vn[0]
    g2r = vn[1]
    g3r = vn[2]
    g = [g1r,g2r,g3r]
    omeRad = -ome*deg2rad
    return rotateAroundZ(g,omeRad)

data = {
    'omega': [],
    'y': [],
    'z': [],
	'g1':[],
	'g2':[],
	'g3':[],
    'ringNr': [],
    'ringNrInt': [],
    'strain': [],
    'ds': [],
    'grainID': [],
    'grainIDColor': [],
    'spotID':[],
    'detY':[],
    'detZ':[],
    'spotSize':[],
    }

data2 = {
    'x': [],
    'y': [],
    'z': [],
    'GrainSize': [],
    'Confidence': [],
    'ID':[],
    'IDColor':[],
    'Eulers':[],
    'Euler0':[],
    'Euler1':[],
    'Euler2':[],
    'Error':[],
    'StrainError':[],
    'Completeness':[]
    }

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MIDAS FF-HEDM Interactive Viewer"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('MIDAS FF-HEDM Interactive viewer', className="text-primary text-center fs-3")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(options=[{"label": x, "value": x} for x in ['ringNr', 'grainIDColor','strain','spotSize']],
                        value='grainIDColor',
                        inline=True,
                        id='radio-buttons-spots')
        ], width=6),
        dbc.Col([
            dbc.RadioItems(options=[{"label": x, "value": x} for x in ['ringNr', 'grainIDColor','strain','spotSize']],
                        value='grainIDColor',
                        inline=True,
                        id='radio-buttons-spots_polar')
        ], width=6),
    ]),
    dbc.Row([
         dbc.Col([
              html.Div(
                   ['SelectRingNrs'],
                   id='selectRingNr_spots'
              )
         ],width=2),
         dbc.Col([
            dcc.Checklist(
                id="checklist_spots",
                inline=True,
            ),
         ],width=4),
         dbc.Col([
              html.Div(
                   ['SelectRingNrs'],
                   id='selectRingNr_spots_polar'
              )
         ],width=2),
         dbc.Col([
            dcc.Checklist(
                id="checklist_spots_polar",
                inline=True,
            ),
         ],width=4),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='spots')
        ], width=6),
        dbc.Col([
            dcc.Graph(figure={}, id='spots_polar')
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
        dbc.RadioItems(options=[{"label": x, "value": x} for x in ['Confidence', 'GrainSize', 'IDColor', 'Error','Euler0','Euler1','Euler2','StrainError']],
                       value='IDColor',
                       inline=True,
                       id='radio-buttons-grains')
        ], width=6),
        dbc.Col([
        dbc.RadioItems(options=[{"label": x, "value": x} for x in ['ringNr', 'grainIDColor','strain','spotSize']],
                       value='strain',
                       inline=True,
                       id='radio-buttons-spots_filtered')
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='grains',clickData={'points':[{'customdata':id_first}]})
        ], width=6),
        dbc.Col([
            dcc.Graph(figure={}, id='filtered_spots',clickData={'points':[{'customdata':id_spot_first}]})
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='filtered_spots_2d',clickData={'points':[{'customdata':id_spot_first}]})
        ], width=6),
        dbc.Col([
            dcc.Graph(figure={}, id='image_data')
        ], width=6),
    ]),

], fluid=True)

@callback(Output('checklist_spots','value'),Input('selectRingNr_spots','n_clicks'))
def set_spots_ringNr_value(selectedRingNr):
    return values
@callback(Output('checklist_spots','options'),Input('checklist_spots','value'))
def set_spots_ringNr_options(vals):
    return options
@callback(Output('checklist_spots_polar','value'),Input('selectRingNr_spots_polar','n_clicks'))
def set_spots_ringNr_value(selectedRingNr):
    return values
@callback(Output('checklist_spots_polar','options'),Input('checklist_spots_polar','value'))
def set_spots_ringNr_options(vals):
    return options

# Add controls to build the interaction
@callback(
    Output(component_id='grains', component_property='figure'),
    Input(component_id='radio-buttons-grains', component_property='value')
)
def update_graph(col_chosen):
    fig = px.scatter_3d(df2,
                    x='x',
                    y='y',
                    z='z',
                    color=col_chosen,
                    size='GrainSize',
                    title='Grains in 3D',
                    color_continuous_scale='jet',
                    hover_name='ID',
                    )
    fig.update_traces(customdata=df2['ID'])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50),height=700)
    return fig

@callback(
    Output(component_id='filtered_spots',component_property='figure'),
    Input('grains','clickData'),
    Input(component_id='radio-buttons-spots_filtered', component_property='value')
)
def newFilteredSpots(clickData,col_chosen):
    if 'hovertext' not in clickData['points'][0]:
        fig = px.scatter()
    else:
        print(clickData)
        print("I'm in this loop")
        ID = df2[df2['ID']== clickData['points'][0]['hovertext'] ]['ID'].item()
        dff = df[df['grainID']==ID]
        global selectedID
        selectedID = ID
        meanStrain = np.mean(np.abs(dff['strain']))
        medianStrain = np.median(np.abs(dff['strain']))
        fig = px.scatter_3d(dff,
                        x='omega',
                        y='y',
                        z='z',
                        color=col_chosen,
                        size='ds',
                        title=f'FilteredSpots for ID:{int(ID)}, MeanStrainErr: {int(meanStrain)}, MedianStrainErr: {int(medianStrain)}',
                        color_continuous_scale='jet',
                        hover_name='spotID',
                        )
        fig.update_traces(customdata=dff['spotID'])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50),height=700)
    return fig

@callback(
    Output(component_id='filtered_spots_2d',component_property='figure'),
    Input('grains','clickData'),
    Input(component_id='radio-buttons-spots_filtered', component_property='value')
)
def newFilteredSpots2(clickData,col_chosen):
    if 'hovertext' not in clickData['points'][0]:
        fig = px.scatter()
    else:
        print(clickData)
        ID = df2[df2['ID']== clickData['points'][0]['hovertext'] ]['ID'].item()
        dff = df[df['grainID']==ID]
        meanStrain = np.mean(np.abs(dff['strain']))
        medianStrain = np.median(np.abs(dff['strain']))
        fig = px.scatter(dff,
                        x='y',
                        y='z',
                        color=col_chosen,
                        size='ds',
                        title=f'2D FilteredSpots for ID:{int(ID)}, MeanStrainErr: {int(meanStrain)}, MedianStrainErr: {int(medianStrain)}',
                        color_continuous_scale='jet',
                        hover_name='spotID',
                        )
        fig.update_traces(customdata=dff['spotID'])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50),height=700)
    return fig

@callback(
    Output(component_id='image_data',component_property='figure'),
    Input('filtered_spots_2d','clickData'),
)
def imageData(clickData):
    if 'hovertext' in clickData['points'][0]:
        spotID = clickData['points'][0]['hovertext']
        dff = df[df['spotID']==spotID]
        dff = dff[dff['grainID']==selectedID]
        # we need to find the properties of this ID
        detY = int(dff['detY'].item())
        detZ = int(dff['detZ'].item())
        omega = dff['omega'].item()
        frameNrMid = int((omega-omegaStart)/omegaStep)
        frameMin = frameNrMid - windowFrame
        if frameMin<0: frameMin = 0
        frameMax = frameNrMid+windowFrame+1
        if frameMax> nFrames: frameMax=nFrames
        yMin = detY - window
        if yMin < 0: yMin = 0
        yMax = detY + window + 1
        if yMax > nPxY: yMax = nPxY
        zMin = detZ - window
        if zMin < 0: zMin = 0
        zMax = detZ + window + 1
        if zMax > nPxZ: zMax = nPxZ
        # Let's do the transformations here:::::
        for transOpt in ImTransOpt:
            if transOpt == 1:
                yT = yMin
                yMin = nPxY - yMax
                yMax = nPxY - yT
            if transOpt == 2:
                zT = zMin
                zMin = nPxZ - zMax
                zMax = nPxZ - zT
            if transOpt ==3:
                T = [yMin,yMax]
                yMin = zMin
                yMax = zMax
                zMin = T[0]
                zMax = T[1]
        extracted_data = rawDataLink[frameMin:frameMax,zMin:zMax,yMin:yMax].astype(np.double)
        extracted_data -= darkMean[zMin:zMax,yMin:yMax]
        extracted_data[extracted_data<thresh] = 0
        extracted_data = extracted_data.astype(np.uint16)
        X,Y,Z = np.mgrid[frameMin:frameMax,zMin:zMax,yMin:yMax]
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=extracted_data.flatten(),
            isomin=1,
            isomax=np.max(extracted_data),
            opacity=0.1,
            surface_count=100,
            ))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=50),height=700,title=f'Selected Spot ID {spotID} in 3D')
        return fig
    else:
        return {}

@callback(
    Output(component_id='spots', component_property='figure'),
    Input(component_id='radio-buttons-spots', component_property='value'),
    Input("checklist_spots", "value"),
)
def update_graph2(col_chosen2,ringNr):
    dff = df[df['ringNrInt'].isin(ringNr)]
    fig = px.scatter_3d(dff,
                    x='omega',
                    y='y',
                    z='z',
                    color=col_chosen2,
                    size='ds',
                    title='Spots in 3D',
                    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50),height=700)
    return fig

@callback(
    Output(component_id='spots_polar', component_property='figure'),
    Input(component_id='radio-buttons-spots_polar', component_property='value'),
    Input("checklist_spots_polar", "value"),
)
def update_graph3(col_chosen3,ringNr):
    dff = df[df['ringNrInt'].isin(ringNr)]
    fig = px.scatter_3d(dff,
                        x='g1',
                        y='g2',
                        z='g3',
                        color=col_chosen3,
                        size='ds',
                        title='G-vectors in 3D',
                        color_continuous_scale='jet',
                        )
    fig.update_layout(
        scene = dict(xaxis=dict(range=[-dsMax,dsMax]),yaxis=dict(range=[-dsMax,dsMax]),zaxis=dict(range=[-dsMax,dsMax])),
        margin=dict(l=0, r=0, b=0, t=50),hovermode='closest',height=700)
    return fig


df = []
df2 = []
dsMax = 0
# Run the app


parser = MyParser(description='''MIDAS FF Interactive Plotter''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where the reconstruction exists')
parser.add_argument('-dataFileName', type=str, required=True, help='Name of the input datafile')
parser.add_argument('-HostName', type=str, required=False, default="0.0.0.0", help='HostName IP')
parser.add_argument('-portNr', type=int, required=False, default=8050, help='HostName IP')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
dataFile = args.dataFileName
hn = args.HostName
portNr = args.portNr
zf = zarr.open(dataFile,'r')
pixSz = zf['/analysis/process/analysis_parameters/PixelSize'][0]
wl = zf['/analysis/process/analysis_parameters/Wavelength'][0]
Lsd = zf['/analysis/process/analysis_parameters/Lsd'][0]
thresh = zf['/analysis/process/analysis_parameters/RingThresh'][0][1]
omegaStep = zf['/measurement/process/scan_parameters/step'][0]
omegaStart = zf['/measurement/process/scan_parameters/start'][0]
rawDataLink = zf['exchange/data']
ImTransOpt = zf['/analysis/process/analysis_parameters/ImTransOpt'][:]
dark = zf['exchange/dark'][:]
darkMean = np.mean(dark,axis=0).astype(np.double)
nFrames,nPxY,nPxZ = rawDataLink.shape
spots = np.genfromtxt(resultDir+'/SpotMatrix.csv',skip_header=1)
spotsOrig = np.genfromtxt(resultDir+'/InputAll.csv',skip_header=1)
grains = np.genfromtxt(resultDir+'/Grains.csv',skip_header=9)
for i in range(spots.shape[0]):
    if spots[i][5]==0.0:
        continue
    data['omega'].append(spots[i][2]) # This is rotation direction
    data['y'].append(spots[i][8]/pixSz)
    data['z'].append(spots[i][9]/pixSz)
    data['ringNr'].append(str(int(spots[i][7])))
    data['ringNrInt'].append(int(spots[i][7]))
    data['grainID'].append(spots[i][0])
    data['spotID'].append(spots[i][1])
    data['spotSize'].append(spotsOrig[int(spots[i][1])-1,3])
    data['detY'].append(spots[i][3])
    data['detZ'].append(spots[i][4])
    data['strain'].append(1000000*np.abs(spots[i][11]))
    data['grainIDColor'].append(str(int(spots[i][0])))
    x,y,z = spot2gv(Lsd,spots[i][8],spots[i][9],spots[i][2])
    length = np.linalg.norm(np.array([x,y,z]))
    x = x/length
    y = y/length
    z = z/length
    ds = wl / (2*sin(spots[i][10]*deg2rad))
    x *=ds
    y *=ds
    z *=ds
    if ds > dsMax:
        dsMax = ds
    data['g1'].append(x)
    data['g2'].append(y)
    data['g3'].append(z)
    data['ds'].append(ds)
df = pd.DataFrame(data)

largestSize = np.max(grains[:,sizeRowNr])

for i in range(grains.shape[0]):
    data2['x'].append(grains[i][xRowNr])
    data2['y'].append(grains[i][yRowNr])
    data2['z'].append(grains[i][zRowNr])
    data2['GrainSize'].append(20*grains[i][sizeRowNr]/largestSize)
    data2['Confidence'].append(grains[i][completenessRowNr])
    data2['ID'].append(grains[i][0])
    data2['Euler0'].append(grains[i][-3])
    data2['Euler1'].append(grains[i][-2])
    data2['Euler2'].append(grains[i][-1])
    data2['StrainError'].append(grains[i][-5])
    data2['IDColor'].append(f'{int(grains[i][0])}')
    data2['Eulers'].append(f'Eul1: {grains[i][-3]}, Eul2: {grains[i][-2]}, Eul3: {grains[i][-1]}')
    data2['Error'].append(grains[i][19])
    data2['Completeness'].append(f'Completeness: {grains[i][completenessRowNr]}')

df2 = pd.DataFrame(data2)
df = df.sort_values(by=['grainID'])
df2 = df2.sort_values(by=['ID'])
id_first = df2['ID'][0]
values = df['ringNrInt'].unique().tolist()
values.sort()
options = [{"label":ringNr,"value":ringNr} for ringNr in values]

app.run_server(port=portNr,host=hn,debug=False)
