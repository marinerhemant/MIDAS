import numpy as np
import pandas as pd
import plotly.express as px
import argparse
import sys
from math import cos,sin,sqrt

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

data = {
    'x': [],
    'y': [],
    'z': [],
    'GrainSize': [],
    'Confidence': [],
    'ID':[],
    'IDColor':[],
    'Eulers':[],
    'Error':[],
    'Completeness':[]
    }

parser = MyParser(description='''Plot grains in 3D.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where the reconstruction exists')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
grains = np.genfromtxt(resultDir+'/Grains.csv',skip_header=9)
xRowNr = 10
yRowNr = 11
zRowNr = 12
sizeRowNr = 22
completenessRowNr = 23

largestSize = np.max(grains[:,sizeRowNr])

for i in range(grains.shape[0]):
	data['x'].append(grains[i][xRowNr]) # This is rotation direction
	data['y'].append(grains[i][yRowNr])
	data['z'].append(grains[i][zRowNr])
	data['GrainSize'].append(20*grains[i][sizeRowNr]/largestSize)
	data['Confidence'].append(grains[i][completenessRowNr])
	data['ID'].append(grains[i][0])
	data['IDColor'].append(f'{int(grains[i][0])}')
	data['Eulers'].append(f'Eul1: {grains[i][-3]}, Eul2: {grains[i][-2]}, Eul3: {grains[i][-1]}')
	data['Error'].append(f'Error: {grains[i][19]}')
	data['Completeness'].append(f'Completeness: {grains[i][completenessRowNr]}')

df = pd.DataFrame(data)
df = df.sort_values(by=['ID'])

fig = px.scatter_3d(df,
                    x='x',
                    y='y',
                    z='z',
                    color='Confidence',
                    size='GrainSize',
                    title='Grains in 3D',
                    color_continuous_scale='jet',
                    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))

fig.write_html('grains3DConfidence.html')

fig2 = px.scatter_3d(df,
                    x='x',
                    y='y',
                    z='z',
                    color='IDColor',
                    size='GrainSize',
                    title='Grains in 3D',
                    color_continuous_scale='jet',
                    )

fig2.update_layout(margin=dict(l=0, r=0, b=0, t=50))

fig2.write_html('grains3DID.html')
