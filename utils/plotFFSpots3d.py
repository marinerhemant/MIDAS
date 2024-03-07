import numpy as np
import pandas as pd
import plotly.express as px
import argparse
import sys
from math import cos,sin,tan

def spheric2cartesian(r, theta, phi):
    x = r*cos(theta) *sin(phi)
    y = r*sin(theta)*sin(phi)
    z= r*cos(phi)
    return x, y, z

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

data = {
    'omega': [],
    'y': [],
    'z': [],
	'x_pol':[],
	'y_pol':[],
	'z_pol':[],
    'size': [],
    'rad': [],
    'ringNr': [],
    }

parser = MyParser(description='''Plot Spots in 3D.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where the reconstruction exists')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
spots = np.genfromtxt(resultDir+'/InputAll.csv',skip_header=1)

largestSize = np.max(spots[:,3])

for i in range(spots.shape[0]):
	if spots[i][5]==0.0:
		continue
	x,y,z = spheric2cartesian(1000000*tan(spots[i][7]*np.pi/180)/200,spots[i][6],spots[i][2])
	data['x_pol'].append(x) # This is rotation direction
	data['y_pol'].append(y)
	data['z_pol'].append(z)
	data['omega'].append(spots[i][2]) # This is rotation direction
	data['y'].append(spots[i][0])
	data['z'].append(spots[i][1])
	data['size'].append(20*spots[i][3]/largestSize)
	data['rad'].append(tan(spots[i][7]*np.pi/180))
	data['ringNr'].append(str(int(spots[i][5])))

df = pd.DataFrame(data)

fig = px.scatter_3d(df,
                    x='omega',
                    y='y',
                    z='z',
                    color='ringNr',
                    size='size',
                    title='Spots in 3D',
                    color_continuous_scale='jet',
                    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))

fig.write_html('spots3D.html')
fig2 = px.scatter_3d(df,
                    x='x_pol',
                    y='y_pol',
                    z='z_pol',
                    color='ringNr',
                    size='rad',
                    title='Spots in 3D',
                    color_continuous_scale='jet',
                    )

fig2.update_layout(margin=dict(l=0, r=0, b=0, t=50))

fig2.write_html('spots3D_polar.html')
