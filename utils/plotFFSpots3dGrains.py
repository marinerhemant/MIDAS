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
    'omega': [],
    'y': [],
    'z': [],
    'ringNr': [],
    'grainID': [],
    }

parser = MyParser(description='''Plot Spots in 3D.py''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-resultFolder', type=str, required=True, help='Folder where the reconstruction exists')
args, unparsed = parser.parse_known_args()
resultDir = args.resultFolder
spots = np.genfromtxt(resultDir+'/SpotMatrix.csv',skip_header=1)
grains = np.genfromtxt(resultDir+'/Grains.csv',skip_header=9)

for i in range(spots.shape[0]):
	if spots[i][5]==0.0:
		continue
	data['omega'].append(spots[i][2]) # This is rotation direction
	data['y'].append(spots[i][8])
	data['z'].append(spots[i][9])
	data['ringNr'].append(spots[i][7])
	data['grainID'].append(str(int(spots[i][0])))

df = pd.DataFrame(data)

fig = px.scatter_3d(df,
                    x='omega',
                    y='y',
                    z='z',
                    color='grainID',
                    size='ringNr',
                    title='Spots in 3D',
                    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))

fig.write_html('spots3Dgrains.html')
