import numpy as np
import pandas as pd
import plotly.express as px
import argparse
import sys
from math import cos,sin,sqrt
import plotly.graph_objects as go


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
    'grainIDColor': [],
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
	data['grainID'].append(spots[i][0])
	data['grainIDColor'].append(str(int(spots[i][0])))

df = pd.DataFrame(data)

exclude_cols = ['grainIDColor']
labels = [col_name for col_name in df.columns if col_name not in exclude_cols]
data = [go.Scatter3d(x=df['omega'],y=df['y'],z=df['z'],visible=False,mode='markers',marker_color=df[c],marker_size=df['ringNr']) for c in labels]
data[0]['visible'] = True
fig = go.Figure(data=data,layout=go.Layout(title="Spots in 3D"))
visible = np.identity(len(labels), dtype=bool)
buttons= [dict(args=["visible", visible[i, :]], label=l, method="restyle")
            for (i, l) in enumerate(labels)]
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            showactive=True,
            x=0.05,
            xanchor="left",
            y=1.2,
            yanchor="top"
        ),
    ]
)

fig.update_layout(
    annotations=[
        dict(
            x=0.01,
            xref="paper",
            y=1.16,
            yref="paper",
            align="left",
            showarrow=False),
    ])

fig.write_html('spots3DgrainsSelector.html')

fig2 = px.scatter_3d(df,
                    x='omega',
                    y='y',
                    z='z',
                    color='grainIDColor',
                    size='ringNr',
                    title='Spots in 3D',
                    )

fig2.update_layout(margin=dict(l=0, r=0, b=0, t=50))

fig2.write_html('spots3Dgrains.html')
