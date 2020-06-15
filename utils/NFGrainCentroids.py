import numpy as np
import os
from math import radians, cos, sin

nLayers = 7
# ~ nGrains = 226
# ~ nLayers = 5
nGrains = 382
micFN = 'MicrostructureZrTxt.mic'
outFN = 'GrainCentroidsWeighted.csv'
minConfidence = 0.5
ConfidenceThresh = 0.05
lowLimitConf = 0.10
rotAngle = -4

outArr = np.zeros((nGrains,4))
grainCtr = 0
for grainNr in range(1,nGrains+1):
	xPos = 0
	yPos = 0
	xPos2 = 0
	yPos2 = 0
	nLayersFound = 0
	for layerNr in range(nLayers):
		subDir = 'Layer_' + str(layerNr) + '_grainNr_' + str(grainNr) + '/'
		thisFN = subDir+micFN
		if os.path.exists(thisFN):
			print("Reading file: " + thisFN)
			thisMic = np.genfromtxt(thisFN,skip_header=4)
			thisMic = thisMic[thisMic[:,10] > minConfidence,:]
			nRows,_f = thisMic.shape
			if nRows == 0:
				continue
			maxConfidence = np.max(thisMic[:,10])
			if maxConfidence - lowLimitConf > minConfidence:
				filteredMic = thisMic[thisMic[:,10] > maxConfidence - ConfidenceThresh,:]
				nRows,_f = filteredMic.shape
				xCen = np.mean(filteredMic[:,3])
				yCen = np.mean(filteredMic[:,4])
				filteredMic2 = thisMic[thisMic[:,10] > maxConfidence - lowLimitConf,:]
				xCen2 = np.mean(filteredMic2[:,3])
				yCen2 = np.mean(filteredMic2[:,4])
				xPos += xCen * nRows
				yPos += yCen * nRows
				xPos2 += xCen2 * nRows
				yPos2 += yCen2 * nRows
				nLayersFound += nRows
	if nLayersFound > 0:
		xPos /= nLayersFound
		yPos /= nLayersFound
		xPos2 /= nLayersFound
		yPos2 /= nLayersFound
		xErr = abs(xPos - xPos2)
		yErr = abs(yPos - yPos2)
		rotXPos = xPos*cos(radians(rotAngle)) - yPos*sin(radians(rotAngle))
		rotYPos = xPos*sin(radians(rotAngle)) + yPos*cos(radians(rotAngle))
		outArr[grainCtr,0] = rotXPos
		outArr[grainCtr,1] = rotYPos
		outArr[grainCtr,2] = xErr
		outArr[grainCtr,3] = yErr
		grainCtr += 1

np.savetxt(outFN,outArr)

import matplotlib.pyplot as plt
plt.errorbar(outArr[:,0],outArr[:,1],yerr=outArr[:,3],xerr=outArr[:,2],linewidth=0,markersize=10,marker='o')
plt.show()

import numpy as np
outArr = np.genfromtxt('GrainCentroidsWeighted.csv')
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,1); ax.errorbar(outArr[:,0],outArr[:,1],yerr=outArr[:,3],xerr=outArr[:,2],linewidth=0,markersize=10,marker='o'); ax.set_aspect('equal'); plt.show()
