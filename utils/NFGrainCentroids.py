import numpy as np
import os

nLayers = 7
nGrains = 226
micFN = 'MicrostructureZrTxt.mic'
outFN = 'GrainCentroidsWeighted.csv'
minConfidence = 0.5
ConfidenceThresh = 0.05
lowLimitConf = 0.10

outArr = np.zeros((nGrains,4))
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
				xPos2 += xCen2
				yPos2 += yCen2
				nLayersFound += nRows
	xPos /= nLayersFound
	yPos /= nLayersFound
	xPos2 /= nLayersFound
	yPos2 /= nLayersFound
	xErr = abs(xPos - xPos2)
	yErr = abs(yPos - yPos2)
	outArr[grainNr-1,0] = xPos
	outArr[grainNr-1,1] = yPos
	outArr[grainNr-1,2] = xErr
	outArr[grainNr-1,3] = yErr

np.savetxt(outFN,outArr)

import matplotlib.pyplot as plt
plt.errorbar(outArr[:,0],outArr[:,1],yerr=outArr[:,3],xerr=outArr[:,2],linewidth=0,markersize=10,marker='o')
plt.show()
