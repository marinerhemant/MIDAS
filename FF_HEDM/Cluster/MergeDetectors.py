#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

## Files to merge/update: InputAll.csv, InputAllExtraInfoFittingAll.csv, 
##						  paramstest.txt, SpotsToIndex.csv, IDsHash.csv
## New file to create: DetNrs.txt, MatchIDs.txt

import numpy as np
import sys

totDataAll = []
totDataExtra = []
Lsds = []
startIDNrs = [[],[],[],[]]
endIDNrs = [[],[],[],[]]
totalIDs = []
RingNrs = []
DSpacings = []

### Get Lsds
paramfile = sys.argv[1]
f = open(paramfile,'r')
paramcontents = f.readlines()
f.close()
for line in paramcontents:
	if line.startswith('DetParams '):
		Lsds.append(float(line.split()[1]))
LsdMean = sum(Lsds)/len(Lsds)

totalNrSpots = 0
for detNr in range(1,5):
	f = open('InputAllExtraInfoFittingAll'+str(detNr)+'.csv','r')
	dataExtra = np.loadtxt(f,skiprows=1)
	dataExtra[:,4] += totalNrSpots
	dataExtra[:,0] *= LsdMean/Lsds[detNr-1]
	dataExtra[:,1] *= LsdMean/Lsds[detNr-1]
	if len(dataExtra.shape) is 2:
		nSpots = dataExtra.shape[0]
	else:
		nSpots = 1
	totalIDs.append(nSpots)
	if len(totDataExtra) is 0:
		totDataExtra = np.copy(dataExtra)
	else:
		totDataExtra = np.concatenate(totDataExtra,dataExtra)
	f.close()
	f = open('IDsHash'+str(detNr)+'.csv')
	hashContents = f.readlines()
	f.close()
	nRings = len(hashContents)
	for line in hashContents:
		startIDNrs[detNr-1].append(int(line.split()[1]))
		endIDNrs[detNr-1].append(int(line.split()[2]))
		if detNr is 1:
			RingNrs.append(int(line.split()[0]))
			DSpacings.append(float(line.rstrip().split()[-1]))
	totalNrSpots += nSpots 

