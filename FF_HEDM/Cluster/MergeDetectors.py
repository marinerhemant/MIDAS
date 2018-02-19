#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

## Files to merge/update: InputAll.csv, InputAllExtraInfoFittingAll.csv, 
##						  paramstest.txt, SpotsToIndex.csv, IDsHash.csv
## New file to create: DetNrs.txt, MatchIDs.txt

######## TODO: WHAT TO DO IF ONE OF THE DETECTORS HAS NO SPOTS!!!!! ####

import numpy as np
import sys, os

totDataExtra = []
Lsds = []
startIDNrs = [[],[],[],[]]
endIDNrs = [[],[],[],[]]
totalIDs = []
RingNrs = []
DSpacings = []
RingRadii = []
OmegaRangeIndex = [-180,180]

### Get Lsds
paramfile = sys.argv[1]
f = open(paramfile,'r')
paramcontents = f.readlines()
f.close()
detParamsLines = []
for line in paramcontents:
	if line.startswith('DetParams '):
		Lsds.append(float(line.split()[1]))
		detParamsLines.append(line)
	elif line.startswith('RingThresh '):
		RingNrs.append(int(line.split()[1]))
	elif line.startswith('BigDetSize '):
		bigdetinfo = line
	elif line.startswith('MinOmeSpotIDsToIndex '):
		OmegaRangeIndex[0] = float(line.split()[1])
	elif line.startswith('MaxOmeSpotIDsToIndex '):
		OmegaRangeIndex[1] = float(line.split()[1])
	elif line.startswith('OverAllRingToIndex '):
		ringToIndex = int(line.split()[1])
	elif line.startswith('px '):
		px = line;
LsdMean = sum(Lsds)/len(Lsds)
nRings = len(RingNrs)

### Get DSpacings and RingRadii
f = open('hkls.csv','r')
f.readline()
hklinfo = f.readlines()
for ringnr in RingNrs:
	for line in hklinfo:
		if int(line.split()[4]) is ringnr:
			DSpacings.append(float(line.split()[3]))
			RingRadii.append(float(line.rstrip().split()[10]))
			break

### Update paramstest.txt
f = open('paramstest.txt','r')
paramcontents = f.readlines()
f.close()
cntr = 0
for (idx,line) in enumerate(paramcontents):
	if line.startswith('Distance '):
		paramcontents[idx] = 'Distance ' + str(LsdMean) + ';\n'
	elif line.startswith('OutputFolder '):
		paramcontents[idx] = 'OutputFolder ' + os.getcwd() + '/Output\n'
	elif line.startswith('ResultFolder '):
		paramcontents[idx] = 'ResultFolder ' + os.getcwd() + '/Results\n'
	elif line.startswith('RingRadii '):
		paramcontents[idx] = 'RingRadii ' + str(RingRadii[cntr]) + ';\n'
		cntr += 1
paramcontents.append('Mask BigDetectorMask.bin\n')
paramcontents.append(bigdetinfo)
paramcontents.append(px)
for line in detParamsLines:
	paramcontents.append(line)
f = open('paramstest.txt','w')
for line in paramcontents:
	f.write(line)

### Load extra info
totalNrSpots = 0
for detNr in range(4):
	f = open('IDsHash'+str(detNr+1)+'.csv')
	hashContents = f.readlines()
	f.close()
	for line in hashContents:
		startIDNrs[detNr].append(totalNrSpots+int(line.split()[1]))
		endIDNrs[detNr].append(totalNrSpots+int(line.split()[2]))
	f = open('InputAllExtraInfoFittingAll'+str(detNr+1)+'.csv','r')
	dataExtra = np.loadtxt(f,skiprows=1)
	f.seek(0,0)
	head = f.readline()
	f.close()
	if len(dataExtra) == 0:
		continue
	if len(dataExtra.shape) is 2:
		nSpots = dataExtra.shape[0]
	else:
		nSpots = 1
	totalIDs.append(nSpots)
	dataExtra[:,4] += totalNrSpots
	dataExtra[:,0] *= LsdMean/Lsds[detNr]
	dataExtra[:,1] *= LsdMean/Lsds[detNr]
	dataExtra[:,9] *= LsdMean/Lsds[detNr]
	dataExtra[:,10] *= LsdMean/Lsds[detNr]
	if len(totDataExtra) is 0:
		totDataExtra = np.copy(dataExtra)
	else:
		totDataExtra = np.concatenate((totDataExtra,dataExtra))
	totalNrSpots += nSpots

headinp = open('Detector1/InputAll.csv','r').readline()

startingID = 1
finput = open('InputAll.csv','a')
fextra = open('InputAllExtraInfoFittingAll.csv','a')
fSpotsToIndex = open('SpotsToIndex.csv','w')
fIDsHash = open('IDsHash.csv','a')
fIDsMap = open('IDsDetectorMap.csv','a')
finput.write(headinp)
fextra.write(head)
for (idx,ringNr) in enumerate(RingNrs):
	SpotsThisRing = []
	for detNr in range(4):
		thisStartNr = startIDNrs[detNr][idx] - 1 # To get index, rows start from 0
		thisEndNr = endIDNrs[detNr][idx]
		if len(SpotsThisRing) is 0:
			SpotsThisRing = totDataExtra[thisStartNr:thisEndNr,:]
		else:
			SpotsThisRing = np.concatenate((SpotsThisRing,totDataExtra[thisStartNr:thisEndNr,:]))
	nSpotsThisRing = SpotsThisRing.shape[0]
	SpotsThisRing = SpotsThisRing[SpotsThisRing[:,2].argsort()]
	IDHashThisRing = np.zeros(nSpotsThisRing)
	for detNr in range(4):
		thisStartNr = startIDNrs[detNr][idx]
		thisEndNr = endIDNrs[detNr][idx]
		IDHashThisRing[np.logical_and(SpotsThisRing[:,4] >= thisStartNr, 
									  SpotsThisRing[:,4] <= thisEndNr)] = detNr + 1
	SpotsThisRing[:,4] = np.arange(startingID,startingID+nSpotsThisRing,dtype=float)
	np.savetxt(finput,SpotsThisRing[:,:8],fmt='%12.5f',delimiter=' ',newline='\n')
	np.savetxt(fextra,SpotsThisRing,fmt='%12.5f',delimiter=' ',newline='\n')
	if ringNr is ringToIndex:
		IDsToIndex = SpotsThisRing[np.logical_and(SpotsThisRing[:,2] >= OmegaRangeIndex[0],
												  SpotsThisRing[:,2] <= OmegaRangeIndex[1]),4]
		np.savetxt(fSpotsToIndex,IDsToIndex,fmt='%d',newline='\n')
	fIDsHash.write(str(ringNr)+' '+ str(startingID)+ ' ' + 
		str(startingID+nSpotsThisRing-1) + ' ' + str(DSpacings[idx])+ '\n')
	np.savetxt(fIDsMap,IDHashThisRing,fmt='%d',newline='\n')
	startingID += nSpotsThisRing
