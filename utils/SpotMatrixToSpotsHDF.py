import os
import sys
import numpy as np
import h5py

# What we want to write:
# Software Information, contact info
# Number of Grains
# For each grain
#		|- All the 40ish properties
#		|- Spots Info
#				|- All the spotMatrix Info
#				|- All the info from Radius...csv file
# All the files: Parameters.txt, InputAll.csv, InputExtra, SpotsToIndex, hkls.csv, IDsHash.csv, IDRings.csv, paramstest.csv, SpotMatrix.csv, Grains.csv

def getValueFromParamFile(paramfn,searchStr,nLines=1,wordNr=1,nWords=1):
	ret_list = []
	nrLines = 0
	f = open(paramfn,'r')
	PSContents = f.readlines()
	for line in PSContents:
		if line.startswith(searchStr+' '):
			line = line.replace('\t',' ')
			line = line.replace('\n',' ')
			words = line.split(' ')
			words = [_f for _f in words if _f]
			ret_list.append(words[wordNr:wordNr+nWords])
			nrLines += 1
			if (nrLines == nLines):
				return ret_list
	return ret_list

paramFile = sys.argv[1]
outFN = sys.argv[2]
ringNrs = [s[0] for s in getValueFromParamFile(paramFile,'RingThresh',100)]
startNr = int(getValueFromParamFile(paramFile,'StartNr')[0][0])
endNr = int(getValueFromParamFile(paramFile,'EndNr')[0][0])
fStem = getValueFromParamFile(paramFile,'FileStem')[0][0]
pad = int(getValueFromParamFile(paramFile,'Padding')[0][0])
layerNr = os.path.basename(os.getcwd()).split('Layer')[1].split('_')[0]
Grains = np.genfromtxt('Grains.csv',skip_header=9)
SpotMatrix = np.genfromtxt('SpotMatrix.csv',skip_header=1)
IDRings = np.genfromtxt('IDRings.csv',skip_header=1)
IDsHash = np.genfromtxt('IDsHash.csv',skip_header=1)
InputAll = np.genfromtxt('InputAll.csv',skip_header=1)
InputAllExtra = np.genfromtxt('InputAllExtraInfoFittingAll.csv',skip_header=1)
SpotsToIndex = np.genfromtxt('SpotsToIndex.csv',skip_header=1)
HKLs = np.genfromtxt('hkls.csv',skip_header=1)
outFile = h5py.File(outFN,'w')

headSpots = 'GrainID SpotID Omega DetectorHor DetectorVert OmeRaw Eta RingNr YLab ZLab Theta StrainError OriginalRadiusFileSpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax MinOme(degrees) MaxOme(degress) Radius(px) Theta(degrees) Eta(degrees) DeltaOmega NImgs RingNr GrainVolume GrainRadius PowderIntensity SigmaR SigmaEta'

f = open('Grains.csv','r')
nGrains = int(f.readline().split()[1])
beamCenter = float(f.readline().split()[1])
beamThickness = float(f.readline().split()[1])
globalPosition = float(f.readline().split()[1])
f.readline()
f.readline()
f.readline()
f.readline()
hGr = f.readline()
f.close()

outFile.attrs['Software'] = np.string_("MIDAS")
outFile.attrs['Version'] = np.string_("5.0")
outFile.attrs['Contact'] = np.string_("hsharma@anl.gov")
outFile.create_dataset('ParametersFile',data=np.string_(open(paramFile).read()))
group1 = outFile.create_group('RawFiles')
group1.create_dataset('paramstest',data=np.string_(open('paramstest.txt').read()))
sm = group1.create_dataset('SpotMatrix',data=SpotMatrix)
sm.attrs['head'] = np.string_(open('SpotMatrix.csv').readline())
gr = group1.create_dataset('Grains',data=Grains)
gr.attrs['head'] = np.string_(hGr)
idr = group1.create_dataset('IDRings',data=IDRings)
idr.attrs['head'] = np.string_(open('IDRings.csv').readline())
idh = group1.create_dataset('IDsHash',data=IDsHash)
idh.attrs['head'] = np.string_(open('IDsHash.csv').readline())
ipa = group1.create_dataset('InputAll',data=InputAll)
ipa.attrs['head'] = np.string_(open('InputAll.csv').readline())
ipe = group1.create_dataset('InputAllExtraInfo',data=InputAllExtra)
ipe.attrs['head'] = np.string_(open('InputAllExtraInfoFittingAll.csv').readline())
group1.create_dataset('SpotsToIndex',data=SpotsToIndex)
hk = group1.create_dataset('HKLs',data=HKLs)
hk.attrs['head'] = np.string_(open('hkls.csv').readline())

radii = []
for ring in ringNrs:
	group2 = group1.create_group('Ring'+ring)
	group3 = group2.create_group('Temp')
	tempFileStem = os.getcwd() + '/Ring' + ring + '/Temp/'
	for fNr in range(startNr, endNr+1):
		fileName = tempFileStem + fStem + '_' + layerNr + '_' + str(fNr).zfill(pad) + '_' + ring +'_PS.csv'
		if os.path.exists(fileName):
			arr = np.genfromtxt(fileName,skip_header=1)
			if arr.shape[0] > 0:
				tmpd = group3.create_dataset(os.path.basename(fileName),data=arr)
				tmpd.attrs['head'] = np.string_(open(fileName).readline())
	fileName = os.getcwd() + '/Radius_StartNr_' + str(startNr) + '_EndNr_' + str(endNr) + '_RingNr_' + ring + '.csv'
	arr = np.genfromtxt(fileName,skip_header=1)
	radd = group2.create_dataset(os.path.basename(fileName),data=arr)
	radd.attrs['head'] = np.string_(open(fileName).readline())
	radii.append(arr)

ringNrs = [int(r) for r in ringNrs]

for grain in Grains:
	thisID = int(grain[0])
	spotsThisGrain = SpotMatrix[SpotMatrix[:,0] == thisID]
	RadiusInfo = np.empty((spotsThisGrain.shape[0],19))
	for ctr,spot in enumerate(spotsThisGrain):
		spotID = int(spot[1])
		orig_ID = int(IDRings[IDRings[:,2]==spotID,1])
		ringNr = int(IDRings[IDRings[:,2]==spotID,0])
		pos = ringNrs.index(ringNr)
		subInfo = radii[pos][orig_ID-1]
		RadiusInfo[ctr,:] = subInfo
	RadiusInfo = np.hstack((spotsThisGrain,RadiusInfo))
	spd = outFile.create_dataset('GrainID'+str(thisID)+'SpotMatrix_Radius',data=RadiusInfo)
	spd.attrs['header'] = headSpots
