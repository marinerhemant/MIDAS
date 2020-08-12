import numpy as np
from subprocess import check_call
from glob import glob
from os.path import expanduser
import os
from math import sqrt, degrees, acos
import argparse

parser = argparse.ArgumentParser(description='HEDM reconstruction.')
parser.add_argument('-peak',   type=str, required=True, help='peak info csv')
parser.add_argument('-para',   type=str, required=True, help='parameters txt')
args, unparsed = parser.parse_known_args()

peaksFN = args.peak
pFile = args.para
nCPUs = 2

###### pFile must not have seedFolder and FolderName arguments, also the filename must not have path appended to it.

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

def CalcEtaAngle(y,z):
	alpha = -degrees(acos(z/sqrt(y*y+z*z))) if y > 0 else degrees(acos(z/sqrt(y*y+z*z)))
	return alpha

thisFolder = os.getcwd() + '/'
paramFile = thisFolder + pFile
binFolder = expanduser('~')+ '/opt/MIDAS/FF_HEDM/bin/'
fileStem = getValueFromParamFile(paramFile,'FileStem')[0][0]
px = float(getValueFromParamFile(paramFile,'px')[0][0])
padding = int(getValueFromParamFile(paramFile,'Padding')[0][0])
BC = [float(s) for s in getValueFromParamFile(paramFile,'BC',nWords=2)[0]]
Rings = [int(s[0]) for s in getValueFromParamFile(paramFile,'RingThresh',nLines=100)]
omegaStep = float(getValueFromParamFile(paramFile,'OmegaStep')[0][0])
omegaStart = float(getValueFromParamFile(paramFile,'OmegaFirstFile')[0][0])

width=float(getValueFromParamFile(paramFile,'Width')[0][0])
folder = fileStem + '_Layer1_Analysis_Time_2020_06_11_12_55_10/'
check_call('mkdir -p '+thisFolder+folder,shell=True)
check_call(binFolder+'/GetHKLList ' + paramFile,shell=True)
hkls = np.genfromtxt('hkls.csv',skip_header=1)
check_call('cp '+thisFolder+'hkls.csv '+thisFolder+folder,shell=True)
# ~ ringNrs=(set(list(hkls[:,4].flat)))
# ~ ringNrs = [int(s) for s in ringNrs]
ringNrs = Rings
ringSzs = np.zeros(len(ringNrs))
for ctr,ringNr in enumerate(ringNrs):
	for row in hkls:
		if int(row[4]) == ringNr:
			ringSzs[ctr] = float(row[10])

peaksInfo = np.genfromtxt(peaksFN,delimiter=',',skip_header=1)
nPeaks,ncols = peaksInfo.shape

firstFrameNr = int(np.min(peaksInfo[:,1])) + 1
lastFrameNr = int(np.max(peaksInfo[:,1])) + 1
head = 'SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax Radius(px) Eta(degrees) SigmaR SigmaEta\n'
for ringNr in ringNrs:
	folN = thisFolder+folder+'Ring'+str(ringNr)+'/Temp/'
	check_call('mkdir -p '+folN,shell=True)
	for fNr in range(firstFrameNr,lastFrameNr+1):
		fn = folN + fileStem + '_1_' + str(fNr).zfill(padding) + '_' + str(ringNr) + '_PS.csv'
		f = open(fn,'w')
		f.write(head)
		f.close()

NrSpots = 0
for peakNr in range(nPeaks):
	thisY = peaksInfo[peakNr,2] - BC[0]
	thisZ = peaksInfo[peakNr,3] - BC[1]
	thisInt = peaksInfo[peakNr,4]
	thisRad = sqrt(thisY**2+thisZ**2) * px
	thisR = thisRad / px
	thisEta = CalcEtaAngle(thisY,thisZ)
	thisFrame = int(peaksInfo[peakNr,1]) + 1
	thisOmega = omegaStart + (thisFrame-1)*omegaStep
	for ctr,ringNr in enumerate(ringNrs):
		tmpRingSz = ringSzs[ctr]
		if thisRad < tmpRingSz + width and thisRad > tmpRingSz - width:
			# now we write
			folN = thisFolder+folder+'Ring'+str(ringNr)+'/Temp/'
			fn = folN + fileStem + '_1_' + str(thisFrame).zfill(padding) + '_' + str(ringNr) + '_PS.csv'
			f = open(fn,'r')
			SpotID = len(f.readlines()) - 1
			f.close()
			outstr = str(SpotID) + ' '+ str(thisInt) + ' ' + str(thisOmega) + ' ' + str(thisY+BC[0]) + ' ' + str(thisZ+BC[1]) + ' 100 ' + str(thisR) + ' ' + str(thisEta) + ' 1 1 1 1 1\n'
			f = open(fn,'a')
			f.write(outstr)
			f.close()
			NrSpots += 1

# Now run MergeOverlappingPeaks, then CalcRadius, then run a new FF analysis without peaksearch. A bit convoluted, but easiest way for now
# First MergeOverlappingPeaks
os.chdir(thisFolder+folder)
for ringNr in ringNrs:
	os.chdir(thisFolder+folder)
	folN = thisFolder+folder+'Ring'+str(ringNr)
	thisParamFile = thisFolder + folder + 'Layer1_' + pFile
	check_call('cp '+paramFile+' '+thisParamFile,shell=True)
	pf = open(thisParamFile,'a')
	pf.write('Folder '+folN+'\n')
	pf.write('RingToIndex '+str(ringNr)+'\n')
	pf.write('RingNumbers '+str(ringNr)+'\n')
	pf.write('LayerNr 1\n')
	pf.close()
	check_call(binFolder+'MergeOverlappingPeaks '+thisParamFile+' '+str(ringNr),shell=True)
	check_call(binFolder+'CalcRadius '+thisParamFile+' '+str(ringNr),shell=True)
	resFile=folN + '/PeakSearch/' + fileStem + '_1/Radius_StartNr_' + str(firstFrameNr) + '_EndNr_' + str(lastFrameNr) + '_RingNr_' + str(ringNr) + '.csv'
	check_call('cp ' + resFile + ' ' + thisFolder + folder,shell=True)

os.chdir(thisFolder)
newParamFile = paramFile + '.New.txt'
check_call('cp '+paramFile+' '+newParamFile,shell=True)
f = open(newParamFile,'a')
f.write('FolderName '+folder[:-1]+'\n')
f.write('SeedFolder '+thisFolder+'\n')
f.close()

check_call(expanduser('~')+'/.MIDAS/MIDAS_V5_FarField_Layers '+pFile+'.New.txt 1 1 0 '+ str(nCPUs) + ' local sd',shell=True)
