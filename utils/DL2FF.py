import numpy as np
from subprocess import check_call
from glob import glob
from os.path import expanduser
import os
from math import sqrt, degrees, acos, cos, sin
import argparse
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='DL-FF-HEDM reconstruction.')
parser.add_argument('-peakFile',     type=str, required=True, help='peakinfo.csv')
parser.add_argument('-nCPUs',    type=int, required=True, help='Number of CPUs to use')
parser.add_argument('-paramFile',     type=str, required=True, help='parameters.txt')
parser.add_argument('-COfolder', type=str, required=True, help='Folder with conventional peaksearch')
args, unparsed = parser.parse_known_args()

peaksFN = args.peakFile
pFile = args.paramFile
COfolder = args.COfolder
nCPUs = args.nCPUs
maxDiff = 3

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

# ~ def CalcEtaAngle(y,z):
	# ~ alpha = -degrees(acos(z/sqrt(y*y+z*z))) if y > 0 else degrees(acos(z/sqrt(y*y+z*z)))
	# ~ return alpha

thisFolder = os.getcwd() + '/'
paramFile = thisFolder + pFile
binFolder = expanduser('~')+ '/opt/MIDAS/FF_HEDM/bin/'
fileStem = getValueFromParamFile(paramFile,'FileStem')[0][0]
# ~ px = float(getValueFromParamFile(paramFile,'px')[0][0])
padding = int(getValueFromParamFile(paramFile,'Padding')[0][0])
# ~ BC = [float(s) for s in getValueFromParamFile(paramFile,'BC',nWords=2)[0]]
Rings = [int(s[0]) for s in getValueFromParamFile(paramFile,'RingThresh',nLines=100)]
# ~ omegaStep = float(getValueFromParamFile(paramFile,'OmegaStep')[0][0])
# ~ omegaStart = float(getValueFromParamFile(paramFile,'OmegaFirstFile')[0][0])
# ~ tx = float(getValueFromParamFile(paramFile,'tx')[0][0])
# ~ ty = float(getValueFromParamFile(paramFile,'ty')[0][0])
# ~ tz = float(getValueFromParamFile(paramFile,'tz')[0][0])
# ~ Lsd = float(getValueFromParamFile(paramFile,'Lsd')[0][0])
# ~ width=float(getValueFromParamFile(paramFile,'Width')[0][0])
# ~ RhoD=float(getValueFromParamFile(paramFile,'RhoD')[0][0])
# ~ p0=float(getValueFromParamFile(paramFile,'p0')[0][0])
# ~ p1=float(getValueFromParamFile(paramFile,'p1')[0][0])
# ~ p2=float(getValueFromParamFile(paramFile,'p2')[0][0])
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

# ~ deg2rad = 0.0174532925199433
# ~ rad2deg = 57.2957795130823
# ~ txr = deg2rad*tx
# ~ tyr = deg2rad*ty
# ~ tzr = deg2rad*tz
# ~ Rx = np.array([[1,0,0],[0,cos(txr),-sin(txr)],[0,sin(txr),cos(txr)]])
# ~ Ry = np.array([[cos(tyr),0,sin(tyr)],[0,1,0],[-sin(tyr),0,cos(tyr)]])
# ~ Rz = np.array([[cos(tzr),-sin(tzr),0],[sin(tzr),cos(tzr),0],[0,0,1]])
# ~ R = np.dot(Rx,np.dot(Ry,Rz))
# ~ n0=2
# ~ n1=4
# ~ n2=2

# find the layerNr for CO
COLayer = [s for s in COfolder.split('_') if 'Layer' in s][0].replace('Layer','')
COFStem = COfolder.split('Layer')[0] + COLayer + '_'

#### Will read CO folder peaks and only write matching peaks
peaksInfo = np.genfromtxt(peaksFN,delimiter=',',skip_header=1)
nPeaks,ncols = peaksInfo.shape
firstFrameNr = int(np.min(peaksInfo[:,1])) + 1
lastFrameNr = int(np.max(peaksInfo[:,1])) + 1
fNotFound = open('COPeaksNotFound.csv','w')
head = 'SpotID IntegratedIntensity Omega(degrees) YCen(px) ZCen(px) IMax Radius(px) Eta(degrees) SigmaR SigmaEta\n'
NrSpots = 0
for fNr in range(firstFrameNr,lastFrameNr+1):
	peaksThisFrame = peaksInfo[peaksInfo[:,1] == fNr-1]
	for ringNr in ringNrs:
		folN = thisFolder+folder+'Ring'+str(ringNr)+'/Temp/'
		check_call('mkdir -p '+folN,shell=True)
		fn = folN + fileStem + '_1_' + str(fNr).zfill(padding) + '_' + str(ringNr) + '_PS.csv'
		f = open(fn,'w')
		f.write(head)
		# Read COfolder analysis, find the same peaks and write out
		COFname = COfolder + '/Ring' + str(ringNr) +'/Temp/' + COFStem + str(fNr).zfill(padding) + '_' + str(ringNr) + '_PS.csv'
		COPeaks = np.genfromtxt(COFname,skip_header=1)
		nPeaks = COPeaks.size//19
		if nPeaks ==1:
			filtCO = COPeaks
			thisY = filtCO[3]
			thisZ = filtCO[4]
			peaksDiff = np.sqrt((peaksThisFrame[:,2] - thisY)**2+(peaksThisFrame[:,3] - thisZ)**2)
			filtPeaks = peaksThisFrame[peaksDiff < maxDiff]
			if filtPeaks.size == 0:
				writestr = str(fNr-1) + ' '
				for i in range (19):
					writestr += str(filtCO[i]) + ' '
				writestr += '\n'
				fNotFound.write(writestr)
				continue
			filtCO[3] = filtPeaks[0][2]
			filtCO[4] = filtPeaks[0][3]
			writestr = ''
			for i in range (19):
				writestr += str(filtCO[i]) + ' '
			writestr += '\n'
			f.write(writestr)
		else:
			for filtCO in COPeaks:
				thisY = filtCO[3]
				thisZ = filtCO[4]
				peaksDiff = np.sqrt((peaksThisFrame[:,2] - thisY)**2+(peaksThisFrame[:,3] - thisZ)**2)
				filtPeaks = peaksThisFrame[peaksDiff < maxDiff]
				if filtPeaks.size == 0:
					writestr = str(fNr-1) + ' '
					for i in range (19):
						writestr += str(filtCO[i]) + ' '
					writestr += '\n'
					fNotFound.write(writestr)
					continue
				filtCO[3] = filtPeaks[0][2]
				filtCO[4] = filtPeaks[0][3]
				writestr = ''
				for i in range (19):
					writestr += str(filtCO[i]) + ' '
				writestr += '\n'
				f.write(writestr)
		f.close()

# ~ for peakNr in range(nPeaks):
	# ~ Yc = (peaksInfo[peakNr,2] - BC[0])*px
	# ~ Zc = (peaksInfo[peakNr,3] - BC[1])*px
	# ~ ABC = np.array([0,Yc,Zc])
	# ~ ABCPr = np.dot(ABC,R)
	# ~ XYZ = ABCPr
	# ~ XYZ[0] += Lsd
	# ~ Rad = (Lsd/(XYZ[0]))*(sqrt(XYZ[1]*XYZ[1] + XYZ[2]*XYZ[2]))
	# ~ Eta = CalcEtaAngle(XYZ[1],XYZ[2])
	# ~ RNorm = Rad/RhoD
	# ~ EtaT = 90 - Eta
	# ~ DistortFunc = p0*(RNorm**n0)*cos(degrees(2*EtaT)) + p1*(RNorm**n1)*cos(degrees(4*EtaT)) + p2*(RNorm**n2) + 1
	# ~ thisRad = Rad * DistortFunc
	# ~ thisInt = peaksInfo[peakNr,4]
	# ~ thisR = thisRad / px
	# ~ thisEta = Eta
	# ~ thisFrame = int(peaksInfo[peakNr,1]) + 1
	# ~ thisOmega = omegaStart + (thisFrame-1)*omegaStep
	# ~ if thisFrame != 9:
		# ~ continue
	# ~ for ctr,ringNr in enumerate(ringNrs):
		# ~ tmpRingSz = ringSzs[ctr]
		# ~ if ringNr != 1:
			# ~ continue
		# ~ print(peaksInfo[peakNr,2],peaksInfo[peakNr,3],thisRad-tmpRingSz)
		# ~ if thisRad < tmpRingSz + width and thisRad > tmpRingSz - width:
			# ~ print(peaksInfo[peakNr])
			# ~ # now we write
			# ~ folN = thisFolder+folder+'Ring'+str(ringNr)+'/Temp/'
			# ~ fn = folN + fileStem + '_1_' + str(thisFrame).zfill(padding) + '_' + str(ringNr) + '_PS.csv'
			# ~ f = open(fn,'r')
			# ~ SpotID = len(f.readlines())
			# ~ f.close()
			# ~ outstr = str(SpotID) + ' '+ str(thisInt) + ' ' + str(thisOmega) + ' ' + str(peaksInfo[peakNr,2]) + ' ' + str(peaksInfo[peakNr,3]) + ' 100 ' + str(thisR) + ' ' + str(thisEta) + ' 1 1 1 1 1\n'
			# ~ f = open(fn,'a')
			# ~ f.write(outstr)
			# ~ f.close()
			# ~ NrSpots += 1

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
