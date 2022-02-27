import subprocess
import numpy as np
from argparse import RawTextHelpFormatter
import argparse
import warnings
import time
import os
import datetime
from pathlib import Path
import shutil
from math import acos,sqrt, atan

def CalcEtaAngle(y, z):
	alpha = 57.2957795130823*acos(z/sqrt(y*y+z*z))
	if (y>0):
		alpha = -alpha
	return alpha


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='''MIDAS_FF, contact hsharma@anl.gov Parameter file must be in the same folder as the desired output folder(SeedFolder)''', formatter_class=RawTextHelpFormatter)
parser.add_argument('-nCPUs',    type=int, required=True, help='Number of CPUs to use')
parser.add_argument('-paramFile', type=str, required=True, help='ParameterFileName')
parser.add_argument('-nNodes', type=str, required=True, help='Number of Nodes')
parser.add_argument('-machineName', type=str, required=True, help='Machine Name')
args, unparsed = parser.parse_known_args()
paramFN = args.paramFile
machineName = args.machineName
numProcs = args.nCPUs
nNodes = args.nNodes
os.environ["nNODES"] = str(nNodes)
os.environ["nCPUs"] = str(numProcs)

baseNameParamFN = paramFN.split('/')[-1]
homedir = os.path.expanduser('~')
paramContents = open(paramFN).readlines()
for line in paramContents:
	if line.startswith('StartFileNrFirstLayer'):
		startNrFirstLayer = int(line.split()[1])
	if line.startswith('NrFilesPerSweep'):
		nrFilesPerSweep = int(line.split()[1])
	if line.startswith('FileStem'):
		fStem = line.split()[1]
	if line.startswith('SeedFolder'):
		topdir = line.split()[1]
	if line.startswith('StartNr'):
		startNr = int(line.split()[1])
	if line.startswith('EndNr'):
		endNr = int(line.split()[1])
	if line.startswith('nScans'):
		nScans = int(line.split()[1])
	if line.startswith('Lsd'):
		Lsd = float(line.split()[1])

print(Lsd)
positions = open('positions.csv').readlines()

nFrames = endNr - startNr + 1
for layerNr in range(1,nScans+1):
	print(layerNr)
	ypos = float(positions[layerNr-1])
	thisStartNr = startNrFirstLayer + (layerNr-1)*nrFilesPerSweep
	folderName = str(thisStartNr)
	thisDir = topdir + '/' + folderName + '/'
	Path(thisDir).mkdir(parents=True,exist_ok=True)
	os.chdir(thisDir)
	thisParamFN = thisDir + baseNameParamFN
	thisPF = open(thisParamFN,'w')
	for line in paramContents:
		thisPF.write(line)
	thisPF.write('Folder '+thisDir+'\n')
	thisPF.write('LayerNr '+str(layerNr)+'\n')
	thisPF.write('StartFileNr '+str(thisStartNr)+'\n')
	thisPF.close()
	Path(thisDir+'/Temp').mkdir(parents=True,exist_ok=True)
	# ~ Path(thisDir+'Output').mkdir(parents=True,exist_ok=True)
	# ~ Path(thisDir+'Results').mkdir(parents=True,exist_ok=True)
	# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+" "+thisParamFN,shell=True)
	# TODO: call the PeaksFittingOMP code using swift to run PeakSearch on all the scans in parallel
	# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMP")+' '+baseNameParamFN+' 0 1 '+str(nFrames)+' '+str(numProcs),shell=True)
	# These need to be done sequentially
	# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAll")+' '+baseNameParamFN,shell=True)
	# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAll")+' '+baseNameParamFN,shell=True)
	# ~ subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetup")+' '+baseNameParamFN,shell=True)
	# Now do the position correction
	AllF = open('InputAllExtraInfoFittingAll.csv','r')
	allcontents = AllF.readlines()
	AllF.close()
	AllF = open(topdir+'/InputAllExtraInfoFittingAll'+str(layerNr-1)+'.csv','w')
	for line2 in allcontents:
		if line2[0] == '%':
			AllF.write(line2)
		else:
			line2sp = line2.split()
			y = float(line2sp[0])
			z = float(line2sp[1])
			ome = float(line2sp[2])
			grR = float(line2sp[3])
			ID = float(line2sp[4])
			RNr = float(line2sp[5])
			Eta = float(line2sp[6])
			Ttheta = float(line2sp[7])
			omeIniNoW = float(line2sp[8])
			yOrigNoW = float(line2sp[9])
			zOrigNoW = float(line2sp[10])
			yDet = float(line2sp[11])
			zDet = float(line2sp[12])
			omegaDet = float(line2sp[13])
			y = y - ypos
			if (y*y+z*z) < np.finfo(np.float32).eps:
				continue
			Eta = CalcEtaAngle(y,z)
			Ttheta = 57.2957795130823*atan(sqrt(y*y+z*z)/Lsd)
			yOrigNoW = yOrigNoW - ypos
			outstr = '{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n'.format(y,z,ome,grR,ID,RNr,Eta,Ttheta,omeIniNoW,yOrigNoW,zOrigNoW,yDet,zDet,omegaDet)
			AllF.write(outstr)
	AllF.close()

os.chdir(topdir)
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinDataScanning")+' '+str(nScans),shell=True)
# NEED TO MAKE PARAMSTEST.TXT, update folders, add BeamSize and px
# Parallel after this
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerScanningOMP")+' paramstest.txt 0 1 '+ str(nScans)+' '+str(numProcs))

print("Time Elapsed: "+str(time.time()-startTime)+" seconds.")
