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
from calcMiso import *
import glob

def CalcEtaAngle(y, z):
	alpha = 57.2957795130823*acos(z/sqrt(y*y+z*z))
	if (y>0):
		alpha = -alpha
	return alpha

startTime = time.time()

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='''MIDAS_FF, contact hsharma@anl.gov Parameter file must be in the same folder as the desired output folder(SeedFolder).\nMUST BE RUN FROM MIDAS DIRECTORY.''', formatter_class=RawTextHelpFormatter)
parser.add_argument('-nCPUs',    type=int, required=True, help='Number of CPUs to use')
parser.add_argument('-paramFile', type=str, required=True, help='ParameterFileName: Use the full path.')
parser.add_argument('-nNodes', type=str, required=True, help='Number of Nodes')
parser.add_argument('-machineName', type=str, required=True, help='Machine Name')
parser.add_argument('-doPeakSearch',type=int,required=True,help='0 if PeakSearch is already done. InputAllExtra...0..n.csv should exist in the folder')
parser.add_argument('-oneSolPerVox',type=int,required=True,help='0 if want to allow multiple solutions per voxel. 1 if want to have only 1 solution per voxel.')
args, unparsed = parser.parse_known_args()
paramFN = args.paramFile
machineName = args.machineName
doPeakSearch = args.doPeakSearch
oneSolPerVox = args.oneSolPerVox
numProcs = args.nCPUs
nNodes = args.nNodes
os.environ["nNODES"] = str(nNodes)
os.environ["nCPUs"] = str(numProcs)

maxAng = 0.5

baseNameParamFN = paramFN.split('/')[-1]
homedir = os.path.expanduser('~')
paramContents = open(paramFN).readlines()
RingNrs = []
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
	if line.startswith('OverAllRingToIndex'):
		RingToIndex = int(line.split()[1])
	if line.startswith('BeamSize'):
		BeamSize = float(line.split()[1])
	if line.startswith('px'):
		px = float(line.split()[1])
	if line.startswith('RingThresh'):
		RingNrs.append(int(line.split()[1]))

hkls = np.genfromtxt('hkls.csv',skip_header=1)
_rnr,idx = np.unique(hkls[:,4],return_index=True)
hkls = hkls[idx,:]
rads = []
for rnr in RingNrs:
	for hkl in hkls:
		if hkl[4] == rnr:
			rads.append(hkl[-1])

rads = [hkl[-1] for rnr in RingNrs for hkl in hkls if hkl[4] == rnr]

positions = open(topdir+'/positions.csv').readlines()

nFrames = endNr - startNr + 1
if doPeakSearch == 1:
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
		Path(thisDir+'Output').mkdir(parents=True,exist_ok=True)
		Path(thisDir+'Results').mkdir(parents=True,exist_ok=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+" "+thisParamFN,shell=True)
		# TODO: call the PeaksFittingOMP code using swift to run PeakSearch on all the scans in parallel
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/PeaksFittingOMP")+' '+baseNameParamFN+' 0 1 '+str(nFrames)+' '+str(numProcs),shell=True)
		# These need to be done sequentially
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetup")+' '+baseNameParamFN,shell=True)
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
		shutil.copy2(thisDir+'/paramstest.txt',topdir+'/paramstest.txt')
		shutil.copy2(thisDir+'/hkls.csv',topdir+'/hkls.csv')

os.chdir(topdir)
Path(topdir+'Output').mkdir(parents=True,exist_ok=True)
Path(topdir+'Results').mkdir(parents=True,exist_ok=True)
paramsf = open('paramstest.txt','r')
lines = paramsf.readlines()
paramsf.close()
paramsf = open('paramstest.txt','w')
for line in lines:
	if line.startswith('RingNumbers'):
		continue
	if line.startswith('RingRadii'):
		continue
	if line.startswith('OutputFolder'):
		paramsf.write('OutputFolder '+topdir+'/Output\n')
	elif line.startswith('ResultFolder'):
		paramsf.write('ResultFolder '+topdir+'/Results\n')
	else:
		paramsf.write(line)
for idx in range(len(RingNrs)):
	paramsf.write('RingNumbers '+str(RingNrs[idx])+'\n')
	paramsf.write('RingRadii '+str(rads[idx])+'\n')
paramsf.write('BeamSize '+str(BeamSize)+'\n')
paramsf.write('px '+str(px)+'\n')
paramsf.write('RingToIndex '+str(RingToIndex)+'\n')
paramsf.close()
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinDataScanning")+' '+str(nScans),shell=True)
# Parallel after this
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/IndexerScanningOMP")+' paramstest.txt 0 1 '+ str(nScans)+' '+str(numProcs),shell=True)

folder = topdir
xpos, ypos = np.meshgrid(positions,positions)
xpositions = np.transpose(np.transpose(xpos).reshape((nScans*nScans)))
ypositions = np.transpose(np.transpose(ypos).reshape((nScans*nScans)))
nVoxels = nScans*nScans
bestCoordsList = []
files = glob.glob(folder+'/Output/*.csv')
uniqueArr = np.zeros((nVoxels,3))
uniqueOrientArr = []
for voxNr in range(nVoxels):
	# find all files with that blurb
	blurb = '_'+str.zfill(str(voxNr),6)+'_'
	fns = [fn for fn in files if blurb in fn]
	print(blurb,len(fns))
	if len(fns) == 0:
		continue
	PropList = []
	highestConf = -1
	for fn in fns:
		f = open(fn)
		str1= f.readline()
		str1= f.readline()
		line = f.readline().split()
		f.close()
		IAthis = float(line[0][:-1])
		OMthis = [float(a[:-1]) for a in line[1:10]]
		nExp = float(line[-2][:-1])
		nObs = float(line[-1][:-1])
		ConfThis = nObs/nExp
		idnr = int((fn.split('.')[-2]).split('_')[-1])
		# ~ print(idnr)
		if ConfThis > highestConf:
			highestConf = ConfThis
		PropList.append([ConfThis,IAthis,OMthis,idnr,voxNr])
	sortedPropList = sorted(PropList,key=lambda x: (x[0],x[1]),reverse=True)
	uniqueArr[voxNr][0] = xpositions[voxNr]
	uniqueArr[voxNr][1] = ypositions[voxNr]
	if oneSolPerVox == 1:
		uniqueArr[voxNr][1] = 1
		uniqueOrientArr.append(sortedPropList[0])
		continue
	# If more solutions are wanted
	# sortedPropList now has all the orientations, we can try to find the unique ones
	# starting with best orientation, compute miso with all next (unmarked) orientations, if angle is smaller than maxAng, mark the orientation.
	marked = np.zeros(len(sortedPropList))
	uniqueOrients = []
	for idx in range(len(sortedPropList)):
		if marked[idx] == 1:
			continue
		else:
			val1 = sortedPropList[idx]
			uniqueOrients.append(val1)
			orient1 = val1[2]
			for idx2 in range(idx+1,len(sortedPropList)):
				if marked[idx2] == 1:
					continue
				orient2 = sortedPropList[idx2][2]
				ang = GetMisOrientationAngleOM(orient1,orient2,sgnum)
				if ang*rad2deg < maxAng:
					marked[idx2] = 1
	print(['VoxelNr:',voxNr,'nSols:',len(fns),'nUniqueSols:',len(uniqueOrients)])
	uniqueArr[voxNr][2] = len(uniqueOrients)
	uniqueOrientArr.append(uniqueOrients)

# ~ uniqueArr = uniqueArr[uniqueArr[:,2] > 0,:]
# ~ totalGrains = np.sum(uniqueArr[:,2])
# Generate SpotsToIndex.csv file with all jobs to do.
if oneSolPerVox == 0:
	IDsToDo = [orient2[3] for orient in uniqueOrientArr for orient2 in orient]
	IDsToDo2 = [orient2[4] for orient in uniqueOrientArr for orient2 in orient]
elif oneSolPerVox == 1:
	IDsToDo = [orient[3] for orient in uniqueOrientArr]
	IDsToDo2 = [orient[4] for orient in uniqueOrientArr]
nIDs = len(IDsToDo)
with open(folder+'/SpotsToIndex.csv','w') as SpotsF:
	for nr in range(nIDs):
		SpotsF.write(str(IDsToDo[nr])+' '+str(IDsToDo2[nr])+'\n')

# Run FitOrStrainsScanning
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitOrStrainsScanningOMP")+' paramstest.txt 0 1 '+ str(nIDs)+' '+str(numProcs),shell=True)

# go through the output
files2 = glob.glob(folder+'/Results/*.csv')
filesdata = np.zeros((len(files2),43))
i=0;
for fileN in files2:
	f = open(fileN)
	str1 = f.readline()
	data = f.readline().split()
	for j in range(len(data)):
		filesdata[i][j] = float(data[j])
	OM = filesdata[i][1:10]
	quat = BringDownToFundamentalRegionSym(OrientMat2Quat(OM),12,HexSym)
	filesdata[i][39:43] = quat
	i+=1
	f.close()

plt.scatter(filesdata[:,11],filesdata[:,12],s=300,c=filesdata[:,42],cmap='jet');plt.colorbar(); plt.show()

np.savetxt('microstrFull.csv',filesdata,fmt='%.6f',delimiter=',',header='SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,Radius,Completeness,E11,E12,E13,E21,E22,E23,E31,E32,E33,Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4')

print("Time Elapsed: "+str(time.time()-startTime)+" seconds.")
