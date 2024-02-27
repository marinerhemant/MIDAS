import subprocess
import numpy as np
from argparse import RawTextHelpFormatter
import argparse
import warnings
import time
import os,sys,glob
import datetime
from pathlib import Path
import shutil
from math import acos,sqrt,atan,floor
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from calcMiso import *
import matplotlib.pyplot as plt
from skimage.transform import iradon
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from oneSolPerVoxTomoFilter import runRecon
from multSolPerVox import runReconMulti

def CalcEtaAngle(y, z):
	alpha = 57.2957795130823*acos(z/sqrt(y*y+z*z))
	if (y>0):
		alpha = -alpha
	return alpha

startTime = time.time()

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='''MIDAS_PF, contact hsharma@anl.gov Parameter file must be in the same folder as the desired output folder(SeedFolder).
Provide positions.csv file (negative positions with respect to actual motor position. Motor position is normally position of the rotation axis, opposite to the voxel position.''', formatter_class=RawTextHelpFormatter)
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

baseNameParamFN = paramFN.split('/')[-1]
homedir = os.path.expanduser('~')
paramContents = open(paramFN).readlines()
RingNrs = []
nMerges = 0
omegaOffset = 0
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
	if line.startswith('SpaceGroup'):
		sgnum = int(line.split()[1])
	if line.startswith('nStepsToMerge'):
		nMerges = int(line.split()[1])
	if line.startswith('omegaOffsetBetweenScans'):
		omegaOffset = float(line.split()[1])
	if line.startswith('nScans'):
		nScans = int(line.split()[1])
	if line.startswith('Lsd'):
		Lsd = float(line.split()[1])
	if line.startswith('OverAllRingToIndex'):
		RingToIndex = int(line.split()[1])
	if line.startswith('BeamSize'):
		BeamSize = float(line.split()[1])
	if line.startswith('OmegaStep'):
		omegaStep = float(line.split()[1])
	if line.startswith('px'):
		px = float(line.split()[1])
	if line.startswith('RingThresh'):
		RingNrs.append(int(line.split()[1]))

subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+' ' + paramFN,shell=True)
hkls = np.genfromtxt('hkls.csv',skip_header=1)
_rnr,idx = np.unique(hkls[:,4],return_index=True)
hkls = hkls[idx,:]
rads = []
for rnr in RingNrs:
	for hkl in hkls:
		if hkl[4] == rnr:
			rads.append(hkl[-1])

rads = [hkl[-1] for rnr in RingNrs for hkl in hkls if hkl[4] == rnr]
print(RingNrs)
print(rads)

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
		# Call the PeaksFittingOMP code using swift to run PeakSearch on each scan in parallel
		swiftcmd = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') + ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runPeakSearchOnly.swift') + ' -folder=' + thisDir + ' -paramfn='+ baseNameParamFN + ' -nrNodes=' + str(nNodes) + ' -nFrames=' + str(nFrames) + ' -numProcs='+ str(numProcs)
		print(swiftcmd)
		subprocess.call(swiftcmd,shell=True)
		# These need to be done sequentially
		if omegaOffset != 0:
			# We need to open each Temp/* file and modify its omega, write back
			fns = glob.glob('Temp/*PS.csv')
			for fn in fns:
				with open(fn,'r') as f:
					lines = f.readlines()
					head_this = lines[0]
				with open(fn,'w') as f:
					f.write(head_this)
					if len(lines)==1: continue
					omega_this = float(lines[1].split()[2])
					omegaOffsetThis = omegaOffset*layerNr
					omegaOffsetThis = omegaOffsetThis%360.0
					omega_new = omega_this - omegaOffsetThis
					for line in lines[1:]:
						line_new = line.split()
						line_new[2] = f"{omega_new:.6f}"
						f.write(' '.join(line_new))
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetup")+' '+baseNameParamFN,shell=True)
		# Now do the position correction
		AllF = open('InputAllExtraInfoFittingAll.csv','r')
		allcontents = AllF.readlines()
		AllF.close()
		AllF = open(topdir+'/InputAllExtraInfoFittingAll'+str(layerNr-1)+'.csv','w')
		IDRings = np.genfromtxt('IDRings.csv',skip_header=1,delimiter=' ')
		Result = np.genfromtxt(f'Radius_StartNr_{startNr}_EndNr_{endNr}.csv',skip_header=1,delimiter=' ')
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
				y = y + ypos # If positions.csv is flipped, this should be positive.
				if (y*y+z*z) < np.finfo(np.float32).eps:
					continue # We are skipping the 0 lines, but does this ruin the ordering of the peaks?
				Eta = CalcEtaAngle(y,z)
				Ttheta = 57.2957795130823*atan(sqrt(y*y+z*z)/Lsd)
				yOrigNoW = yOrigNoW + ypos # Confirm this
				# Get intensity from Radius... file
				if nMerges!=0:
					if len(IDRings.shape) == 1:
						if len(Result.shape) == 2:
							intensitySpot = Result[0,1]
						else:
							intensitySpot = Result[1]
					else:
						origID = IDRings[IDRings[:,2] == int(ID),1]
						intensitySpot = Result[Result[:,0]==origID,1]
					outstr = '{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n'.format(y,z,ome,grR,ID,RNr,Eta,Ttheta,omeIniNoW,yOrigNoW,zOrigNoW,yDet,zDet,omegaDet,intensitySpot)
				else:
					outstr = '{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n'.format(y,z,ome,grR,ID,RNr,Eta,Ttheta,omeIniNoW,yOrigNoW,zOrigNoW,yDet,zDet,omegaDet)
				AllF.write(outstr)
		AllF.close()
		shutil.copy2(thisDir+'/paramstest.txt',topdir+'/paramstest.txt')
		shutil.copy2(thisDir+'/hkls.csv',topdir+'/hkls.csv')

if nMerges != 0:
	# We want to merge every nMerges datasets
	# We will update nScans, positions and positions.csv
	# We will move the InputallExtraInfoFittingAll*.csv files to original_InputAllExtraInfoFittingAll*.csv, then generate merged files
	# To merge peaks, we will use the following reasoning: peak position should be within +/- 2 pixels, 2*omegaStep
	nFinScans = int(floor(nScans / nMerges))
	shutil.move('positions.csv','original_positions.csv')
	posF = open('positions.csv','w')
	headOut = '%YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni(NoWedgeCorr) YOrig(NoWedgeCorr) ZOrig(NoWedgeCorr) YOrig(DetCor) ZOrig(DetCor) OmegaOrig(DetCor)'
	positionsNew = np.zeros(nFinScans)
	for scanNr in range(nFinScans):
		thisPosition = float(positions[scanNr])
		outFAll = open(f'InputAllExtraInfoFittingAll{scanNr}.csv','w')
		outFAll.write('%YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta OmegaIni(NoWedgeCorr) YOrig(NoWedgeCorr) ZOrig(NoWedgeCorr) YOrig(DetCor) ZOrig(DetCor) OmegaOrig(DetCor)')
		startScanNr = scanNr*nMerges
		shutil.move(f'InputAllExtraInfoFittingAll{startScanNr}.csv',f'original_InputAllExtraInfoFittingAll{startScanNr}.csv')
		spots = np.genfromtxt(f'original_InputAllExtraInfoFittingAll{startScanNr}.csv',skip_header=1)
		if len(spots.shape) < 2:
			spots = np.zeros((2,14))
			spots[:,2] = -360 # Hook to keep sanity
		for scan in range(1,nMerges):
			thisScanNr = startScanNr + scan
			thisPosition += float(positions[thisScanNr])
			shutil.move(f'InputAllExtraInfoFittingAll{thisScanNr}.csv',f'original_InputAllExtraInfoFittingAll{thisScanNr}.csv')
			spots2 = np.genfromtxt(f'original_InputAllExtraInfoFittingAll{thisScanNr}.csv',skip_header=1)
			if (len(spots2.shape)<2): continue
			for spot in spots2:
				# Check for all spots which are close to this spot
				filteredSpots = spots[np.fabs(spots[:,0]-spot[0])<2*px,:]
				found = 1
				if (len(filteredSpots) == 0): found = 0
				else:
					if (len(filteredSpots.shape) > 1):
						filteredSpots = filteredSpots[np.fabs(filteredSpots[:,1]-spot[1])<2*px,:]
					else:
						filteredSpots = filteredSpots[np.fabs(filteredSpots[1]-spot[1])<2*px,:]
					if (len(filteredSpots) == 0): found = 0
					else:
						if (len(filteredSpots.shape) > 1):
							filteredSpots = filteredSpots[np.fabs(filteredSpots[:,2]-spot[2])<2*omegaStep,:]
						else:
							filteredSpots = filteredSpots[np.fabs(filteredSpots[2]-spot[2])<2*omegaStep,:]
						if (len(filteredSpots) == 0): found = 0
						elif len(filteredSpots.shape) == 1:
							# Generate mean values weighted by spot integrated intensity
							rowNr = np.argwhere(spots[:,4]==filteredSpots[4]).item()
							weightedValSpots = spots[rowNr,:]*spots[rowNr,-1]
							weightedValSpot = spot[:]*spot[-1]
							totalWts = spots[rowNr,-1] + spot[-1]
							newVals = (weightedValSpot+weightedValSpots)/(totalWts)
							spots[rowNr,:] = newVals
				if found == 0:
					spots = np.vstack((spots,spot))
		positionsNew[scanNr] = thisPosition/nMerges
		spots = spots[spots[:,2]!=-360,:]
		print(f'ScanNr: {scanNr}, position: {positionsNew[scanNr]}, nSpots: {spots.shape[0]}')
		# Update the new positions array
		if (len(spots.shape)>1): np.savetxt(outFAll,spots[:,:-1],fmt="%12.5f",delimiter="  ")
	np.savetxt('positions.csv',positionsNew,fmt='%.5f',delimiter=' ')
	positions = positionsNew
	nScans = nFinScans

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
	if line.startswith('RingToIndex'):
		continue
	if line.startswith('BeamSize'):
		continue
	if line.startswith('px'):
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
swiftcmdIdx = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') + ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runIndexingScanning.swift') + ' -folder=' + topdir + ' -nrNodes=' + str(nNodes) + ' -nScans=' + str(nScans) + ' -numProcs='+ str(numProcs)
print(swiftcmdIdx)
subprocess.call(swiftcmdIdx,shell=True)

if oneSolPerVox==1:
	runRecon(topdir,startNrFirstLayer,nScans,endNr,sgnum,numProcs,nrFilesPerSweep=nrFilesPerSweep,removeDuplicates=1,maxang=3,tol_eta=2,tol_ome=2,findUniques=1,thresh_reqd=1)
else:
	runReconMulti(topdir,nScans,positions,sgnum,numProcs)
print("Time Elapsed: "+str(time.time()-startTime)+" seconds.")
