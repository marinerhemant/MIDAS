import subprocess
import numpy as np
from argparse import RawTextHelpFormatter
import argparse
import warnings
import time
import os,sys,glob
import os.path
from pathlib import Path
import shutil
from math import floor
utilsDir = os.path.expanduser('~/opt/MIDAS/utils/')
sys.path.insert(0,utilsDir)
from calcMiso import *
import warnings
warnings.filterwarnings('ignore')
from oneSolPerVoxTomoFilter import runRecon
from multSolPerVox import runReconMulti
import pandas as pd

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

rad2deg = 57.2957795130823

def CalcEtaAngleAll(y, z):
	alpha = 57.2957795130823*np.arccos(z/np.linalg.norm(np.array([y,z]),axis=0))
	alpha[y>0] *= -1
	return alpha

startTime = time.time()

warnings.filterwarnings('ignore')
parser = MyParser(description='''
MIDAS_PF, contact hsharma@anl.gov Parameter file must be in the same folder as the desired output folder(SeedFolder).
Provide positions.csv file (negative positions with respect to actual motor position. 
Motor position is normally position of the rotation axis, opposite to the voxel position.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-nCPUs', type=int, required=True, help='Number of CPUs to use')
parser.add_argument('-nCPUsLocal', type=int, required=False, default=3, help='Local Number of CPUs to use')
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
numProcsLocal = args.nCPUsLocal
nNodes = args.nNodes
os.environ["nNODES"] = str(nNodes)
os.environ["nCPUs"] = str(numProcs)

baseNameParamFN = paramFN.split('/')[-1]
homedir = os.path.expanduser('~')
paramContents = open(paramFN).readlines()
RingNrs = []
nMerges = 0
omegaOffset = 0
micFN = ''
for line in paramContents:
	if line.startswith('StartFileNrFirstLayer'):
		startNrFirstLayer = int(line.split()[1])
	if line.startswith('NrFilesPerSweep'):
		nrFilesPerSweep = int(line.split()[1])
	if line.startswith('MicFile'):
		micFN = line.split()[1]
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


nFrames = endNr - startNr + 1
if doPeakSearch == 1:
	positions = open(topdir+'/positions.csv').readlines()
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
		# Call the PeaksFittingOMP code using swift to run PeakSearch on each scan in parallel on multiple nodes.
		swiftcmd  = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') 
		swiftcmd += ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runPeakSearchOnly.swift') + ' -folder=' 
		swiftcmd += thisDir + ' -paramfn='+ baseNameParamFN + ' -nrNodes=' + str(nNodes) + ' -nFrames=' + str(nFrames) + ' -numProcs='+ str(numProcs)
		print(swiftcmd)
		subprocess.call(swiftcmd,shell=True)
		if omegaOffset != 0:
			fns = glob.glob('Temp/*PS.csv')
			for fn in fns:
				df = pd.read_csv(fn,delimiter=' ')
				if df.shape[0] == 0:
					continue
				omega_this = df['Omega(degrees)'][0]
				omegaOffsetThis = omegaOffset*layerNr
				omegaOffsetThis = omegaOffsetThis%360.0
				omega_new = omega_this - omegaOffsetThis
				df['Omega(degrees)'] = omega_new
				df.to_csv(fn,sep=' ',header=True,float_format='%.6f',index=False)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/MergeOverlappingPeaksAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalcRadiusAll")+' '+baseNameParamFN,shell=True)
		subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/FitSetup")+' '+baseNameParamFN,shell=True)
		Result = np.genfromtxt(f'Radius_StartNr_{startNr}_EndNr_{endNr}.csv',skip_header=1,delimiter=' ')
		if Result.shape[0]==0:
			shutil.copy2('InputAllExtraInfoFittingAll.csv',topdir+'/InputAllExtraInfoFittingAll'+str(layerNr-1)+'.csv')
			continue
		uniqueRings,uniqueIndices = np.unique(Result[:,13],return_index=True)
		ringPowderIntensity = []
		for iter in range(len(uniqueIndices)):
			ringPowderIntensity.append([uniqueRings[iter],Result[uniqueIndices[iter],16]])
		ringPowderIntensity = np.array(ringPowderIntensity)
		dfAllF = pd.read_csv('InputAllExtraInfoFittingAll.csv',delimiter=' ',skipinitialspace=True)
		dfAllF.loc[dfAllF['GrainRadius']>0.001,'%YLab'] += ypos
		dfAllF.loc[dfAllF['GrainRadius']>0.001,'YOrig(NoWedgeCorr)'] += ypos
		dfAllF['Eta'] = CalcEtaAngleAll(dfAllF['%YLab'],dfAllF['ZLab'])
		dfAllF['Ttheta'] = rad2deg*np.arctan(np.linalg.norm(np.array([dfAllF['%YLab'],dfAllF['ZLab']]),axis=0)/Lsd)
		
		for iter in range(len(ringPowderIntensity)):
			ringNr = ringPowderIntensity[iter,0]
			powInt = ringPowderIntensity[iter,1]
			dfAllF.loc[dfAllF['RingNumber']==ringNr,'GrainRadius'] *= powInt**(1/3)
		outFN2 = topdir+'/InputAllExtraInfoFittingAll'+str(layerNr-1)+'.csv'
		dfAllF.to_csv(outFN2,sep=' ',header=True,float_format='%.6f',index=False)
		shutil.copy2(thisDir+'/paramstest.txt',topdir+'/paramstest.txt')
		shutil.copy2(thisDir+'/hkls.csv',topdir+'/hkls.csv')
else:
	if nMerges!=0:
		os.chdir(topdir)
		if os.path.exists('original_positions.csv'):
			shutil.move('original_positions.csv','positions.csv')
		positions = open(topdir+'/positions.csv').readlines()
		for layerNr in range(0,nMerges*(nScans//nMerges)):
			if os.path.exists(f'original_InputAllExtraInfoFittingAll{layerNr}.csv'):
				shutil.move(f'original_InputAllExtraInfoFittingAll{layerNr}.csv',f'InputAllExtraInfoFittingAll{layerNr}.csv')

if nMerges != 0:
	shutil.move('positions.csv','original_positions.csv')
	for layerNr in range(0,nMerges*(nScans//nMerges)):
		if os.path.exists(f'InputAllExtraInfoFittingAll{layerNr}.csv'):
			shutil.move(f'InputAllExtraInfoFittingAll{layerNr}.csv',f'original_InputAllExtraInfoFittingAll{layerNr}.csv')
	subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/mergeScansScanning")+f" {nMerges*(nScans//nMerges)} {nMerges} {2*px} {2*omegaStep} {numProcsLocal}",shell=True)
	positions = open(topdir+'/positions.csv').readlines()
	nScans = int(floor(nScans / nMerges))

positions = open(topdir+'/positions.csv').readlines()
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
if len(micFN) > 0:
	paramsf.write(f'MicFile {micFN}\n')
paramsf.close()

subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/SaveBinDataScanning")+' '+str(nScans),shell=True)
swiftcmdIdx = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf')
swiftcmdIdx += ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runIndexingScanning.swift') + ' -folder=' 
swiftcmdIdx += topdir + ' -nrNodes=' + str(nNodes) + ' -nScans=' + str(nScans) + ' -numProcs='+ str(numProcs)
print(swiftcmdIdx)
subprocess.call(swiftcmdIdx,shell=True)

if oneSolPerVox==1:
	runRecon(topdir,nScans,sgnum,numProcs,numProcsLocal,maxang=2,tol_eta=1,tol_ome=1,thresh_reqd=1,nNodes=nNodes,machineName=machineName)
else:
	runReconMulti(topdir,nScans,positions,sgnum,numProcs,numProcsLocal,nNodes=nNodes,machineName=machineName)
print("Time Elapsed: "+str(time.time()-startTime)+" seconds.")
