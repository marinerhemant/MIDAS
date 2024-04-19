import subprocess
from argparse import RawTextHelpFormatter
import argparse
import warnings
import time
import os
import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='''MIDAS_FF, contact hsharma@anl.gov Parameter file must be in the same folder as the desired output folder(SeedFolder)''', formatter_class=RawTextHelpFormatter)
parser.add_argument('-nCPUs',    type=int, required=True, help='Number of CPUs to use')
parser.add_argument('-startLayerNr',type=int,required=True,help='Start Layer Number')
parser.add_argument('-endLayerNr',type=int,required=True,help='End Layer Number')
parser.add_argument('-paramFile', type=str, required=True, help='ParameterFileName')
parser.add_argument('-nNodes', type=str, required=True, help='Number of Nodes')
parser.add_argument('-machineName', type=str, required=True, help='Machine Name')
args, unparsed = parser.parse_known_args()
paramFN = args.paramFile
machineName = args.machineName
startLayerNr = int(args.startLayerNr)
endLayerNr = int(args.endLayerNr)
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

nFrames = endNr - startNr + 1
for layerNr in range(startLayerNr,endLayerNr+1):
	startTime = time.time()
	thisStartNr = startNrFirstLayer + (layerNr-1)*nrFilesPerSweep
	thisT = datetime.datetime.now()
	tod = datetime.date.today()
	folderName = fStem + '_Layer_' + str(layerNr).zfill(4) + '_Analysis_Time_' + str(tod.year) + '_' + str(tod.month).zfill(2) + '_' + str(tod.day).zfill(2) + '_' + str(thisT.hour).zfill(2) + '_' + str(thisT.minute).zfill(2) + '_' + str(thisT.second).zfill(2)
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
	swiftcmd = os.path.expanduser('~/.MIDAS/swift/bin/swift') + ' -config ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/sites.conf') + ' -sites ' + machineName + ' ' + os.path.expanduser('~/opt/MIDAS/FF_HEDM/newCluster/runLayer.swift') + ' -folder=' + thisDir + ' -paramfn='+ baseNameParamFN + ' -nrNodes=' + str(nNodes) + ' -nFrames=' + str(nFrames) + ' -numProcs='+ str(numProcs)
	print(swiftcmd)
	subprocess.call(swiftcmd,shell=True)
	print("Time Elapsed: "+str(time.time()-startTime)+" seconds.")
