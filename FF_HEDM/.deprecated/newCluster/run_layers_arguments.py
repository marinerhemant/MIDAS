#### Copy parameter file over to the outputFolder before execution

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
import sys

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='''MIDAS_FF, contact hsharma@anl.gov. Run with -h as argument to see help.''', formatter_class=RawTextHelpFormatter)
parser.add_argument('-nCPUs',    type=int, required=True, help='Number of CPUs to use')
parser.add_argument('-layerNr',type=int,required=True,help='Start Layer Number')
parser.add_argument('-paramFile', type=str, required=True, help='ParameterFileName')
parser.add_argument('-nNodes', type=str, required=True, help='Number of Nodes')
parser.add_argument('-machineName', type=str, required=True, help='Machine Name')
parser.add_argument('-startFileNrFirstLayer', type=str, required=True, help='Starting file number for first layer')
parser.add_argument('-numberOfFilesPerLayer', type=str, required=True, help='Difference between start file number for succssive layers')
parser.add_argument('-fileStem', type=str, required=True, help='File stem before the digits without _')
parser.add_argument('-outputFolder', type=str, required=True, help='Folder to save the data')
parser.add_argument('-nrFramesPerLayer', type=str, required=True, help='Number of frames in a layer')
parser.add_argument('-darkFileName', type=str, required=True, help='Full name for the dark file')
parser.add_argument('-rawFolder', type=str, required=True, help='Folder where the raw files will be (including dark)')
parser.add_argument('-wavelength', type=str, required=True, help='Wavelength [Angstorm]')
parser.add_argument('-headerSize', type=str, required=True, help='Beginning of file in bytes to skip [eg 8396800]')
parser.add_argument('-padding', type=str, required=True, help='Number of digits in the filename')
parser.add_argument('-ext', type=str, required=True, help='File extension [eg .ge3.edf]')
parser.add_argument('-imageTransformations', type=str, required=True, help='Filps etc. that need to be done to the image. 0 for Sector 1 GE detectors, 2 for Sector 1 Pilatus. Always supply with double quotes if multiple like "1 2 3", otherwise just a number.')
parser.add_argument('-beamCurrent', type=str, required=True, help='IOC fraction value to normalize data if needed. If not 1, use with caution, it will scale all the intensities')
parser.add_argument('-saturationIntensity', type=str, required=True, help='Maximum usable intensity from the detector. Eg. 12000 for geX detectors')
parser.add_argument('-omegaStep', type=str, required=True, help='Step size for rotation during acquisition. If left handed rotation (eg. Aerotech), should be negative')
parser.add_argument('-omegaFirstFrame', type=str, required=True, help='Rotation angle for the first frame. If left handed rotation (eg. Aerotech), should be negative of real value')
parser.add_argument('-pixelSize', type=str, required=True, help='Pixel size of the detector [microns]')
parser.add_argument('-nrPixelsY', type=str, required=True, help='Horizontal number of pixels on the detector')
parser.add_argument('-nrPixelsZ', type=str, required=True, help='Vertical number of pixels on the detector')
parser.add_argument('-omegaRange', type=str, required=True, help='Omega Range for the experiment. Must be passed as "startOme stopOme startOme2 stopOme2..." with the double quotes')
parser.add_argument('-boxSize', type=str, required=True, help='Useful detector area. Must be passed as "yMin1 yMax1 zMin1 zMax1 yMin2 yMax2 zMin2 zMax2..." with the double quotes. Must have one yzMinMax set for each pair in omegaRange')
args, unparsed = parser.parse_known_args()
paramFN = args.paramFile
machineName = args.machineName
startLayerNr = int(args.layerNr)
endLayerNr = int(args.layerNr)
numProcs = args.nCPUs
nNodes = args.nNodes
startFileNrFirstLayer = int(args.startFileNrFirstLayer)
numberOfFilesPerLayer = int(args.numberOfFilesPerLayer)
fileStem = args.fileStem
outFolder = args.outputFolder
nFrames = int(args.nrFramesPerLayer)
startNr = 1
endNr = nFrames
darkFN = args.darkFileName
rawFolder = args.rawFolder
dark = rawFolder+'/'+darkFN
wavelength = float(args.wavelength)
head = int(args.headerSize)
pad = int(args.padding)
ext = args.ext
transformations = args.imageTransformations
if len(transformations) == 1:
	imTransOptA = 1
	imTransOpt = int(transformations)
else:
	imTransOptA = 0
	imTransOpt = [int(transfor) for transfor in transformations.split()]
beamCurrent = float(args.beamCurrent)
upperBoundThreshold = float(args.saturationIntensity)
omegaStep = float(args.omegaStep)
omegaFirstFile = float(args.omegaFirstFrame)
px = float(args.pixelSize)
nrPixelsY = int(args.nrPixelsY)
nrPixelsZ = int(args.nrPixelsZ)
omegas = args.omegaRange
omegaRange = [float(om) for om in omegas.split()]
boxes = args.boxSize
boxSize = [float(bs) for bs in boxes.split()]
os.environ["nNODES"] = str(nNodes)
os.environ["nCPUs"] = str(numProcs)

if nrPixelsY != nrPixelsZ:
	print("Only works for square detector for now. Exiting!")
	sys.exit()
if len(omegaRange)%2 != 0:
	print('Wrong OmegaRange. Exiting!')
	sys.exit()
if len(boxSize)%4 != 0:
	print('Wrong BoxSizes. Exiting!')
	sys.exit()
	
ps_template_contents = open(paramFN).read()
paramFN = paramFN + '.upd'
### Add arguments to the parameter file
f = open(paramFN,'w')
f.write(ps_template_contents)
f.write(f'NumPhases 1\n')
f.write(f'PhaseNr 1\n')
f.write(f'StartFileNrFirstLayer {startFileNrFirstLayer}\n')
f.write(f'NrFilesPerSweep {numberOfFilesPerLayer}\n')
f.write(f'FileStem {fileStem}\n')
f.write(f'SeedFolder {outFolder}\n')
f.write(f'StartNr {startNr}\n')
f.write(f'EndNr {endNr}\n')
f.write(f'Dark {dark}\n')
f.write(f'RawFolder {rawFolder}\n')
f.write(f'Wavelength {wavelength}\n')
f.write(f'HeadSize {head}\n')
f.write(f'Padding {pad}\n')
f.write(f'Ext {ext}\n')
if imTransOptA == 1:
	f.write(f'ImTransOpt {imTransOpt}\n')
else:
	for i in range(len(imTransOpt)):
		f.write(f'ImTransOpt {imTransOpt[i]}\n')
f.write(f'BeamCurrent {beamCurrent}\n')
f.write(f'UpperBoundThreshold {upperBoundThreshold}\n')
f.write(f'OmegaStep {omegaStep}\n')
f.write(f'OmegaFirstFile {omegaFirstFile}\n')
f.write(f'px {px}\n')
f.write(f'NrPixels {nrPixelsY}\n')
for i in range(int(len(omegaRange)/2)):
	f.write(f'OmegaRange {omegaRange[i*2]} {omegaRange[i*2+1]}\n')
for i in range(int(len(boxSize)/4)):
	f.write(f'BoxSize {boxSize[i*4]} {boxSize[i*4+1]} {boxSize[i*4+2]} {boxSize[i*4+3]}\n')
f.write('NewType 1\n')
f.write('UseFriedelPairs 1\n')
f.write('GlobalPosition 0\n')
f.write('DoFit 0\n')
f.write('StepSizePos 100\n')
f.write('StepSizeOrient 0.1\n')
f.write('OmeBinSize 0.2\n')
f.write('EtaBinSize 0.2\n')
f.close()

baseNameParamFN = paramFN.split('/')[-1]
homedir = os.path.expanduser('~')
paramContents = open(paramFN).readlines()
startNrFirstLayer = startFileNrFirstLayer
nrFilesPerSweep = numberOfFilesPerLayer
fStem = fileStem
topdir = outFolder

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
