#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import numpy as np
import time
import sys, os
import os.path

if (len(sys.argv) is not 7):
	print 'Usage: ' + sys.argv[0] + ' parFileName templateFileName fileStemString nFramesPerLayer(include junk frames) nNodes machineName '
	sys.exit()
parFN = sys.argv[1]
templateFN = sys.argv[2]  #os.path.expanduser('~') + '/opt/MIDAS/utils/psf.tif'
fnString = sys.argv[3]
nFramesFile = int(sys.argv[4])
nNodes = sys.argv[5]
machineName = sys.argv[6]
parFile = open(parFN,'r')
templateFile = open(templateFN,'r')
parContents = parFile.readlines()
templateContents = templateFile.read()
fStems = []
usefulLines = []
for line in parContents:
	if (' ' + fnString) in line:
		usefulLines.append(line)
		for word in line.split():
			if fnString in word:
				idx =  (line.split()).index(word)
				fStem = word
		nFiles = int(line.split()[idx-2])
		if nFiles == nFramesFile:
			fStems.append(fStem)
fStems = list(set(fStems))
nLayers = [0] * len(fStems)
firstFileNrs = [0]*len(fStems)
for line in usefulLines:
	for i, fStem in enumerate(fStems):
		if line.split()[idx] == fStem:
			if nLayers[i] == 0:
				# Get the firstFileNr
				firstFileNrs[i] = int(line.split()[idx+1])
			nLayers[i] += 1

runCommand = ''
for i, fStem in enumerate(fStems):
	thisTemplate = templateContents + 'FileStem ' + fStem + '\n'
	thisTemplate = thisTemplate + 'StartFileNrFirstLayer ' + str(firstFileNrs[i]) + '\n'
	fnOut = 'ps_' + fStem + '.txt'
	f = open(fnOut,'w')
	f.write(thisTemplate)
	f.close()
	runCommand += '~/.MIDAS/MIDAS_V4_FarField_Layers ' + fnOut + ' 1 ' + str(nLayers[i]) + ' 1 ' + nNodes + ' ' + machineName + ' hsharma@anl.gov\n'
print runCommand
runFile = open('batchJob_' + fnString + '.sh','w')
runFile.write(runCommand)
