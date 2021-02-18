#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys, os
import os.path

paramfn = sys.argv[1]
layernr = int(sys.argv[2])
darkfn = ''
if (len(sys.argv)>3):
	ext = '.ge'+sys.argv[3]
	darkfn = ''
paramcontents = open(paramfn,'r').readlines()

for line in paramcontents:
	if line.startswith('StartFileNrFirstLayer '):
		overallStartNr = int(line.split()[1])
	elif line.startswith('NrFilesPerSweep '):
		nrFilesPerSweep = int(line.split()[1])
	elif line.startswith('FileStem '):
		fileStem = line.split()[1]
	elif line.startswith('RawFolder '):
		rawFolder = line.split()[1]
	elif line.startswith('Dark '):
		darkfn = line.split()[1]
	elif line.startswith('Padding '):
		padding = int(line.split()[1])
	elif line.startswith('Ext '):
		ext = line.split()[1]

if darkfn != '':
	if not os.path.isfile(darkfn):
		print('DARK FILE ' + darkfn + ' does not exist. Please check!!!')
		# ~ sys.exit(1)

startNr = overallStartNr + nrFilesPerSweep*(layernr-1)
for fnr in range(startNr,startNr+nrFilesPerSweep):
	fn = rawFolder + '/' + fileStem + '_' + str(fnr).zfill(padding) + ext
	if not os.path.isfile(fn):
		print('FILE ' + fn + ' does not exist. Please check!!!')
		# ~ sys.exit(1)
