#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys, os
import os.path

paramfn = sys.argv[1]
layernr = int(sys.argv[2])
paramcontents = open(paramfn,'r').readlines()

for line in paramcontents:
	if line.startswith('StartFileNrFirstLayer '):
		overallStartNr = int(line.split()[1])
	elif line.startswith('NrFilesPerSweep '):
		nrFilesPerSweep = int(line.split()[1])
startNr = overallStartNr + nrFilesPerSweep*(layernr-1)
print(startNr)
