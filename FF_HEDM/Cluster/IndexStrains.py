#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#
import sys
import os
from subprocess import call
from os.path import expanduser
from math import floor

home = expanduser("~")
pathsf = open(home + '/.MIDAS/paths')
paths = pathsf.readlines()
for line in paths:
	if 'BINFOLDER' in line:
		binfolder = line.split('=')[1].split('\n')[0]
chunkNr = int(sys.argv[1])
folder = sys.argv[2]
os.chdir(folder)
IDs = open('SpotsToIndex.csv').readlines()
num_lines = len(IDs)
multF = 1 + int(floor(num_lines/2000))
startRowNr = multF*(chunkNr-1)
endRowNr = multF*chunkNr
for rown in range(startRowNr,endRowNr,1):
	if rown > num_lines-1:
		break
	ID = IDs[rown]
	print(ID)
	call([binfolder+'/IndexerLinuxArgsShm','paramstest.txt',str(ID)])
	call([binfolder+'/FitPosOrStrains','paramstest.txt',str(ID)])
