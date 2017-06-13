
#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys

grid_old = 'grid_all.txt'
grid_new = 'grid.txt'
micFN = sys.argv[1]
micFNout = micFN + '_ffResult'
minConfidence = float(sys.argv[2])
lowMinConfidence = float(sys.argv[3])
micF = open(micFN,'r')
gridF = open(grid_old,'r')
mic = micF.readlines()
grid = gridF.readlines()
i = 1
gridOut = []
linesOut = []
header = []
for line in mic:
	if line[0] == '%':
		header.append(line)
	else:
		words = line.split("\t")
		if float(words[10]) < minConfidence and float(words[10]) > lowMinConfidence:
			gridOut.append(grid[i])
			linesOut.append(line)
		i = i + 1
gridOutF = open(grid_new,'w')
print "Total number of grid points to try again: " + str(len(gridOut))
gridOutF.write("%d\n" % len(gridOut))
for line in gridOut:
	gridOutF.write(line)
gridOutF.close()
micOutF = open(micFNout,'w')
for line in header:
	micOutF.write(line)
for line in linesOut:
	micOutF.write(line)
micOutF.close()
