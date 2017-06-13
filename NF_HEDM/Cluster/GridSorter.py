
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
		i = i + 1
		header.append(line)
	else:
		words = line.split("\t")
		print words
		if words[10] < minConfidence:
			gridOut.append(grid[i])
			linesOut.append(line)
		i = i + 1
gridOutF = open(grid_new,'w')
gridOutF.write("%d\n" % len(gridOut))
for line in gridOut:
	gridOutF.write(line)
micOutF = open(micFNout,'w')
for line in header:
	micOutF.write(line)
for line in linesOut:
	micOutF.write(line)
