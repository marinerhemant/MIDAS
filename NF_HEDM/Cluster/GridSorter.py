
#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys

grid_old = 'grid_all.txt'
gid_new = 'grid.txt'
micFN = sys.argv[1]
minConfidence = float(sys.argv[2])
micF = open(micFN,'r')
gridF = open(grid_old,'r')
mic = micF.readlines()
grid = gridF.readlines()
i = 1
gridOut = []
for line in mic:
	if line[0] == '%':
		i = i + 1
	else:
		words = line.split("\t")
		if words[10] < minConfidence:
			gridOut.append(grid[i])
			tmpOut.append(grid[i])
		i = i + 1
gridOutF = open(grid_new,'w')
gridOutF.write("%d\n" % len(gridOut))
for line in gridOut:
	gridOutF.write(line)
