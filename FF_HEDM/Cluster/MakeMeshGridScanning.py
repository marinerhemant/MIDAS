#!/APSshare/anaconda/x86_64/bin/python

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys
import os
import numpy as np

configFile = sys.argv[1]
pscontent = open(configFile).readlines()
for line in pscontent:
	line = [s for s in line.rstrip().split() if s]
	if len(line) > 0:
		if line[0] == 'PositionsFile':
			positionsFile = os.getcwd() + '/' + line[1]

positionlines = open(positionsFile).readlines()
positions = []
for line in positionlines:
	if line[0] == '%':
		continue
	positions.append(1000 * float(line.split('\t')[0]))

x = np.array(positions)
y = np.array(positions)
xv, yv = np.meshgrid(x,y)
nrpoints = len(xv)

gridfile = open('grid.txt','w')
gridfile.write(str(nrpoints)+'\n')
for [i,pos] in enumerate(xv):
	gridfile.write(str(pos)+' '+str(yv[i])+' '+i+'\n')
