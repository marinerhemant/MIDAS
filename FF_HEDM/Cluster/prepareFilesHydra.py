#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys
import os
from os.path import expanduser

home = expanduser("~")
pathsf = open(home + '/.MIDAS/paths')
paths = pathsf.readlines()
for line in paths:
	if 'BINFOLDER' in line:
		binfolder = line.split('=')[1].split('\n')[0]
	if 'PFDIR' in line:
		pfdir = line.split('=')[1].split('\n')[0]

folder = os.getcwd()
psfn = sys.argv[1]

print [psfn, sLayer, eLayer, doPeakSearch, nNodes, machineName, em]

lsd = []
bc = []
ts = []
ps = []
paramContents = open(psfn,'r').readlines()
for line in paramContents:
	if line == '\n':
		continue
	if line.split()[0] == 'DetParams':
		lsd.append(line.split()[1])
		bc.append([line.split()[2],line.split()[3]])
		ts.append([line.split()[4],line.split()[5],line.split()[6]])
		ps.append([line.split()[7],line.split()[8],line.split()[9],line.split()[10]])
	if line.split()[0] == 'NumDetectors':
		ndetectors = int(line.split()[1])

for detnr in range(1,ndetectors+1):
	newfolder = folder+'/Detector'+str(detnr)+'/'
	newfn = psfn+'_det'+str(detnr)
	os.system('mkdir '+newfolder)
	os.system('cp '+folder+'/'+psfn+' '+newfolder+newfn)
	f = open(newfolder+newfn,'a')
	f.write('SeedFolder '+newfolder+'\n')
	f.write('Lsd '+lsd[detnr-1]+'\n')
	f.write('BC '+bc[detnr-1][0]+' '+bc[detnr-1][1]+'\n')
	f.write('tx '+ts[detnr-1][0]+'\n')
	f.write('ty '+ts[detnr-1][1]+'\n')
	f.write('tz '+ts[detnr-1][2]+'\n')
	f.write('p0 '+ps[detnr-1][0]+'\n')
	f.write('p1 '+ps[detnr-1][1]+'\n')
	f.write('p2 '+ps[detnr-1][2]+'\n')
	f.write('RhoD '+ps[detnr-1][3]+'\n')
	f.close()
