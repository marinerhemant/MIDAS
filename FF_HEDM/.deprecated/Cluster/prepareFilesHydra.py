#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import sys
import os

psfn = sys.argv[1]
detnr = int(sys.argv[2])

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
	if line.split()[0] == 'DarkStem':
		darkstem = line.split()[1]
	if line.split()[0] == 'RawFolder':
		rawfolder = line.split()[1]
	if line.split()[0] == 'DarkNum':
		darknum = int(line.split()[1])
	if line.split()[0] == 'Padding':
		padding = int(line.split()[1])

LsdMean = 0
for i in range(4):
	LsdMean += float(lsd[i])/4

f = open(psfn,'a')
f.write('LsdMean '+str(LsdMean)+'\n')
if detnr is not 0:
	darkname = rawfolder + '/' + darkstem + '_' + str(darknum).zfill(padding) + '.ge'+str(detnr)
	f.write('Ext .ge'+str(detnr)+'\n')
	f.write('Dark '+darkname+'\n')
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
