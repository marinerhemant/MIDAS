#!/clhome/TOMO1/opt/midasconda/bin/python

#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#


import sys
import os
from os.path import expanduser
from subprocess import call
from math import acos,sqrt, atan

def CalcEtaAngle(y, z):
	alpha = 57.2957795130823*acos(z/sqrt(y*y+z*z))
	if (y>0):
		alpha = -alpha
	return alpha

print("We are going to process the flt files to generate MIDAS input.")

if len(sys.argv) == 1:
	print( 'To use this code, add the following parameters to the params.txt and then use as')
	print( '\tMergeMultipleScans.py params.txt')
	print( 'Parameters to add: nLayers, PositionsFile, Padding, FltStem, FltExt, OutDirPath')
	print( "Give no beam center, but provide a BCAll parameter for the rotation axis at 0,0")
	print( "Positions file MUST have positions in mm.")
	sys.exit(1)

configFile = sys.argv[1]
nLayers = 163
parentfolder = os.getcwd()
positionsFile = os.getcwd() +'/positions.csv'
padding = 6
fstem = os.getcwd() + '/flt/peak_'
ext = '.ext.flt'
outdir = os.getcwd() +'/TOMO_Recon'

pscontent = open(configFile).readlines()
rings = []
threshs = []
bcall = [0,0]
for line in pscontent:
	line = [s for s in line.rstrip().split() if s]
	if len(line) > 0:
		if line[0] == "RingThresh":
			rings.append(int(line[1]))
			threshs.append(float(line[2]))
		elif line[0] == 'FileStem':
			filestem = line[1]
		elif line[0] == 'OverAllRingToIndex':
			ringtoindex = line[1]
		elif line[0] == 'nLayers':
			nLayers = int(line[1])
		elif line[0] == 'PositionsFile':
			positionsFile = os.getcwd() + '/' + line[1]
		elif line[0] == 'Padding':
			padding = int(line[1])
		elif line[0] == 'FltStem':
			fstem = os.getcwd() + '/' + line[1]
		elif line[0] == 'FltExt':
			ext = line[1]
		elif line[0] == 'BinStem':
			binstem = os.getcwd() + '/' + line[1]
		elif line[0] == 'BinExt':
			binext = line[1]
		elif line[0] == 'OutDirPath':
			outdir = os.getcwd() + '/' + line[1]
		elif line[0] == 'BCAll':
			bcall[0] = float(line[1])
			bcall[1] = float(line[2])
		elif line[0] == 'px':
			px = float(line[1])
		elif line[0] == 'Lsd':
			Lsd = float(line[1])

positions = open(positionsFile).readlines()
call(['mkdir','-p',outdir])
call(['cp',configFile,outdir])
os.chdir(outdir)
paths = open(expanduser("~")+'/.MIDAS/paths').readlines()
for line in paths:
	if 'BINFOLDER' in line:
		binfolder = line.split('=')[1].rstrip()

call([binfolder+'/GetHKLList',configFile])

cwd = os.getcwd()
layernr = 1
for line in positions:
	if line[0] == '%':
		continue
	line = line.rstrip()
	ypos = 1000 * float(line.split('\t')[0])
	filenr = int(line.split('\t')[2])
	fname = fstem + str(filenr).zfill(padding) + ext
	binname = binstem + str(filenr).zfill(padding) + binext
	pfname2 = configFile + 'Layer' + str(layernr) + "MultiRing.txt"
	call(['cp',configFile,pfname2])
	f2 = open(pfname2,"a")
	f2.write("Folder "+cwd+'\n')
	f2.write("RingToIndex "+ringtoindex+'\n')
	f2.write("LayerNr "+str(layernr)+'\n')
	call(['mkdir','-p',cwd+'/Layer'+str(layernr)])
	# Do FitTiltBCLsdSample for each ring
	ringradii = []
	for [i,ring] in enumerate(rings):
		pfname = configFile+'Layer'+str(layernr)+'ring'+str(ring)+'.txt'
		call(['cp',configFile,pfname])
		f = open(pfname,"a")
		f.write("LayerNr "+str(layernr)+'\n')
		f.write("Folder "+cwd+'/Ring'+str(ring)+'/\n')
		f2.write("RingNumbers "+str(ring)+'\n')
		f.write("RingNumbers "+str(ring)+'\n')
		f.write("RingToIndex "+str(ring)+'\n')
		f.write("LowerBoundThreshold " + str(threshs[i])+'\n')
		f.write("StartFileNr " + str(filenr)+'\n')
		f.write('BC '+str(bcall[0])+ ' ' + str(bcall[1]) + '\n')
		f.close()
		fldr = 'Ring'+str(ring)
		outfldr = fldr+'/PeakSearch/'+filestem+"_"+str(layernr)
		call(['mkdir','-p',outfldr])
		call(['cp','hkls.csv',fldr])
		call([binfolder+'/FitTiltBCLsdSample',pfname,fname])
		# Update the InputAllExtraInfoFittingAll.csv with the new BC
		AllF = open(outfldr + '/InputAllExtraInfoFittingAll.csv','r')
		allcontents = AllF.readlines()
		AllF.close()
		AllF = open(outfldr + '/InputAllExtraInfoFittingAll.csv','w')
		for line2 in allcontents:
			if line2[0] == '%':
				AllF.write(line2)
			else:
				line2sp = line2.split()
				y = float(line2sp[0])
				z = float(line2sp[1])
				ome = float(line2sp[2])
				grR = float(line2sp[3])
				ID = float(line2sp[4])
				RNr = float(line2sp[5])
				Eta = float(line2sp[6])
				Ttheta = float(line2sp[7])
				omeIniNoW = float(line2sp[8])
				yOrigNoW = float(line2sp[9])
				zOrigNoW = float(line2sp[10])
				yDet = float(line2sp[11])
				zDet = float(line2sp[12])
				omegaDet = float(line2sp[13])
				y = y - ypos
				Eta = CalcEtaAngle(y,z)
				Ttheta = 57.2957795130823*atan(sqrt(y*y+z*z)/Lsd)
				yOrigNoW = yOrigNoW - ypos
				outstr = '{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}\n'.format(y,z,ome,grR,ID,RNr,Eta,Ttheta,omeIniNoW,yOrigNoW,zOrigNoW,yDet,zDet,omegaDet)
				AllF.write(outstr)
		AllF.close()
		call(['rm',pfname])
		outpfn = 'Layer'+str(layernr)+'/paramstest_RingNr'+str(ring)+'.txt'
		call(['cp',outfldr+'/paramstest.txt',outpfn])
		paramscontents = open(outpfn).readlines()
		for paramsline in paramscontents:
			if "RingRadii" in paramsline:
				ringradii.append('RingNumbers '+str(ring)+'\n'+paramsline)
	## Do Merge Multiple rings
	f2.close()
	call([binfolder+'/MergeMultipleRings',pfname2])
	call(['cp','hkls.csv','InputAll.csv','InputAllExtraInfoFittingAll.csv','SpotsToIndex.csv','IDsHash.csv','IDRings.csv','Layer'+str(layernr)])
	# Generate correct paramstest.txt file first
	outfolder = outdir+'/Layer'+str(layernr)
	call(['mv',pfname2,outfolder])
	os.chdir(outfolder)
	fparams = open(outfolder+'/paramstest.txt','w')
	for paramsline in paramscontents:
		if 'RingRadii' not in paramsline:
			fparams.write(paramsline)
	for writeline in ringradii:
		fparams.write(writeline)
	fparams.write('NoSaveAll 1\n');
	fparams.close()
	# Make shm
	call([binfolder+'/SaveBinData'])
	# MapBND
	call([binfolder+'/MapBND',pfname2,fname,binname]) # Not needed right now if we don't have the bounding box data.
	os.chdir(outdir)
	layernr = layernr + 1

os.chdir(parentfolder)
call([binfolder+'/MergeMultipleScans',configFile])
