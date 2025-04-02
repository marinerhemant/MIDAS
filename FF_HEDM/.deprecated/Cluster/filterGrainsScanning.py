import sys
from math import fabs
import time
import os

if len(sys.argv) < 2:
	print 'Supply following parameters: PS.txt'
	sys.exit(1)

configFile = sys.argv[1]
pscontent = open(configFile).readlines()
for line in pscontent:
	line = [s for s in line.rstrip().split() if s]
	if len(line) > 0:
		if line[0] == 'PositionsFile':
			positionsFile = os.getcwd() + '/' + line[1]
		elif line[0] == 'OutDirPath':
			outdir = os.getcwd() + '/' + line[1]

t0 = time.time()
grainsFile = open('Grains.csv')
grains = grainsFile.readlines()
grainsFile.close()
spotsFile = open('SpotMatrix.csv')
spotinfo = spotsFile.readline()
uniquegrains = []
grainIDlist = []
spotsList = []
spotsPositions = {}
writearr = []
writearr2 = []
nSpots = []
for line in grains:
	if line[0] == '%' :
		continue
	else:
		e1 = float(line.split()[-3])
		e2 = float(line.split()[-2])
		e3 = float(line.split()[-1])
		if (len(uniquegrains) == 0):
			uniquegrains.append([e1,e2,e3])
			grainIDlist.append([int(line.split()[0])])
			writearr.append([line])
			spotinfo = spotsFile.readline()
			spotsList.append([int(spotinfo.split()[1])])
			writearr2.append([int(spotinfo.split()[1])])
			nSpots.append([1])
			spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
			spotinfo = spotsFile.readline()
			while (int(spotinfo.split()[0]) == int(line.split()[0])):
				spotsList[0].append(int(spotinfo.split()[1]))
				writearr2[0].append(int(spotinfo.split()[1]))
				nSpots[0][0] = nSpots[0][0] + 1
				spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
				spotinfo = spotsFile.readline()
				if spotinfo == '':
					break
			spotsFile.seek(spotsFile.tell()-len(spotinfo))
			nGrains = 1
		else:
			grainFound = 0
			for grainNr in range(nGrains):
				eG1 = uniquegrains[grainNr][0]
				if (fabs(eG1-e1) < 10): ## 10 degrees tolerance for first euler angle. This is good enough for now.
					grainIDlist[grainNr].append(int(line.split()[0]))
					writearr[grainNr].append(line)
					nSpots[grainNr].append(0)
					spotinfo = spotsFile.readline()
					grainFound = 1
					while (int(spotinfo.split()[0]) == int(line.split()[0])):
						spotsList[grainNr].append(int(spotinfo.split()[1]))
						writearr2[grainNr].append(int(spotinfo.split()[1]))
						nSpots[grainNr][-1] = nSpots[grainNr][-1] + 1
						spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
						spotinfo = spotsFile.readline()
						if spotinfo == '':
							break
					spotsFile.seek(spotsFile.tell()-len(spotinfo))
			if grainFound == 0:
				uniquegrains.append([e1,e2,e3])
				grainIDlist.append([int(line.split()[0])])
				writearr.append([line])
				spotinfo = spotsFile.readline()
				spotsList.append([int(spotinfo.split()[1])])
				writearr2.append([int(spotinfo.split()[1])])
				nSpots.append([1])
				spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
				spotinfo = spotsFile.readline()
				while (int(spotinfo.split()[0]) == int(line.split()[0])):
					spotsList[nGrains].append(int(spotinfo.split()[1]))
					writearr2[nGrains].append(int(spotinfo.split()[1]))
					nSpots[nGrains][0] = nSpots[nGrains][0] + 1
					spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
					spotinfo = spotsFile.readline()
					if spotinfo == '':
						break
				spotsFile.seek(spotsFile.tell()-len(spotinfo))
				nGrains = nGrains + 1
spotsFile.close()

idsFile = open(outdir+'/IDsHash.csv')
idsInfo = idsFile.readlines()

for grainNr in range(nGrains):
	print "Writing Grain " + str(grainNr) + ' of ' + str(nGrains) + ' grains.'
	f = open('SpotMatch.csv.'+str(grainNr),'w')
	for line in writearr2[grainNr]:
		f.write(str(line)+'\n')
	f.close()
	splist = sorted(set(spotsList[grainNr]))
	f = open('SpotList.csv.' + str(grainNr),'w')
	for sp in splist:
		for line in idsInfo:
			if int(line.split()[2]) <= sp and int(line.split()[3]) >= sp:
				layernr = int(line.split()[0])
				startnr = int(line.split()[4])
				ringnr = int(line.split()[1])
				break
		f2 = open(outdir+'/'+'Layer'+str(layernr)+'/IDRings.csv')
		lines = f2.readlines()
		origID = lines[sp-startnr].split()[1]
		newRingNr = int(lines[sp-startnr].split()[0])
		f.write(origID+'\t'+str(sp)+'\t'+str(layernr)+'\t'+str(spotsPositions[sp][0])+'\t'+str(spotsPositions[sp][1])+'\t'+str(spotsPositions[sp][2])+'\n')
	f.close()
	f = open('GrainList.csv.'+str(grainNr),'w')
	for line in writearr[grainNr]:
		f.write(line)
	f.close()
	mapFile = open('mapFile.csv.'+str(grainNr),'w')
	mapFile.write(str(len(writearr[grainNr]))+'\t'+str(len(splist))+'\t'+str(sum(nSpots[grainNr]))+'\n')
	for nrSpots in nSpots[grainNr]:
		mapFile.write(str(nrSpots)+'\n')
	mapFile.close()

t1 = time.time()
print "Time elapsed: " + str(t1-t0) + " s."
