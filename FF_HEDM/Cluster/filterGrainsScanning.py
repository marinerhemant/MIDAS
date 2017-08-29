import sys
from math import fabs
import time

if len(sys.argv) < 4:
	print 'Supply following parameters: Grains.csv, SpotMatrix.csv and IDsHash.csv'
	sys.exit(1)

t0 = time.time()
grainsFile = open(sys.argv[1])
grains = grainsFile.readlines()
grainsFile.close()
spotsFile = open(sys.argv[2])
spotinfo = spotsFile.readline()
uniquegrains = []
grainIDlist = []
spotsList = []
spotsPositions = {}
writearr = []
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
			spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
			spotinfo = spotsFile.readline()
			while (int(spotinfo.split()[0]) == int(line.split()[0])):
				spotsList[0].append(int(spotinfo.split()[1]))
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
					spotinfo = spotsFile.readline()
					grainFound = 1
					while (int(spotinfo.split()[0]) == int(line.split()[0])):
						spotsList[grainNr].append(int(spotinfo.split()[1]))
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
				spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
				spotinfo = spotsFile.readline()
				while (int(spotinfo.split()[0]) == int(line.split()[0])):
					spotsList[nGrains].append(int(spotinfo.split()[1]))
					spotsPositions[int(spotinfo.split()[1])] = [float(spotinfo.split()[2]),float(spotinfo.split()[3]),float(spotinfo.split()[4])]
					spotinfo = spotsFile.readline()
					if spotinfo == '':
						break
				spotsFile.seek(spotsFile.tell()-len(spotinfo))
				nGrains = nGrains + 1
spotsFile.close()

idsFile = open(sys.argv[3])
idsInfo = idsFile.readlines()

for grainNr in range(nGrains):
	print "Writing Grain " + str(grainNr) + ' of ' + str(nGrains) + ' grains.'
	splist = sorted(set(spotsList[grainNr]))
	f = open('SpotList.csv.' + str(grainNr),'w')
	for sp in splist:
		for line in idsInfo:
			if int(line.split()[2]) <= sp and int(line.split()[3]) >= sp:
				layernr = int(line.split()[0])
		f.write(str(sp)+'\t'+str(layernr)+'\t'+str(spotsPositions[sp][0])+'\t'+str(spotsPositions[sp][1])+'\t'+str(spotsPositions[sp][2])+'\n')
	f.close()
	f = open('GrainList.csv.'+str(grainNr),'w')
	for line in writearr[grainNr]:
		f.write(line)
	f.close()

t1 = time.time()
print "Time elapsed: " + str(t1-t0) + " s."
