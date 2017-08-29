import sys
from math import fabs
import time

t0 = time.time()
grainsFile = open(sys.argv[1])
grains = grainsFile.readlines()
grainsFile.close()
spotsFile = open(sys.argv[2])
spotinfo = spotsFile.readline()
uniquegrains = []
grainIDlist = []
spotsList = []
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
			spotinfo = spotsFile.readline()
			spotsList.append([int(spotinfo.split()[1])])
			spotinfo = spotsFile.readline()
			while (int(spotinfo.split()[0]) == int(line.split()[0])):
				spotsList[0].append(int(spotinfo.split()[1]))
				spotinfo = spotsFile.readline()
				if spotinfo == '':
					break
			spotsFile.seek(spotsFile.tell()-len(spotinfo))
			nGrains = 1
		else:
			grainFound = 0
			for grainNr in range(nGrains):
				eG1 = uniquegrains[grainNr][0]
				if (fabs(eG1-e1) < 10):
					grainIDlist[grainNr].append(int(line.split()[0]))
					spotinfo = spotsFile.readline()
					grainFound = 1
					while (int(spotinfo.split()[0]) == int(line.split()[0])):
						spotsList[grainNr].append(int(spotinfo.split()[1]))
						spotinfo = spotsFile.readline()
						if spotinfo == '':
							break
					spotsFile.seek(spotsFile.tell()-len(spotinfo))
			if grainFound == 0:
				uniquegrains.append([e1,e2,e3])
				grainIDlist.append([int(line.split()[0])])
				nGrains = nGrains + 1
				spotinfo = spotsFile.readline()
				spotsList.append([int(spotinfo.split()[1])])
				spotinfo = spotsFile.readline()
				while (int(spotinfo.split()[0]) == int(line.split()[0])):
					spotsList[nGrains-1].append(int(spotinfo.split()[1]))
					spotinfo = spotsFile.readline()
					if spotinfo == '':
						break
				spotsFile.seek(spotsFile.tell()-len(spotinfo))
spotsFile.close()
for grainNr in range(nGrains):
	splist = set(spotsList[grainNr])
	f = open('SpotList.csv.' + str(grainNr),'w')
	for sp in splist:
		f.write(str(sp)+'\n')
	f.close()
	f = open('GrainList.csv'+str(grainNr),'w')
	for ID in grainIDlist[grainNr]:
		f.write(str(ID)+'\n')
	f.close()

t1 = time.time()
print "Time elapsed: " + str(t1-t0) + " s."
