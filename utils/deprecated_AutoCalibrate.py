#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from skimage import measure
import skimage
import matplotlib.patches as mpatches
plt.rcParams['figure.figsize'] = [10, 10]
import argparse
import sys

class MyParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

parser = MyParser(description='''Automated Calibration for WAXS using continuous rings-like signal''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dataFN', type=str, required=True, help='DataFileName')
parser.add_argument('-darkFN', type=str, required=True, help='DarkFileName')
parser.add_argument('-Wavelength', type=float, required=False, default=0.172979, help='Wavelength')
parser.add_argument('-MakePlots', type=int, required=False, default=0, help='MakePlots: to draw, use 1.')
parser.add_argument('-FirstRingNr', type=int, required=False, default=1, help='FirstRingNumber on data.')
parser.add_argument('-MultFactor', type=float, required=False, default=2.5, help='If set, any ring MultFactor times average would be thrown away.')
parser.add_argument('-StoppingStrain', type=float, required=False, default=0.00004, help='If refined pseudo-strain is below this value and all rings are "good", we would have converged.')
args, unparsed = parser.parse_known_args()
rawFN = args.dataFN
darkFN = args.darkFN
Wavelength = float(args.Wavelength)
DrawPlots = int(args.MakePlots)
firstRing = int(args.FirstRingNr)
multFactor = float(args.MultFactor)
needed_strain = float(args.StoppingStrain)

space_group = 225 # 225 for CeO2
latc = [5.4116,5.4116,5.4116,90,90,90] # CeO2
mrr = 2000000 # maximum radius to simulate rings. This, combined with initialLsd should give enough coverage.
px = 200 # Pixel size of the detector in microns
initialLsd = 1000000 # Only used for simulation, real Lsd can be anything.
header = 8192
NrPixelsY = 2048
NrPixelsZ = 2048
etaBinSize = 2 # Degrees
threshold = 500 # Threshold to use for some initial pre-processing. It will clean up the image and then apply the threshold. No dark correction is used here.
minArea = 300 # Minimum number of pixels that constitutes signal. For powder calibrants, 300 is typically used. Partial rings are okay, as long as they are at least 300 pixels big.

def fileReader(fn):
	size = os.path.getsize(fn)
	nFrames = int((size - header)/(sizePerFrame))
	average = np.zeros((NrPixelsY,NrPixelsZ))
	for i in range(nFrames):
		bytes_to_skip = i*sizePerFrame + header
		average += np.fromfile(fn,dtype=np.uint16,count=(NrPixelsY*NrPixelsZ),offset=bytes_to_skip).astype(float).reshape((NrPixelsY,NrPixelsZ))
	return average/nFrames

def redraw_figure():
	plt.draw()
	plt.pause(1)

def runMIDAS(fn):
	global ringsToExclude, folder, fstem, ext, ty_refined, tz_refined, p0_refined, p1_refined
	global p2_refined, p3_refined, darkName, fnumber, pad, wl, sg, lsd_refined, bc_refined, latc
	global nPlanes, mean_strain, std_strain
	with open(fn+'ps.txt','w') as pf:
		for ringNr in ringsToExclude:
			pf.write('RingsToExclude '+str(ringNr)+'\n')
		pf.write('Folder '+folder+'\n')
		pf.write('FileStem '+fstem+'\n')
		pf.write('Ext '+ext+'\n')
		pf.write('ImTransOpt 0\n')
		pf.write('Width 2000\n')
		pf.write('tolTilts 3\n')
		pf.write('tolBC 20\n')
		pf.write('tolLsd 15000\n')
		pf.write('tolP 2E-3\n')
		pf.write('tx 0\n')
		pf.write('ty '+ty_refined+'\n')
		pf.write('tz '+tz_refined+'\n')
		pf.write('Wedge 0\n')
		pf.write('p0 '+p0_refined+'\n')
		pf.write('p1 '+p1_refined+'\n')
		pf.write('p2 '+p2_refined+'\n')
		pf.write('p3 '+p3_refined+'\n')
		pf.write(f'EtaBinSize {etaBinSize}\n')
		pf.write('HeadSize '+str(header)+'\n')
		pf.write('Dark '+darkName+'\n')
		pf.write('StartNr '+str(fnumber)+'\n')
		pf.write('EndNr '+str(fnumber)+'\n')
		pf.write('Padding '+str(pad)+'\n')
		pf.write('NrPixelsY '+str(NrPixelsY)+'\n')
		pf.write('NrPixelsZ '+str(NrPixelsZ)+'\n')
		pf.write('px '+str(px)+'\n')
		pf.write('Wavelength '+str(Wavelength)+'\n')
		pf.write('SpaceGroup '+str(space_group)+'\n')
		pf.write('Lsd '+lsd_refined+'\n')
		pf.write('RhoD '+str(RhoDThis)+'\n')
		pf.write('BC '+bc_refined+'\n')
		pf.write('LatticeConstant '+str(latc[0])+' '+str(latc[1])+' '+str(latc[2])+' '+str(latc[3])+' '+str(latc[4])+' '+str(latc[5])+'\n')
	subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/CalibrantOMP")+' '+fn+"ps.txt 10",shell=True,stdout=open('calibrant_screen_out.csv','w'))
	output = open('calibrant_screen_out.csv').readlines()
	useful = 0
	for line in output:
		if 'Number of planes being considered' in line and nPlanes == 0:
			nPlanes = int(line.rstrip().split()[-1][:-1])
		if useful == 1:
			if 'Copy to par' in line:
				continue
			if 'Lsd ' in line:
				lsd_refined = line.split()[1]
			if 'BC ' in line:
				bc_refined = line.split()[1] + ' ' + line.split()[2]
			if 'ty ' in line:
				ty_refined = line.split()[1]
			if 'tz ' in line:
				tz_refined = line.split()[1]
			if 'p0 ' in line:
				p0_refined = line.split()[1]
			if 'p1 ' in line:
				p1_refined = line.split()[1]
			if 'p2 ' in line:
				p2_refined = line.split()[1]
			if 'p3 ' in line:
				p3_refined = line.split()[1]
			if 'MeanStrain ' in line:
				mean_strain = line.split()[1]
			if 'StdStrain ' in line:
				std_strain = line.split()[1]
		if 'Mean Values' in line:
			useful = 1
	results = np.genfromtxt(fn+'.corr.csv',skip_header=1)
	unique_tth = np.unique(results[:,-1])
	mean_strains_per_ring = np.zeros(len(unique_tth))
	for ringNr in range(len(unique_tth)):
		subarr = results[results[:,-1]==unique_tth[ringNr],:]
		mean_strains_per_ring[ringNr] = np.mean(subarr[:,1])
	ringsToExcludenew = np.argwhere(mean_strains_per_ring > multFactor*np.median(mean_strains_per_ring)) + 1
	rNew = []
	for ring in ringsToExcludenew:
		rNew.append(ring[0])
	if DrawPlots==1:
		plt.scatter(unique_tth,mean_strains_per_ring)
		plt.axhline(multFactor*np.median(mean_strains_per_ring),color='black')
		plt.xlabel('2theta [degrees]')
		plt.ylabel('Average strain')
		plt.show()
	if (len(rNew) == 0 and float(mean_strain) < needed_strain) and DrawPlots == 1:
		plt.scatter(results[:,-1],results[:,1])
		plt.scatter(unique_tth,mean_strains_per_ring,c='red')
		plt.axhline(multFactor*np.median(mean_strains_per_ring),color='black')
		plt.xlabel('2theta [degrees]')
		plt.ylabel('Computed strain')
		plt.title('Best fit results for '+fn)
		plt.show()
	return(rNew)


with open('ps.txt','w') as pf:
    pf.write('Wavelength '+str(Wavelength)+'\n')
    pf.write('SpaceGroup '+str(space_group)+'\n')
    pf.write('Lsd '+str(initialLsd)+'\n')
    pf.write('MaxRingRad '+str(mrr)+'\n')
    pf.write('LatticeConstant '+str(latc[0])+' '+str(latc[1])+' '+str(latc[2])+' '+str(latc[3])+' '+str(latc[4])+' '+str(latc[5])+'\n')

darkName = darkFN
subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+" ps.txt",shell=True,stdout=open('hkls_screen_out.csv','w'))
hkls = np.genfromtxt('hkls.csv',skip_header=1)
sim_rads = np.unique(hkls[:,-1])/px
sim_rad_ratios = sim_rads / sim_rads[0]

sizePerFrame = 2*NrPixelsY*NrPixelsZ
raw = fileReader(rawFN)
if DrawPlots==1:
	plt.imshow(raw,clim=[1800,2200],cmap='gray')
	plt.axis([0, NrPixelsY, 0, NrPixelsZ])
	plt.show()
dark = fileReader(darkFN)
neighborhood = skimage.morphology.disk(radius=50)
data = raw.astype(np.uint16)
data2 = skimage.filters.rank.median(data, neighborhood)
data2 = skimage.filters.rank.median(data2, neighborhood)
data2 = skimage.filters.rank.median(data2, neighborhood)
data2 = skimage.filters.rank.median(data2, neighborhood)
data2 = skimage.filters.rank.median(data2, neighborhood)
data = data.astype(float)
data_corr = data - data2
data_corr[data_corr<threshold] = 0
corr_img = data - data2
corr_img[corr_img<1] = 1
thresh = data_corr
thresh[thresh>0] = 255
print("Median background calculated. Now finding diffraction signal.")
labels,nlabels = measure.label(thresh,return_num=True)
props = measure.regionprops(labels)
bc = []
for label in range(1,nlabels):
	if np.sum(labels == label) < minArea:
		thresh[labels==label] = 0
	else:
		coords = props[label-1].coords
		bbox = props[label-1].bbox
		edge_coords = coords[coords[:,0]==bbox[0],:]
		edgecoorda = edge_coords[int(len(edge_coords)/2)]
		diffs = np.transpose(coords) - edgecoorda[:,None]
		arcLen = int(np.max(np.linalg.norm(diffs,axis=0)) / 2)
		edgecoordb = coords[np.argmax(np.linalg.norm(diffs,axis=0))]
		candidates = coords[np.abs(np.linalg.norm(diffs,axis=0)-arcLen)<2]
		if candidates.size==0: continue
		candidatea = candidates[int(candidates.shape[0]/2)]
		midpointa = (edgecoorda + candidatea)/2
		candidateb = candidatea
		midpointb = (edgecoordb + candidateb)/2
		x1 = edgecoorda[0]
		x2 = candidatea[0]
		x3 = candidateb[0]
		x4 = edgecoordb[0]
		x5 = midpointa[0]
		x6 = midpointb[0]
		y1 = edgecoorda[1]
		y2 = candidatea[1]
		y3 = candidateb[1]
		y4 = edgecoordb[1]
		y5 = midpointa[1]
		y6 = midpointb[1]
		m1 = (x1-x2)/(y2-y1)
		m2 = (x3-x4)/(y4-y3)
		x = (y6-y5+m1*x5-m2*x6)/(m1-m2)
		y = m1*(x-x5)+y5
		bc.append([x,y])

bc = np.array(bc)
bc_computed = np.array([np.median(bc[:,0]),np.median(bc[:,1])])

rads = []
nrads = 0
for label in range(1,nlabels):
	if np.sum(labels == label) > minArea:
		coords = props[label-1].coords
		rad = np.mean(np.linalg.norm(np.transpose(coords) - bc_computed[:,None],axis=0))
		toAdd = 1
		for radNr in range(nrads):
			if np.abs(rads[radNr]-rad) < 20:
				toAdd = 0
		if toAdd==1:
			rads.append(rad)
			nrads+=1

rads = np.sort(rads)
radRatios = rads/rads[0]
scaler = rads[0]/sim_rads[firstRing-1]
lsds = []
for i in range(rads.shape[0]):
    bestMatch = 10000
    bestRowNr = -1
    for j in range(firstRing-1,len(sim_rads)):
        if np.abs(1-(radRatios[i]/sim_rad_ratios[j])) < 0.02:
            if np.abs(1-(radRatios[i]/sim_rad_ratios[j])) < bestMatch:
                bestMatch = np.abs(1-(radRatios[i]/sim_rad_ratios[j]))
                bestRowNr = j
    if bestRowNr == -1: continue
    lsds.append(initialLsd*rads[i]/sim_rads[bestRowNr])

initialLsd = np.median(lsds)
bc_new = bc_computed
print("FN:",rawFN,"Beam Center guess: ",np.flip(bc_new),' Lsd guess: ',initialLsd)

with open('ps.txt','w') as pf:
    pf.write('Wavelength '+str(Wavelength)+'\n')
    pf.write('SpaceGroup '+str(space_group)+'\n')
    pf.write('Lsd '+str(initialLsd)+'\n')
    pf.write('MaxRingRad '+str(mrr)+'\n')
    pf.write('LatticeConstant '+str(latc[0])+' '+str(latc[1])+' '+str(latc[2])+' '+str(latc[3])+' '+str(latc[4])+' '+str(latc[5])+'\n')

subprocess.call(os.path.expanduser("~/opt/MIDAS/FF_HEDM/bin/GetHKLList")+" ps.txt",shell=True,stdout=open('hkls_screen_out.csv','w'))
hkls = np.genfromtxt('hkls.csv',skip_header=1)
sim_rads = np.unique(hkls[:,-1])/px
sim_rad_ratios = sim_rads / sim_rads[0]

if DrawPlots==1:
	fig,ax = plt.subplots()
	plt.imshow(raw,clim=[1800,2200],cmap='gray'); plt.colorbar(); 
	for rad in sim_rads:
		e1 = mpatches.Arc((bc_new[1],bc_new[0]),2*rad,2*rad,angle = 0,theta1=-180,theta2=180,color='blue')
		ax.add_patch(e1)
	ax.axis([0, NrPixelsY, 0, NrPixelsZ])
	ax.set_aspect('equal')
	plt.show()

fnumber = int(rawFN.split('_')[-1].split('.')[0])
pad = len(rawFN.split('_')[-1].split('.')[0])
fstem = os.path.basename('_'.join(rawFN.split('_')[:-1]))
ext = '.'+'.'.join(rawFN.split('_')[-1].split('.')[1:])
folder = os.path.dirname(rawFN)
if len(folder)==0:
    folder = os.getcwd()
lsd_refined = str(initialLsd)
bc_refined = str(bc_new[1])+ ' ' + str(bc_new[0])
# plt.scatter(bc_new[1],bc_new[0],c='k',marker='s')
# redraw_figure()
ty_refined = '0'
tz_refined = '0'
p0_refined = '0'
p1_refined = '0'
p2_refined = '0'
p3_refined = '0'
ringsToExclude = []
nPlanes = 0
iterNr = 1
print('Running MIDAS calibration, might take a few minutes.     Trial Nr: '+str(iterNr))
# Calculate maximum RhoD for the detector
edges = np.array([[0,0],[NrPixelsY,0],[NrPixelsY,NrPixelsZ],[0,NrPixelsZ]])
RhoDThis = np.max(np.linalg.norm(np.transpose(edges)-bc_new[:,None],axis=0))*px
rOrig = runMIDAS(rawFN)
iterNr += 1
ringsToExclude = rOrig
rNew = rOrig
ringListExcluded = np.zeros(nPlanes+1)
for i in ringsToExclude:
    ringListExcluded[i] = 1
while (len(rNew) > 0 or float(mean_strain) > needed_strain):
    print('Running MIDAS calibration again with updated parameters. Trial Nr: '+str(iterNr)+
        '\n\tPrevious strain: '+mean_strain+'. Number of new rings to exclude: '+str(len(rNew))+
        '. Number of rings to use: '+str(int(nPlanes-1-sum(ringListExcluded))))
    rNew = runMIDAS(rawFN)
    iterNr += 1
    currentRingNr = 0
    for i in range(1,nPlanes+1):
        if ringListExcluded[i] == 1:
            continue
        else:
            currentRingNr += 1
        if currentRingNr in rNew:
            ringListExcluded[i] = 1
            ringsToExclude.append(i)
print("Converged to a good set of parameters.\nBest values: ")
print('Lsd '+lsd_refined)
print('BC '+bc_refined)
print('ty '+ty_refined)
print('tz '+tz_refined)
print('p0 '+p0_refined)
print('p1 '+p1_refined)
print('p2 '+p2_refined)
print('p3 '+p3_refined)
print('Mean Strain: '+mean_strain)
print('Std Strain:  '+std_strain)
