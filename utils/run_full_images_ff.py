import os
import glob
import numpy as np
from skimage import measure
import skimage
import matplotlib.pyplot as plt
from scipy import ndimage

dataFolder = '/data/tomo1/pilatus_midas/data/'
outFolder = '/data/tomo1/pilatus_midas/analysis/'
nrPxY = 1679
nrPxZ = 1475
bytesPerPx = 4
ext = '.raw'
nrPixels = nrPxY*nrPxZ
header = 8192 + bytesPerPx*nrPixels
nrFramesPerFile = 3000
startFileNrs = [15, 25, 40, 53]
padding = 6
nrFilesPerLayer = 6
thresh = 20
fileStems = ['ruby_pilatus_61keV','ruby_pilatus_71keV','ruby_pilatus_81keV','ruby_pilatus_91keV']

for nr,fstm in enumerate(fileStems):
	startFileNr = startFileNrs[nr]
	outfn = outFolder + fstm + '.csv'
	outf = open(outfn,'w')
	outf.write('FrameNr\tNrPixels\txCen\tyCen\tSumInt\n')
	for incr in range(nrFilesPerLayer):
		thisFileNr = startFileNr + incr
		fn = dataFolder + fstm + '_' + str(thisFileNr).zfill(padding) + ext
		f = open(fn)
		f.seek(header,0)
		for frameNr in range(nrFramesPerFile):
			data = np.fromfile(f,dtype=np.int32,count=(nrPixels))
			data = data.reshape((nrPxY,nrPxZ))
			data = np.flip(data,0)
			data[data < thresh] = 0
			lbl = ndimage.label(data)[0]
			useful_lb = []
			npx = []
			for lb in range(1,np.max(lbl)):
				nrPxThisLb = np.sum(lbl == lb)
				if nrPxThisLb > 1:
					useful_lb.append(lb)
					npx.append(nrPxThisLb)
			if len(useful_lb) > 0:
				COMs = ndimage.measurements.center_of_mass(data, lbl, useful_lb)
				sums = ndimage.measurements.sum(data,lbl,useful_lb)
				for ctr,COM in enumerate(COMs):
					outf.write(str(frameNr+incr*nrFramesPerFile)+'\t'+str(npx[ctr])+'\t' + str(COM[0]) + '\t' + str(COM[1]) + '\t' + str(sums[ctr]) + '\n')
					print([frameNr+incr*nrFramesPerFile,npx[ctr],COM[0],COM[1],sums[ctr]])
		f.close()
	outf.close()
