#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

from PIL import Image
import numpy as np
import tifffile as tiff

startnr = 50333
nImages = 2160
nBatch = 4
fileStem = 'Ti7_NF_s2_state0/Ti7_NF_s2_state0'
outFileStem = 'Ti7_NF_s2_state0/Ti7_NF_s2_state0'
outNr = 1

for fileNr in range(startnr, startnr+nImages, nBatch):
	print 'InFNr: '+str(fileNr)+' OutFNr: '+str(outNr)
	fn = fileStem + '_' + str(fileNr).zfill(6)+'.tif'
	imarr = np.array(Image.open(fn),dtype=np.uint16)
	for incr in range(1,nBatch):
		fn = fileStem + '_' + str(fileNr+incr).zfill(6) + '.tif'
		imarr += np.array(Image.open(fn),dtype=np.uint16)
	tiff.imsave(outFileStem + '_' + str(outNr).zfill(6) + '.tif',imarr)
	outNr += 1
