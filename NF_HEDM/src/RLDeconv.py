#
# Copyright (c) 2014, UChicago Argonne, LLC
# See LICENSE file.
#

import numpy as np
from skimage import restoration
from PIL import Image
import matplotlib.pyplot as plt
import time
import sys, os
import os.path

# Arguments: filename, nrIterations
inpFN = sys.argv[1]
psfFN = os.path.expanduser('~') + '/opt/MIDAS/gui/psf.tif'
print "Doing Deblur"
print 'Input file: ' + inpFN
print 'Peak Spread Function File: ' + psfFN
print 'NrIterations: ' + sys.argv[2]
imarr = np.fromfile(inpFN,dtype='uint16')
imarr = imarr.astype('float64')
imarr = imarr.reshape((2048,2048))
imarr2 = imarr.copy()
imarr2 += (np.random.poisson(lam=0.001,size=imarr.shape)-0.0005)/255.

psfim = Image.open(psfFN)
psfarr = np.array(psfim,dtype='float64')
t1 = time.time()
nrIterations = int(sys.argv[2])
deconv_im = restoration.richardson_lucy(imarr2,psfarr,iterations=nrIterations)
print 'Time elapsed: ' + str(time.time()-t1) + 's.'

deconv_im[deconv_im < 0.1] = 0
deconv_im[deconv_im >= 0.1] = 1
#deconv_im = np.array(deconv_im,dtype=bool)
totPxInt = np.count_nonzero(deconv_im)
print 'Total pixels with non-zero intensity: ' + str(totPxInt)
im_out = Image.fromarray(deconv_im)
im_out.save(inpFN+'.tif')
print 'Image written to ' + inpFN + '.tif'
