#!/usr/bin/env python
import numpy as np
import zarr
import matplotlib.pyplot as plt
fn = '/Users/hsharma/Desktop/analysis/ORNL_Adaptive/reduced.mic.sim.zip'
zf = zarr.open(fn,'r')
plt.imshow(np.max(zf[:],axis=0))
plt.show()