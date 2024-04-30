import numpy as np
import blosc2
import zipfile
import matplotlib.pyplot as plt
archive = zipfile.ZipFile('/Users/hsharma/opt/MIDAS/FF_HEDM/Example/Au_FF_000001.ge3.zip','r')
imgMax = np.zeros((2048,2048))
for frameNr in range(1440):
    fn = f'{frameNr}.blosc'
    data = archive.read(fn)
    frame = np.frombuffer(memoryview(blosc2.decompress(bytearray(data))),dtype=np.uint16)
    # nS = np.sum(frame>0)
    imgMax += frame.reshape((2048,2048))
    # if (nS>0):
    #     print(nS)

plt.imshow(imgMax)
plt.show()