# Modified code from https://gist.github.com/rsignell-usgs/49c214c9aaab4935e15a83bf3e228d03/revisions
# Adapted by Hemant Sharma to copy over all HDF5 datasets, no linking.

import numpy as np
import h5py
import zarr
# from zarr.meta import encode_fill_value
from numcodecs import Blosc #Default compression
import warnings
warnings.filterwarnings("ignore")

class Hdf5ToZarr:
    """Translate the content of one HDF5 file into Zarr metadata.
    HDF5 groups become Zarr groups. HDF5 datasets become Zarr arrays. Zarr array
    is copied over from the HDF5 file.
    Parameters
    ----------
    h5f : file-like or str
        Input HDF5 file as a string or file-like Python object.
    store : MutableMapping
        Zarr store.
    """

    def __init__(self, h5f, store):
        # Open HDF5 file in read mode...
        self._h5f = h5py.File(h5f, mode='r')

        # Create Zarr store's root group...
        self._zroot = zarr.group(store=store, overwrite=True)

    def translate(self):
        """Translate content of one HDF5 file into Zarr storage format.
        All data is copied out of the HDF5 file as Blosc compressed arrays.
        """
        self.transfer_attrs(self._h5f, self._zroot)
        self._h5f.visititems(self.translator)

    def transfer_attrs(self, h5obj, zobj):
        """Transfer attributes from an HDF5 object to its equivalent Zarr object.
        Parameters
        ----------
        h5obj : h5py.Group or h5py.Dataset
            An HDF5 group or dataset.
        zobj : zarr.hierarchy.Group or zarr.core.Array
            An equivalent Zarr group or array to the HDF5 group or dataset with
            attributes.
        """
        d1 = {}
        for n, v in h5obj.attrs.items():
            # Fix some attribute values to avoid JSON encoding exceptions...
            if isinstance(v, bytes):
                v = v.decode('utf-8')
            elif isinstance(v, (np.ndarray, np.number)):
                if v.size == 1:
                    v = v.flatten()[0].tolist()
                else:
                    v = v.tolist()
            d1[n] = v
        try:
            zobj.attrs.put(d1)
        except TypeError:
            print(f'Caught TypeError: {n}@{h5obj.name} = {v} ({type(v)})')

    def translator(self, name, h5obj):
        """Produce Zarr metadata for all groups and datasets in the HDF5 file.
        """
        if isinstance(h5obj, h5py.Dataset):
            compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
            # Create a Zarr array equivalent to this HDF5 dataset...
            if ('data' in h5obj.name):
                chunksize = (1,h5obj.shape[1],h5obj.shape[2])
            elif ('dark' in h5obj.name):
                chunksize = (1,h5obj.shape[1],h5obj.shape[2])
            elif ('bright' in h5obj.name):
                chunksize = (1,h5obj.shape[1],h5obj.shape[2])
            else:
                chunksize = h5obj.chunks

            za = self._zroot.create_dataset(h5obj.name, shape=h5obj.shape,
                                            dtype=h5obj.dtype,
                                            chunks=chunksize or False,
                                            fill_value=h5obj.fillvalue,
                                            compression=compressor,
                                            overwrite=True)
            self.transfer_attrs(h5obj, za)

            za[:] = h5obj[()]

        elif isinstance(h5obj, h5py.Group):
            zgrp = self._zroot.create_group(h5obj.name)
            self.transfer_attrs(h5obj, zgrp)