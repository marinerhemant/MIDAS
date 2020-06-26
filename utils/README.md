# MIDAS Utilities

Collection of utilites to pre-post-process HEDM data.

Probably need anaconda python to use most of these in case of dependencies.

**SpotMatrixToSpotsHDF.py**: Probably most useful code, convert FF output to HDF files.

**DL2FF.py**: Run analysis using Deep Learning outputs.

**GE2Tiff.py**: Convert GE detector files from Sector-1, APS to tiff files with individual frames.

**GFF2Grains.py**: Convert .gff output from Fable to Grains.csv used in MIDAS.

**NFGrainCentroids.py**: Code to convert individual grain reconstructions in NF to grain centroids.

**PlotFFNF.py**: Overlay and compute properties of grains from FF and NF HEDM.

**batchImages.py**: Batch add multiple images in NF data to improve reconstruction quality for heavily deformed samples.

**calcMiso.py**: Given two EulerAngles and the SpaceGroup,**GetMisorientationAngle**function will return the misorientation between the two orientations. Everything must be in**radians**.

**extractPeaks.py**: Extract peak windows from raw data using MIDAS FF output.

**hdf_gen_nf.py**: Code to convert nf experiment to HDF file following the DataExchange format.

**mergePeaks.py**: Merge peaks overlapping in omega. Will read the individual frame output from MIDAS.

**nf_paraview_gen.py**: Useful for converting NF output to hdf files, can be viewed directly in Paraview or imported into DREAM.3D

**psf.tif**: Peak spread function used in deblurring for NF.

**run_full_images_ff.py**: Find peak information for full images FF data. This is rather rudimentary, assuming no real peak spreads or overlaps.

**simulatePeaks.py**: Simulate artificial dataset. The peaks will be on the right 2thetas, but the rest is arbitrary. Saves individual tiffs.

**vtkSimExportBin.py**: Code to read in the .vtk files from CPFEM simulations from Purdue group and compute properties and write out hdf files.
