type file;

app Images (string paramf, int layern, int filenr)
{
  runimageprocessingparallel paramf layern filenr;
}


# Parameters to be modified ############

string paramfile = arg("paramfile","/data/tomo1/NFTest/ParametersGoldApril14.txt");
int NrLayers = toInt(arg("NrLayers","3"));
int NrFilesPerLayer = toInt(arg("NrFilesPerLayer","180"));
int NrPixels = toInt(arg("NrPixels","1"));

# End parameters #######################

foreach layer in [1:NrLayers] {
  foreach FileNr in [0:(NrFilesPerLayer-1)]{
    Images(paramfile, layer, FileNr);
  }
}
