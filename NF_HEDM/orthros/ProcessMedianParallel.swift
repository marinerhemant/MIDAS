type file;

app Medians (string pf, int layernr, int rownr)
{
  runmedianparallel pf layernr rownr;
}

# Parameters to be modified ############

string paramfile = arg("paramfile","/data/tomo1/NFTest/ParametersGoldApril14.txt");
int NrLayers = toInt(arg("NrLayers","3"));
int NrFilesPerLayer = toInt(arg("NrFilesPerLayer","180"));
int NrPixels = toInt(arg("NrPixels","1"));

# End parameters #######################

foreach layer in [1:NrLayers] {
  foreach rownr in [0:(NrPixels-1)] {
    Medians(paramfile,layer,rownr);
  }
}
