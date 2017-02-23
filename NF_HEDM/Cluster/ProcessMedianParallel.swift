type file;

app Medians (string pf, int layernr)
{
  runmedianparallel pf layernr;
}

# Parameters to be modified ############

string paramfile = arg("paramfile","/data/tomo1/NFTest/ParametersGoldApril14.txt");
int NrLayers = toInt(arg("NrLayers","3"));
int NrFilesPerLayer = toInt(arg("NrFilesPerLayer","180"));

# End parameters #######################

foreach layer in [1:NrLayers] {
  Medians(paramfile,layer,rownr);
}
