type file;

app (file o, file e) Medians (string pf, int layernr)
{
  runmedian pf layernr stdout=filename(o) stderr=filename(e);
}

app (file oI, file eI) Images (string paramf, int layern, int filenr, file sr)
{
  runimageprocessing paramf layern filenr stdout=filename(oI) stderr=filename(eI);
}


# Parameters to be modified ############

string paramfile = arg("paramfile","/data/tomo1/NFTest/ParametersGoldApril14.txt");
int NrLayers = toInt(arg("NrLayers","3"));
int NrFilesPerLayer = toInt(arg("NrFilesPerLayer","180"));

# End parameters #######################

foreach layer in [1:NrLayers] {
  file MedianOut<single_file_mapper; file=strcat("logs/Median_",layer,".out")>;
  file MedianErr<single_file_mapper; file=strcat("logs/Median_",layer,".err")>;
  (MedianOut, MedianErr) = Medians(paramfile,layer);
  foreach FileNr in [0:(NrFilesPerLayer-1)]{
    file ImageOut<single_file_mapper; file=strcat("logs/Images_",layer,FileNr,".out")>;
    file ImageErr<single_file_mapper; file=strcat("logs/Images_",layer,FileNr,".err")>;
    (ImageOut, ImageErr) = Images(paramfile, layer, FileNr, MedianOut);
  }
}
