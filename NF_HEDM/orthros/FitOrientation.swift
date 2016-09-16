type file;

app runfitorientation (string pf, int nr, string micfn)
{
  fitorientation pf nr micfn;
}

# Parameters to be modified ############

string paramfile = arg("paramfile","/clhome/TOMO1/PeaksAnalysisHemant/NF_HEDM/ParametersSampleApril14.txt");
int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","100"));
string micfn = arg("micfn","microstructure.mic");

# End parameters #######################

tracef("%s: %i\n","StartNr is",startnr);
tracef("%s: %i\n","EndNr is",endnr);

foreach i in [startnr:endnr] {
    runfitorientation(paramfile,i,micfn);
}
