type file;

app (file op, file ep) runPeaks (string paramsfn, int fnr, int ringnr)
{
 peaks paramsfn fnr ringnr stdout=filename(op) stderr=filename(ep);
}

app (file oFO, file eFO) runProcessPeaks (string paramsf, int RNr, file DummyA[], file hkl)
{
 processPeaks paramsf RNr stdout=filename(oFO) stderr=filename(eFO);
}


# Parameters to be modified #############

int startnr = toInt(arg("startnr","1"));
int endnr = toInt(arg("endnr","600"));
int Ring = toInt(arg("ringnr","2"));
string parameterfilename = arg("paramsfile","/clhome/TOMO1/PeaksAnalysisHemant/PeaksFittingCode/90_33ParamsFile1.txt");

# End parameters ########################

file hkl <"hkls.csv">;

string PreFix1 = strcat("PeaksPerFile_",Ring);
file simAerr[]<simple_mapper;location="output",prefix=PreFix1,suffix=".err">;
file simAout[]<simple_mapper;location="output",prefix=PreFix1,suffix=".out">;
foreach i in [startnr:endnr] {
   (simAout[i],simAerr[i]) = runPeaks(parameterfilename,i,Ring);
}

file simBout<single_file_mapper; file=strcat("output/MergePeaks_",Ring,".out")>;
file simBerr<single_file_mapper; file=strcat("output/MergePeaks_",Ring,".err")>;
(simBout,simBerr) = runProcessPeaks(parameterfilename,Ring,simAerr,hkl);
