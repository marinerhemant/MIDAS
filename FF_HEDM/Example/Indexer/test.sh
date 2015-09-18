testbin=Data.bin
if [[ ! -f ${testbin} ]]
then
	tar -xvf bindata.tar.gz
fi

../../bin/Indexer paramstest.txt
