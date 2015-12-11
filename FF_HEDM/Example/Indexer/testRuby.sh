testbin=Ruby/Data.bin
if [[ ! -f ${testbin} ]]
then
	mkdir -p Ruby
	tar -xvf RubyData.tar.gz
fi

cd Ruby
../../../bin/Indexer paramstest.txt $1
