#!/bin/csh
#
# Run standard set of symv sizes & options.
#
# @author Mark Gates

setenv NUMA  "./run numactl --interleave=all"

# denser than other tests
setenv SIZES "-N 123 -N 1234 --range 10:90:1 --range 100:900:10 --range 1000:9000:100 --range 10000:20000:2000"

$NUMA ./testing_ssymv -L $SIZES >>&! ssymv.txt
$NUMA ./testing_ssymv -U $SIZES >>&! ssymv.txt

$NUMA ./testing_dsymv -L $SIZES >>&! dsymv.txt
$NUMA ./testing_dsymv -U $SIZES >>&! dsymv.txt

$NUMA ./testing_chemv -L $SIZES >>&! chemv.txt
$NUMA ./testing_chemv -U $SIZES >>&! chemv.txt

$NUMA ./testing_zhemv -L $SIZES >>&! zhemv.txt
$NUMA ./testing_zhemv -U $SIZES >>&! zhemv.txt
