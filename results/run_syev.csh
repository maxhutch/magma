#!/bin/csh
#
# Run standard set of syev sizes & options.
#
# @author Mark Gates

set echo

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"
setenv SIZES_CPU "$SIZES"

# uncomment next line to get LAPACK results --- takes a VERY LONG time!
#setenv SIZES_CPU "$SIZES --lapack"


$NUMA ../testing/testing_ssyevd     -JN $SIZES_CPU >>&! ssyevd.txt
$NUMA ../testing/testing_ssyevd     -JV $SIZES_CPU >>&! ssyevd.txt
$NUMA ../testing/testing_ssyevd_gpu -JN $SIZES     >>&! ssyevd.txt
$NUMA ../testing/testing_ssyevd_gpu -JV $SIZES     >>&! ssyevd.txt

$NUMA ../testing/testing_dsyevd     -JN $SIZES_CPU >>&! dsyevd.txt
$NUMA ../testing/testing_dsyevd     -JV $SIZES_CPU >>&! dsyevd.txt
$NUMA ../testing/testing_dsyevd_gpu -JN $SIZES     >>&! dsyevd.txt
$NUMA ../testing/testing_dsyevd_gpu -JV $SIZES     >>&! dsyevd.txt

$NUMA ../testing/testing_cheevd     -JN $SIZES_CPU >>&! cheevd.txt
$NUMA ../testing/testing_cheevd     -JV $SIZES_CPU >>&! cheevd.txt
$NUMA ../testing/testing_cheevd_gpu -JN $SIZES     >>&! cheevd.txt
$NUMA ../testing/testing_cheevd_gpu -JV $SIZES     >>&! cheevd.txt

$NUMA ../testing/testing_zheevd     -JN $SIZES_CPU >>&! zheevd.txt
$NUMA ../testing/testing_zheevd     -JV $SIZES_CPU >>&! zheevd.txt
$NUMA ../testing/testing_zheevd_gpu -JN $SIZES     >>&! zheevd.txt
$NUMA ../testing/testing_zheevd_gpu -JV $SIZES     >>&! zheevd.txt
