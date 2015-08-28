#!/bin/csh
#
# Run standard set of syev sizes & options.
#
# @author Mark Gates

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"

$NUMA ./testing_ssyevd     -JN $SIZES >>&! ssyevd.txt
$NUMA ./testing_ssyevd     -JV $SIZES >>&! ssyevd.txt
$NUMA ./testing_ssyevd_gpu -JN $SIZES >>&! ssyevd.txt
$NUMA ./testing_ssyevd_gpu -JV $SIZES >>&! ssyevd.txt

$NUMA ./testing_dsyevd     -JN $SIZES >>&! dsyevd.txt
$NUMA ./testing_dsyevd     -JV $SIZES >>&! dsyevd.txt
$NUMA ./testing_dsyevd_gpu -JN $SIZES >>&! dsyevd.txt
$NUMA ./testing_dsyevd_gpu -JV $SIZES >>&! dsyevd.txt

$NUMA ./testing_cheevd     -JN $SIZES >>&! cheevd.txt
$NUMA ./testing_cheevd     -JV $SIZES >>&! cheevd.txt
$NUMA ./testing_cheevd_gpu -JN $SIZES >>&! cheevd.txt
$NUMA ./testing_cheevd_gpu -JV $SIZES >>&! cheevd.txt

$NUMA ./testing_zheevd     -JN $SIZES >>&! zheevd.txt
$NUMA ./testing_zheevd     -JV $SIZES >>&! zheevd.txt
$NUMA ./testing_zheevd_gpu -JN $SIZES >>&! zheevd.txt
$NUMA ./testing_zheevd_gpu -JV $SIZES >>&! zheevd.txt
