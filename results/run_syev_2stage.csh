#!/bin/csh
#
# Run standard set of syev (2-stage) sizes & options.
#
# @author Mark Gates

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"

$NUMA ./testing_ssyevdx_2stage -JN $SIZES >>&! ssyevd_2stage.txt
$NUMA ./testing_ssyevdx_2stage -JV $SIZES >>&! ssyevd_2stage.txt

$NUMA ./testing_dsyevdx_2stage -JN $SIZES >>&! dsyevd_2stage.txt
$NUMA ./testing_dsyevdx_2stage -JV $SIZES >>&! dsyevd_2stage.txt

$NUMA ./testing_cheevdx_2stage -JN $SIZES >>&! cheevd_2stage.txt
$NUMA ./testing_cheevdx_2stage -JV $SIZES >>&! cheevd_2stage.txt

$NUMA ./testing_zheevdx_2stage -JN $SIZES >>&! zheevd_2stage.txt
$NUMA ./testing_zheevdx_2stage -JV $SIZES >>&! zheevd_2stage.txt
