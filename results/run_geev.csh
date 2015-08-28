#!/bin/csh
#
# Run standard set of geev sizes & options.
#
# @author Mark Gates

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"

$NUMA ./testing_sgeev -RN $SIZES >>&! sgeev.txt
$NUMA ./testing_sgeev -RV $SIZES >>&! sgeev.txt

$NUMA ./testing_dgeev -RN $SIZES >>&! dgeev.txt
$NUMA ./testing_dgeev -RV $SIZES >>&! dgeev.txt

$NUMA ./testing_cgeev -RN $SIZES >>&! cgeev.txt
$NUMA ./testing_cgeev -RV $SIZES >>&! cgeev.txt

$NUMA ./testing_zgeev -RN $SIZES >>&! zgeev.txt
$NUMA ./testing_zgeev -RV $SIZES >>&! zgeev.txt
