#!/bin/csh
#
# Run standard set of geev sizes & options.
# Runs BIG sizes separately so --lapack can be set for just smaller sizes.
#
# @author Mark Gates

set echo

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:10000:1000"
setenv BIG   "-N 123 -N 1234 --range 12000:20000:2000"

# uncomment next line to get LAPACK results --- takes a VERY LONG time!
#setenv SIZES "$SIZES --lapack"
#setenv BIG   "$BIG   --lapack"

# 2 hours = 16 cores * 2 * 60 * 60
limit cputime 115200


$NUMA ../testing/testing_sgeev -RN $SIZES >>&! sgeev.txt
$NUMA ../testing/testing_sgeev -RV $SIZES >>&! sgeev.txt

$NUMA ../testing/testing_dgeev -RN $SIZES >>&! dgeev.txt
$NUMA ../testing/testing_dgeev -RV $SIZES >>&! dgeev.txt

$NUMA ../testing/testing_cgeev -RN $SIZES >>&! cgeev.txt
$NUMA ../testing/testing_cgeev -RV $SIZES >>&! cgeev.txt

$NUMA ../testing/testing_zgeev -RN $SIZES >>&! zgeev.txt
$NUMA ../testing/testing_zgeev -RV $SIZES >>&! zgeev.txt

# ----------
$NUMA ../testing/testing_sgeev -RN $BIG   >>&! sgeev.txt
$NUMA ../testing/testing_sgeev -RV $BIG   >>&! sgeev.txt

$NUMA ../testing/testing_dgeev -RN $BIG   >>&! dgeev.txt
$NUMA ../testing/testing_dgeev -RV $BIG   >>&! dgeev.txt

$NUMA ../testing/testing_cgeev -RN $BIG   >>&! cgeev.txt
$NUMA ../testing/testing_cgeev -RV $BIG   >>&! cgeev.txt

$NUMA ../testing/testing_zgeev -RN $BIG   >>&! zgeev.txt
$NUMA ../testing/testing_zgeev -RV $BIG   >>&! zgeev.txt
