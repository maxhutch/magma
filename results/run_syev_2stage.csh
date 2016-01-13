#!/bin/csh
#
# Run standard set of syev (2-stage) sizes & options.
#
# @author Mark Gates

set echo

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"

# currently these testers don't run LAPACK except when checking results
# uncomment next line to get LAPACK results --- takes a VERY LONG time!
#setenv SIZES "$SIZES --lapack"

# 2 hours = 16 cores * 2 * 60 * 60
limit cputime 115200


$NUMA ../testing/testing_ssyevdx_2stage -JN $SIZES >>&! ssyevd_2stage.txt
$NUMA ../testing/testing_ssyevdx_2stage -JV $SIZES >>&! ssyevd_2stage.txt

$NUMA ../testing/testing_dsyevdx_2stage -JN $SIZES >>&! dsyevd_2stage.txt
$NUMA ../testing/testing_dsyevdx_2stage -JV $SIZES >>&! dsyevd_2stage.txt

$NUMA ../testing/testing_cheevdx_2stage -JN $SIZES >>&! cheevd_2stage.txt
$NUMA ../testing/testing_cheevdx_2stage -JV $SIZES >>&! cheevd_2stage.txt

$NUMA ../testing/testing_zheevdx_2stage -JN $SIZES >>&! zheevd_2stage.txt
$NUMA ../testing/testing_zheevdx_2stage -JV $SIZES >>&! zheevd_2stage.txt
