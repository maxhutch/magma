#!/bin/csh
#
# Run standard set of syev sizes & options.
# Runs --lapack with CPU version only, to avoid duplicate tests.
# Runs BIG sizes separately so --lapack can be set for just smaller sizes.
#
# @author Mark Gates

set echo

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:10000:1000"
setenv BIG   "-N 123 -N 1234 --range 12000:20000:2000"

setenv SIZES_CPU "$SIZES"
setenv BIG_CPU   "$BIG"

# uncomment next line to get LAPACK results --- takes a VERY LONG time!
#setenv SIZES_CPU "$SIZES --lapack"
#setenv BIG_CPU   "$BIG   --lapack"

# 2 hours = 16 cores * 2 * 60 * 60
limit cputime 115200


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

# ----------
$NUMA ../testing/testing_ssyevd     -JN $BIG_CPU   >>&! ssyevd.txt
$NUMA ../testing/testing_ssyevd     -JV $BIG_CPU   >>&! ssyevd.txt
$NUMA ../testing/testing_ssyevd_gpu -JN $BIG       >>&! ssyevd.txt
$NUMA ../testing/testing_ssyevd_gpu -JV $BIG       >>&! ssyevd.txt

$NUMA ../testing/testing_dsyevd     -JN $BIG_CPU   >>&! dsyevd.txt
$NUMA ../testing/testing_dsyevd     -JV $BIG_CPU   >>&! dsyevd.txt
$NUMA ../testing/testing_dsyevd_gpu -JN $BIG       >>&! dsyevd.txt
$NUMA ../testing/testing_dsyevd_gpu -JV $BIG       >>&! dsyevd.txt

$NUMA ../testing/testing_cheevd     -JN $BIG_CPU   >>&! cheevd.txt
$NUMA ../testing/testing_cheevd     -JV $BIG_CPU   >>&! cheevd.txt
$NUMA ../testing/testing_cheevd_gpu -JN $BIG       >>&! cheevd.txt
$NUMA ../testing/testing_cheevd_gpu -JV $BIG       >>&! cheevd.txt

$NUMA ../testing/testing_zheevd     -JN $BIG_CPU   >>&! zheevd.txt
$NUMA ../testing/testing_zheevd     -JV $BIG_CPU   >>&! zheevd.txt
$NUMA ../testing/testing_zheevd_gpu -JN $BIG       >>&! zheevd.txt
$NUMA ../testing/testing_zheevd_gpu -JV $BIG       >>&! zheevd.txt
