#!/bin/csh
#
# Run standard set of SVD sizes & options.
# Runs BIG sizes separately so --lapack can be set for just smaller sizes.
#
# @author Mark Gates

set echo

setenv NUMA  "./run numactl --interleave=all"

# square
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:10000:1000"
setenv BIG   "-N 123 -N 1234 --range 12000:20000:2000"

# 3:1 ratio
setenv SIZES "$SIZES -N 300,100 -N 600,200 -N 900,300 -N 1200,400 -N 1500,500 -N 1800,600 -N 2100,700 -N 2400,800 -N 2700,900"
setenv SIZES "$SIZES -N 3000,1000 -N 6000,2000 -N 9000,3000 -N 12000,4000 -N 15000,5000 -N 18000,6000 -N 21000,7000 -N 24000,8000 -N 27000,9000"

# 1:3 ratio
setenv SIZES "$SIZES -N 100,300 -N 200,600 -N 300,900 -N 400,1200 -N 500,1500 -N 600,1800 -N 700,2100 -N 800,2400 -N 900,2700"
setenv SIZES "$SIZES -N 1000,3000 -N 2000,6000 -N 3000,9000 -N 4000,12000 -N 5000,15000 -N 6000,18000 -N 7000,21000 -N 8000,24000 -N 9000,27000"

# 100:1 ratio
setenv SIZES "$SIZES -N 10000,100 -N 20000,200 -N 30000,300 -N 40000,400 -N 50000,500 -N 60000,600 -N 70000,700 -N 80000,800 -N 90000,900"
setenv SIZES "$SIZES -N 100000,1000 -N 200000,2000"

# 1:100 ratio
setenv SIZES "$SIZES -N 100,10000 -N 200,20000 -N 300,30000 -N 400,40000 -N 500,50000 -N 600,60000 -N 700,70000 -N 800,80000 -N 900,90000"
setenv SIZES "$SIZES -N 1000,100000 -N 2000,200000"

# uncomment next line to get LAPACK results --- takes a VERY LONG time!
#setenv SIZES "$SIZES --lapack"
#setenv BIG   "$SIZES --lapack"

# 2 hours = 16 cores * 2 * 60 * 60
limit cputime 115200


$NUMA ../testing/testing_sgesvd -UN -VN $SIZES >>&! sgesvd.txt
$NUMA ../testing/testing_sgesvd -US -VS $SIZES >>&! sgesvd.txt
$NUMA ../testing/testing_sgesdd -UN -VN $SIZES >>&! sgesvd.txt
$NUMA ../testing/testing_sgesdd -US -VS $SIZES >>&! sgesvd.txt

$NUMA ../testing/testing_dgesvd -UN -VN $SIZES >>&! dgesvd.txt
$NUMA ../testing/testing_dgesvd -US -VS $SIZES >>&! dgesvd.txt
$NUMA ../testing/testing_dgesdd -UN -VN $SIZES >>&! dgesvd.txt
$NUMA ../testing/testing_dgesdd -US -VS $SIZES >>&! dgesvd.txt

$NUMA ../testing/testing_cgesvd -UN -VN $SIZES >>&! cgesvd.txt
$NUMA ../testing/testing_cgesvd -US -VS $SIZES >>&! cgesvd.txt
$NUMA ../testing/testing_cgesdd -UN -VN $SIZES >>&! cgesvd.txt
$NUMA ../testing/testing_cgesdd -US -VS $SIZES >>&! cgesvd.txt

$NUMA ../testing/testing_zgesvd -UN -VN $SIZES >>&! zgesvd.txt
$NUMA ../testing/testing_zgesvd -US -VS $SIZES >>&! zgesvd.txt
$NUMA ../testing/testing_zgesdd -UN -VN $SIZES >>&! zgesvd.txt
$NUMA ../testing/testing_zgesdd -US -VS $SIZES >>&! zgesvd.txt

# ----------
$NUMA ../testing/testing_sgesvd -UN -VN $BIG   >>&! sgesvd.txt
$NUMA ../testing/testing_sgesvd -US -VS $BIG   >>&! sgesvd.txt
$NUMA ../testing/testing_sgesdd -UN -VN $BIG   >>&! sgesvd.txt
$NUMA ../testing/testing_sgesdd -US -VS $BIG   >>&! sgesvd.txt

$NUMA ../testing/testing_dgesvd -UN -VN $BIG   >>&! dgesvd.txt
$NUMA ../testing/testing_dgesvd -US -VS $BIG   >>&! dgesvd.txt
$NUMA ../testing/testing_dgesdd -UN -VN $BIG   >>&! dgesvd.txt
$NUMA ../testing/testing_dgesdd -US -VS $BIG   >>&! dgesvd.txt

$NUMA ../testing/testing_cgesvd -UN -VN $BIG   >>&! cgesvd.txt
$NUMA ../testing/testing_cgesvd -US -VS $BIG   >>&! cgesvd.txt
$NUMA ../testing/testing_cgesdd -UN -VN $BIG   >>&! cgesvd.txt
$NUMA ../testing/testing_cgesdd -US -VS $BIG   >>&! cgesvd.txt

$NUMA ../testing/testing_zgesvd -UN -VN $BIG   >>&! zgesvd.txt
$NUMA ../testing/testing_zgesvd -US -VS $BIG   >>&! zgesvd.txt
$NUMA ../testing/testing_zgesdd -UN -VN $BIG   >>&! zgesvd.txt
$NUMA ../testing/testing_zgesdd -US -VS $BIG   >>&! zgesvd.txt
