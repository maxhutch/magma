#!/bin/csh
#
# Run standard set of 3 amigos (LU, Cholesky, QR) sizes & options.
#
# @author Mark Gates

set echo

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"
setenv SIZES_CPU "$SIZES"

# uncomment next line to get LAPACK results
#setenv SIZES_CPU "$SIZES --lapack"


# ----- LU
$NUMA ../testing/testing_sgetrf     $SIZES_CPU >>&! sgetrf.txt
$NUMA ../testing/testing_sgetrf_gpu $SIZES     >>&! sgetrf.txt

$NUMA ../testing/testing_dgetrf     $SIZES_CPU >>&! dgetrf.txt
$NUMA ../testing/testing_dgetrf_gpu $SIZES     >>&! dgetrf.txt

$NUMA ../testing/testing_cgetrf     $SIZES_CPU >>&! cgetrf.txt
$NUMA ../testing/testing_cgetrf_gpu $SIZES     >>&! cgetrf.txt

$NUMA ../testing/testing_zgetrf     $SIZES_CPU >>&! zgetrf.txt
$NUMA ../testing/testing_zgetrf_gpu $SIZES     >>&! zgetrf.txt


# ----- Cholesky
$NUMA ../testing/testing_spotrf     $SIZES_CPU >>&! spotrf.txt
$NUMA ../testing/testing_spotrf_gpu $SIZES     >>&! spotrf.txt

$NUMA ../testing/testing_dpotrf     $SIZES_CPU >>&! dpotrf.txt
$NUMA ../testing/testing_dpotrf_gpu $SIZES     >>&! dpotrf.txt

$NUMA ../testing/testing_cpotrf     $SIZES_CPU >>&! cpotrf.txt
$NUMA ../testing/testing_cpotrf_gpu $SIZES     >>&! cpotrf.txt

$NUMA ../testing/testing_zpotrf     $SIZES_CPU >>&! zpotrf.txt
$NUMA ../testing/testing_zpotrf_gpu $SIZES     >>&! zpotrf.txt


# ----- QR
$NUMA ../testing/testing_sgeqrf     $SIZES_CPU >>&! sgeqrf.txt
$NUMA ../testing/testing_sgeqrf_gpu $SIZES     >>&! sgeqrf.txt

$NUMA ../testing/testing_dgeqrf     $SIZES_CPU >>&! dgeqrf.txt
$NUMA ../testing/testing_dgeqrf_gpu $SIZES     >>&! dgeqrf.txt

$NUMA ../testing/testing_cgeqrf     $SIZES_CPU >>&! cgeqrf.txt
$NUMA ../testing/testing_cgeqrf_gpu $SIZES     >>&! cgeqrf.txt

$NUMA ../testing/testing_zgeqrf     $SIZES_CPU >>&! zgeqrf.txt
$NUMA ../testing/testing_zgeqrf_gpu $SIZES     >>&! zgeqrf.txt
