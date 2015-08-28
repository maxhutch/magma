#!/bin/csh
#
# Run standard set of 3 amigos (LU, Cholesky, QR) sizes & options.
#
# @author Mark Gates

setenv NUMA  "./run numactl --interleave=all"
setenv SIZES "-N 123 -N 1234 --range 10:90:10 --range 100:900:100 --range 1000:9000:1000 --range 10000:20000:2000"
setenv SIZES "$SIZES --lapack"

# ----- LU
$NUMA ./testing_sgetrf     $SIZES >>&! sgetrf.txt
$NUMA ./testing_sgetrf_gpu $SIZES >>&! sgetrf.txt

$NUMA ./testing_dgetrf     $SIZES >>&! dgetrf.txt
$NUMA ./testing_dgetrf_gpu $SIZES >>&! dgetrf.txt

$NUMA ./testing_cgetrf     $SIZES >>&! cgetrf.txt
$NUMA ./testing_cgetrf_gpu $SIZES >>&! cgetrf.txt

$NUMA ./testing_zgetrf     $SIZES >>&! zgetrf.txt
$NUMA ./testing_zgetrf_gpu $SIZES >>&! zgetrf.txt


# ----- Cholesky
$NUMA ./testing_spotrf     $SIZES >>&! spotrf.txt
$NUMA ./testing_spotrf_gpu $SIZES >>&! spotrf.txt

$NUMA ./testing_dpotrf     $SIZES >>&! dpotrf.txt
$NUMA ./testing_dpotrf_gpu $SIZES >>&! dpotrf.txt

$NUMA ./testing_cpotrf     $SIZES >>&! cpotrf.txt
$NUMA ./testing_cpotrf_gpu $SIZES >>&! cpotrf.txt

$NUMA ./testing_zpotrf     $SIZES >>&! zpotrf.txt
$NUMA ./testing_zpotrf_gpu $SIZES >>&! zpotrf.txt


# ----- QR
$NUMA ./testing_sgeqrf     $SIZES >>&! sgeqrf.txt
$NUMA ./testing_sgeqrf_gpu $SIZES >>&! sgeqrf.txt

$NUMA ./testing_dgeqrf     $SIZES >>&! dgeqrf.txt
$NUMA ./testing_dgeqrf_gpu $SIZES >>&! dgeqrf.txt

$NUMA ./testing_cgeqrf     $SIZES >>&! cgeqrf.txt
$NUMA ./testing_cgeqrf_gpu $SIZES >>&! cgeqrf.txt

$NUMA ./testing_zgeqrf     $SIZES >>&! zgeqrf.txt
$NUMA ./testing_zgeqrf_gpu $SIZES >>&! zgeqrf.txt
