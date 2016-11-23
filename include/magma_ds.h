/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magma_zc.h, mixed zc -> ds, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMA_DS_H
#define MAGMA_DS_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// MAGMA mixed precision function definitions
//
// In alphabetical order of base name (ignoring precision).
magma_int_t
magma_dsgeqrsv_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_dsgesv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magmaInt_ptr dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd,
    magmaFloat_ptr  dworks,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_dsgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr  dA, magma_int_t ldda,
    magmaInt_ptr        dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX,
    magma_int_t *info);

// CUDA MAGMA only
magma_int_t
magma_dssysv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd,
    magmaFloat_ptr  dworks,
    magma_int_t *iter,
    magma_int_t *info);

magma_int_t
magma_dsposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaDouble_ptr dworkd,
    magmaFloat_ptr  dworks,
    magma_int_t *iter,
    magma_int_t *info);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_DS_H */
