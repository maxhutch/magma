#ifndef ZFILL_H
#define ZFILL_H

#include "magma.h"

void zfill_matrix(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda );

void zfill_rhs(
    magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *X, magma_int_t ldx );

void zfill_matrix_gpu(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda );

void zfill_rhs_gpu(
    magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *dX, magma_int_t lddx );

#endif        //  #ifndef ZFILL_H
