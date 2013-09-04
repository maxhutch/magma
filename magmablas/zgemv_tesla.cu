/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions normal z -> z
       
*/
#include "common_magma.h"

#define magmablas_zgemv_tesla magmablas_zgemv

extern "C" void
magmablas_zgemv_tesla(char trans, magma_int_t m, magma_int_t n, 
                      magmaDoubleComplex alpha, const magmaDoubleComplex *A, magma_int_t lda, 
                                             const magmaDoubleComplex *x, magma_int_t incx, 
                      magmaDoubleComplex beta,  magmaDoubleComplex       *y, magma_int_t incy) 
{
    cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
