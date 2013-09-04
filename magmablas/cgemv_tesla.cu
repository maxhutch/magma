/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions normal d

*/
#include "common_magma.h"

#define magmablas_cgemv_tesla magmablas_cgemv

extern "C" void
magmablas_cgemv_tesla(char trans, magma_int_t m, magma_int_t n, 
                      magmaFloatComplex alpha, const magmaFloatComplex *A, magma_int_t lda, 
                                            const magmaFloatComplex *x, magma_int_t incx, 
                      magmaFloatComplex beta,  magmaFloatComplex       *y, magma_int_t incy) 
{
    cublasCgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
