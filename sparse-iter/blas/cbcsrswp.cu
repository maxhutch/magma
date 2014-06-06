/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zbcsrswp.cu normal z -> c, Fri May 30 10:41:35 2014

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif




/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======
    
    For a Block-CSR ILU factorization, this routine swaps rows in the vector *x
    according to the pivoting in *ipiv.
    
    Arguments
    =========

    magma_int_t r_blocks            number of blocks
    magma_int_t size_b              blocksize in BCSR
    magma_int_t *ipiv               array containing pivots
    magmaFloatComplex *x           input/output vector x

    ======================================================================    */

extern "C" magma_int_t
magma_cbcsrswp(   magma_int_t r_blocks,
                  magma_int_t size_b, 
                  magma_int_t *ipiv,
                  magmaFloatComplex *x ){


    const magma_int_t nrhs = 1, n = r_blocks*size_b, ione = 1, inc = 1;

    magmaFloatComplex *work; 
    magma_cmalloc_cpu( &work, r_blocks*size_b );

    // first shift the pivot elements
    for( magma_int_t k=0; k<r_blocks; k++){
            for( magma_int_t l=0; l<size_b; l++)
            ipiv[ k*size_b+l ] = ipiv[ k*size_b+l ] + k*size_b;
    }

    // now the usual pivoting
    magma_cgetmatrix(n, 1, x, n, work, n);
    lapackf77_claswp(&nrhs, work, &n, &ione, &n, ipiv, &inc);
    magma_csetmatrix(n, 1, work, n, x, n);

    magma_free_cpu(work);

    return MAGMA_SUCCESS;
}



