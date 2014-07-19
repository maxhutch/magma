/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zbcsrswp.cu normal z -> d, Fri Jul 18 17:34:27 2014

*/

#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif




/**
    Purpose
    -------
    
    For a Block-CSR ILU factorization, this routine swaps rows in the vector *x
    according to the pivoting in *ipiv.
    
    Arguments
    ---------

    @param
    r_blocks    magma_int_t
                number of blocks

    @param
    size_b      magma_int_t
                blocksize in BCSR

    @param
    ipiv        magma_int_t*
                array containing pivots

    @param
    x           double*
                input/output vector x


    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbcsrswp(   magma_int_t r_blocks,
                  magma_int_t size_b, 
                  magma_int_t *ipiv,
                  double *x ){


    const magma_int_t nrhs = 1, n = r_blocks*size_b, ione = 1, inc = 1;

    double *work; 
    magma_dmalloc_cpu( &work, r_blocks*size_b );

    // first shift the pivot elements
    for( magma_int_t k=0; k<r_blocks; k++){
            for( magma_int_t l=0; l<size_b; l++)
            ipiv[ k*size_b+l ] = ipiv[ k*size_b+l ] + k*size_b;
    }

    // now the usual pivoting
    magma_dgetmatrix(n, 1, x, n, work, n);
    lapackf77_dlaswp(&nrhs, work, &n, &ione, &n, ipiv, &inc);
    magma_dsetmatrix(n, 1, work, n, x, n);

    magma_free_cpu(work);

    return MAGMA_SUCCESS;
}



