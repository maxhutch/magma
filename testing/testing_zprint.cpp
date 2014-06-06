/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
  
       @precisions normal z -> c d s
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    magma_setdevice( 0 );

    magmaDoubleComplex *hA, *dA;

    /* Matrix size */
    magma_int_t m = 5;
    magma_int_t n = 10;
    //magma_int_t ione     = 1;
    //magma_int_t ISEED[4] = {0,0,0,1};
    //magma_int_t size;
    magma_int_t lda, ldda;
    
    lda    = ((m + 31)/32)*32;
    ldda   = ((m + 31)/32)*32;

    /* Allocate host memory for the matrix */
    TESTING_MALLOC_CPU( hA, magmaDoubleComplex, lda *n );
    TESTING_MALLOC_DEV( dA, magmaDoubleComplex, ldda*n );

    //size = lda*n;
    //lapackf77_zlarnv( &ione, ISEED, &size, hA );
    for( int j = 0; j < n; ++j ) {
        for( int i = 0; i < m; ++i ) {
            hA[i + j*lda] = MAGMA_Z_MAKE( i + j*0.01, 0. );
        }
    }
    magma_zsetmatrix( m, n, hA, lda, dA, ldda );
    
    printf( "A=" );
    magma_zprint( m, n, hA, lda );
    printf( "dA=" );
    magma_zprint_gpu( m, n, dA, ldda );
    
    //printf( "dA=" );
    //magma_zprint( m, n, dA, ldda );
    //printf( "A=" );
    //magma_zprint_gpu( m, n, hA, lda );
    
    /* Memory clean up */
    TESTING_FREE_CPU( hA );
    TESTING_FREE_DEV( dA );

    /* Shutdown */
    TESTING_FINALIZE();
}
