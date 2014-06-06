/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
  
       @generated c Tue Dec 17 13:18:56 2013
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
   -- Testing cgetrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    magma_setdevice( 0 );

    magmaFloatComplex *hA, *dA;

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
    TESTING_MALLOC_CPU( hA, magmaFloatComplex, lda *n );
    TESTING_MALLOC_DEV( dA, magmaFloatComplex, ldda*n );

    //size = lda*n;
    //lapackf77_clarnv( &ione, ISEED, &size, hA );
    for( int j = 0; j < n; ++j ) {
        for( int i = 0; i < m; ++i ) {
            hA[i + j*lda] = MAGMA_C_MAKE( i + j*0.01, 0. );
        }
    }
    magma_csetmatrix( m, n, hA, lda, dA, ldda );
    
    printf( "A=" );
    magma_cprint( m, n, hA, lda );
    printf( "dA=" );
    magma_cprint_gpu( m, n, dA, ldda );
    
    //printf( "dA=" );
    //magma_cprint( m, n, dA, ldda );
    //printf( "A=" );
    //magma_cprint_gpu( m, n, hA, lda );
    
    /* Memory clean up */
    TESTING_FREE_CPU( hA );
    TESTING_FREE_DEV( dA );

    /* Shutdown */
    TESTING_FINALIZE();
}
