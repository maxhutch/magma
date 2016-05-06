/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
  
       @generated from testing/testing_zprint.cpp normal z -> s, Mon May  2 23:31:09 2016
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sprint
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    float *hA;
    magmaFloat_ptr dA;
    //magma_int_t ione     = 1;
    //magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t M, N, lda, ldda;  //size
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];
            lda   = M;
            ldda  = magma_roundup( M, opts.align );  // multiple of 32 by default
            //size  = lda*N;

            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( hA, float, lda *N );
            TESTING_MALLOC_DEV( dA, float, ldda*N );
        
            //lapackf77_slarnv( &ione, ISEED, &size, hA );
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    hA[i + j*lda] = MAGMA_S_MAKE( i + j*0.01, 0. );
                }
            }
            magma_ssetmatrix( M, N, hA, lda, dA, ldda, opts.queue );
            
            printf( "A=" );
            magma_sprint( M, N, hA, lda );
            
            printf( "dA=" );
            magma_sprint_gpu( M, N, dA, ldda );
            
            TESTING_FREE_CPU( hA );
            TESTING_FREE_DEV( dA );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
