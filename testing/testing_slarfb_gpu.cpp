/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
       @generated from testing/testing_zlarfb_gpu.cpp normal z -> s, Mon May  2 23:31:15 2016
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <algorithm>  // std::swap

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing slarfb_gpu
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    // constants
    const float c_zero    = MAGMA_S_ZERO;
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;
    
    // local variables
    magma_int_t M, N, K, size, ldc, ldv, ldt, ldw, ldw2, nv;
    magma_int_t ISEED[4] = {0,0,0,1};
    float Cnorm, error, work[1];
    magma_int_t status = 0;
    
    // test all combinations of input parameters
    magma_side_t   side  [] = { MagmaLeft,       MagmaRight    };
    magma_trans_t  trans [] = { MagmaTrans, MagmaNoTrans  };
    magma_direct_t direct[] = { MagmaForward,    MagmaBackward };
    magma_storev_t storev[] = { MagmaColumnwise, MagmaRowwise  };

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%%   M     N     K   storev   side   direct   trans    ||R||_F / ||HC||_F\n");
    printf("%%=======================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      M = opts.msize[itest];
      N = opts.nsize[itest];
      K = opts.ksize[itest];
      if ( M < K || N < K || K <= 0 ) {
          printf( "%5d %5d %5d   skipping because slarfb requires M >= K, N >= K, K >= 0\n",
                  (int) M, (int) N, (int) K );
          continue;
      }
      for( int istor = 0; istor < 2; ++istor ) {
      for( int iside = 0; iside < 2; ++iside ) {
      for( int idir  = 0; idir  < 2; ++idir  ) {
      for( int itran = 0; itran < 2; ++itran ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            ldc = magma_roundup( M, opts.align );  // multiple of 32 by default
            ldt = magma_roundup( K, opts.align );  // multiple of 32 by default
            ldw = (side[iside] == MagmaLeft ? N : M);
            ldw2 = min( M, N );
            // (ldv, nv) get swapped later if rowwise
            ldv = (side[iside] == MagmaLeft ? M : N);
            nv  = K;
            
            // Allocate memory for matrices
            float *C, *R, *V, *T, *W;
            TESTING_MALLOC_CPU( C, float, ldc*N );
            TESTING_MALLOC_CPU( R, float, ldc*N );
            TESTING_MALLOC_CPU( V, float, ldv*K );
            TESTING_MALLOC_CPU( T, float, ldt*K );
            TESTING_MALLOC_CPU( W, float, ldw*K );
            
            magmaFloat_ptr dC, dV, dT, dW, dW2;
            TESTING_MALLOC_DEV( dC,  float, ldc*N );
            TESTING_MALLOC_DEV( dV,  float, ldv*K );
            TESTING_MALLOC_DEV( dT,  float, ldt*K );
            TESTING_MALLOC_DEV( dW,  float, ldw*K );
            TESTING_MALLOC_DEV( dW2, float, ldw2*K );
            
            // C is M x N.
            size = ldc*N;
            lapackf77_slarnv( &ione, ISEED, &size, C );
            //printf( "C=" );  magma_sprint( M, N, C, ldc );
            
            // V is ldv x nv. See larfb docs for description.
            // if column-wise and left,  M x K
            // if column-wise and right, N x K
            // if row-wise and left,     K x M
            // if row-wise and right,    K x N
            size = ldv*nv;
            lapackf77_slarnv( &ione, ISEED, &size, V );
            if ( storev[istor] == MagmaColumnwise ) {
                if ( direct[idir] == MagmaForward ) {
                    lapackf77_slaset( MagmaUpperStr, &K, &K, &c_zero, &c_one, V, &ldv );
                }
                else {
                    lapackf77_slaset( MagmaLowerStr, &K, &K, &c_zero, &c_one, &V[(ldv-K)], &ldv );
                }
            }
            else {
                // rowwise, swap V's dimensions
                std::swap( ldv, nv );
                if ( direct[idir] == MagmaForward ) {
                    lapackf77_slaset( MagmaLowerStr, &K, &K, &c_zero, &c_one, V, &ldv );
                }
                else {
                    lapackf77_slaset( MagmaUpperStr, &K, &K, &c_zero, &c_one, &V[(nv-K)*ldv], &ldv );
                }
            }
            //printf( "# ldv %d, nv %d\n", ldv, nv );
            //printf( "V=" );  magma_sprint( ldv, nv, V, ldv );
            
            // T is K x K, upper triangular for forward, and lower triangular for backward
            magma_int_t k1 = K-1;
            size = ldt*K;
            lapackf77_slarnv( &ione, ISEED, &size, T );
            if ( direct[idir] == MagmaForward ) {
                lapackf77_slaset( MagmaLowerStr, &k1, &k1, &c_zero, &c_zero, &T[1], &ldt );
            }
            else {
                lapackf77_slaset( MagmaUpperStr, &k1, &k1, &c_zero, &c_zero, &T[1*ldt], &ldt );
            }
            //printf( "T=" );  magma_sprint( K, K, T, ldt );
            
            magma_ssetmatrix( M,   N,  C, ldc, dC, ldc, opts.queue );
            magma_ssetmatrix( ldv, nv, V, ldv, dV, ldv, opts.queue );
            magma_ssetmatrix( K,   K,  T, ldt, dT, ldt, opts.queue );
            
            lapackf77_slarfb( lapack_side_const( side[iside] ), lapack_trans_const( trans[itran] ),
                              lapack_direct_const( direct[idir] ), lapack_storev_const( storev[istor] ),
                              &M, &N, &K,
                              V, &ldv, T, &ldt, C, &ldc, W, &ldw );
            //printf( "HC=" );  magma_sprint( M, N, C, ldc );
            
            if ( opts.version == 1 ) {
                magma_slarfb_gpu( side[iside], trans[itran], direct[idir], storev[istor],
                                  M, N, K,
                                  dV, ldv, dT, ldt, dC, ldc, dW, ldw, opts.queue );
            }
            else {
                magma_slarfb_gpu_gemm( side[iside], trans[itran], direct[idir], storev[istor],
                                       M, N, K,
                                       dV, ldv, dT, ldt, dC, ldc, dW, ldw, dW2, ldw2, opts.queue );
            }
            magma_sgetmatrix( M, N, dC, ldc, R, ldc, opts.queue );
            //printf( "dHC=" );  magma_sprint( M, N, R, ldc );
            
            // compute relative error |HC_magma - HC_lapack| / |HC_lapack|
            size = ldc*N;
            blasf77_saxpy( &size, &c_neg_one, C, &ione, R, &ione );
            Cnorm = lapackf77_slange( "Fro", &M, &N, C, &ldc, work );
            error = lapackf77_slange( "Fro", &M, &N, R, &ldc, work ) / Cnorm;
            
            printf( "%5d %5d %5d      %c       %c       %c       %c      %8.2e   %s\n",
                    (int) M, (int) N, (int) K,
                    lapacke_storev_const(storev[istor]), lapacke_side_const(side[iside]),
                    lapacke_direct_const(direct[idir]), lapacke_trans_const(trans[itran]),
                   error, (error < tol ? "ok" : "failed") );
            status += ! (error < tol);
            
            TESTING_FREE_CPU( C );
            TESTING_FREE_CPU( R );
            TESTING_FREE_CPU( V );
            TESTING_FREE_CPU( T );
            TESTING_FREE_CPU( W );
            
            TESTING_FREE_DEV( dC  );
            TESTING_FREE_DEV( dV  );
            TESTING_FREE_DEV( dT  );
            TESTING_FREE_DEV( dW  );
            TESTING_FREE_DEV( dW2 );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }}}}
      printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
