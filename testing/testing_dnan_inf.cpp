/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
  
       @generated from testing/testing_znan_inf.cpp normal z -> d, Mon May  2 23:31:09 2016
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

#define REAL

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing znan_inf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    #define hA(i,j) (hA + (i) + (j)*lda)
    
    double *hA;
    magmaDouble_ptr dA;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t M, N, lda, ldda, size;
    magma_int_t *ii, *jj;
    magma_int_t i, j, cnt, tmp;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    
    /* ====================================================================
       Check scalar operations
       =================================================================== */
    // here "a" denotes finite scalar, "nan" and "inf" denote exceptions.
    // before underbar "_" is real, after underbar "_" is imag.
    printf( "%% checking magma_d_isnan, magma_d_isinf, magma_d_isnan_inf\n" );
    double a_a     = MAGMA_D_MAKE( 1.2345,      4.3456      );
                               
    double a_nan   = MAGMA_D_MAKE( 1.2345,      MAGMA_D_NAN );
    double a_inf   = MAGMA_D_MAKE( 1.2345,      MAGMA_D_INF );
                               
    double nan_a   = MAGMA_D_MAKE( MAGMA_D_NAN, 4.3456      );
    double inf_a   = MAGMA_D_MAKE( MAGMA_D_INF, 4.3456      );
    
    double nan_nan = MAGMA_D_MAKE( MAGMA_D_NAN, MAGMA_D_NAN );
    double nan_inf = MAGMA_D_MAKE( MAGMA_D_NAN, MAGMA_D_INF );
    
    double inf_inf = MAGMA_D_MAKE( MAGMA_D_INF, MAGMA_D_INF );
    double inf_nan = MAGMA_D_MAKE( MAGMA_D_INF, MAGMA_D_NAN );
    
    // ----- isnan
    magma_assert_warn( ! isnan( MAGMA_D_REAL(a_a)   ), "! isnan( real(a_a)   )" );
    magma_assert_warn(   isnan( MAGMA_D_REAL(nan_a) ), "  isnan( real(nan_a) )" );
    magma_assert_warn( ! isnan( MAGMA_D_REAL(inf_a) ), "! isnan( real(inf_a) )" );
    
    // ----- isinf
    magma_assert_warn( ! isinf( MAGMA_D_REAL(a_a)   ), "! isinf( real(a_a)   )" );
    magma_assert_warn( ! isinf( MAGMA_D_REAL(nan_a) ), "! isinf( real(nan_a) )" );
    magma_assert_warn(   isinf( MAGMA_D_REAL(inf_a) ), "  isinf( real(inf_a) )" );
    
    // ----- magma_isnan
    magma_assert_warn( ! magma_d_isnan( a_a     ), "! magma_d_isnan( a_a     )" );
    #ifdef COMPLEX
    magma_assert_warn(   magma_d_isnan( a_nan   ), "  magma_d_isnan( a_nan   )" );
    #else
    magma_assert_warn( ! magma_d_isnan( a_nan   ), "! magma_d_isnan( a_nan   )" );  // for real, a_nan is just a.
    #endif
    magma_assert_warn( ! magma_d_isnan( a_inf   ), "! magma_d_isnan( a_inf   )" );
    magma_assert_warn(   magma_d_isnan( nan_a   ), "  magma_d_isnan( nan_a   )" );
    magma_assert_warn( ! magma_d_isnan( inf_a   ), "! magma_d_isnan( inf_a   )" );
    magma_assert_warn(   magma_d_isnan( nan_nan ), "  magma_d_isnan( nan_nan )" );
    magma_assert_warn(   magma_d_isnan( nan_inf ), "  magma_d_isnan( nan_inf )" );
    magma_assert_warn( ! magma_d_isnan( inf_inf ), "! magma_d_isnan( inf_inf )" );
    #ifdef COMPLEX
    magma_assert_warn(   magma_d_isnan( inf_nan ), "  magma_d_isnan( inf_nan )" );
    #else
    magma_assert_warn( ! magma_d_isnan( inf_nan ), "! magma_d_isnan( inf_nan )" );  // for real, inf_nan is just inf.
    #endif
    
    // ----- magma_isinf
    magma_assert_warn( ! magma_d_isinf( a_a     ), "! magma_d_isinf( a_a     )" );
    magma_assert_warn( ! magma_d_isinf( a_nan   ), "! magma_d_isinf( a_nan   )" );
    #ifdef COMPLEX
    magma_assert_warn(   magma_d_isinf( a_inf   ), "  magma_d_isinf( a_inf   )" );
    #else
    magma_assert_warn( ! magma_d_isinf( a_inf   ), "! magma_d_isinf( a_inf   )" );  // for real, a_inf is just a.
    #endif
    magma_assert_warn( ! magma_d_isinf( nan_a   ), "! magma_d_isinf( nan_a   )" );
    magma_assert_warn(   magma_d_isinf( inf_a   ), "  magma_d_isinf( inf_a   )" );
    magma_assert_warn( ! magma_d_isinf( nan_nan ), "! magma_d_isinf( nan_nan )" );
    #ifdef COMPLEX
    magma_assert_warn(   magma_d_isinf( nan_inf ), "  magma_d_isinf( nan_inf )" );
    #else
    magma_assert_warn( ! magma_d_isinf( nan_inf ), "! magma_d_isinf( nan_inf )" );  // for real, nan_inf is just nan.
    #endif
    magma_assert_warn(   magma_d_isinf( inf_inf ), "  magma_d_isinf( inf_inf )" );
    magma_assert_warn(   magma_d_isinf( inf_nan ), "  magma_d_isinf( inf_nan )" );
    
    // ----- magma_isnan_inf
    magma_assert_warn( ! magma_d_isnan_inf( a_a     ), "! magma_d_isnan_inf( a_a     )" );
    #ifdef COMPLEX
    magma_assert_warn(   magma_d_isnan_inf( a_nan   ), "  magma_d_isnan_inf( a_nan   )" );
    magma_assert_warn(   magma_d_isnan_inf( a_inf   ), "  magma_d_isnan_inf( a_inf   )" );
    #else
    magma_assert_warn( ! magma_d_isnan_inf( a_nan   ), "! magma_d_isnan_inf( a_nan   )" );  // for real, a_nan is just a.
    magma_assert_warn( ! magma_d_isnan_inf( a_inf   ), "! magma_d_isnan_inf( a_inf   )" );  // for real, a_inf is just a.
    #endif
    magma_assert_warn(   magma_d_isnan_inf( nan_a   ), "  magma_d_isnan_inf( nan_a   )" );
    magma_assert_warn(   magma_d_isnan_inf( nan_nan ), "  magma_d_isnan_inf( nan_nan )" );
    magma_assert_warn(   magma_d_isnan_inf( inf_a   ), "  magma_d_isnan_inf( inf_a   )" );
    magma_assert_warn(   magma_d_isnan_inf( inf_inf ), "  magma_d_isnan_inf( inf_inf )" );
    magma_assert_warn(   magma_d_isnan_inf( nan_inf ), "  magma_d_isnan_inf( nan_inf )" );
    magma_assert_warn(   magma_d_isnan_inf( inf_nan ), "  magma_d_isnan_inf( inf_nan )" );
    printf( "\n" );
    
    printf("%% uplo    M     N      CPU nan + inf             GPU nan + inf          actual nan + inf        \n");
    printf("%%==============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int iuplo = 0; iuplo < 3; ++iuplo ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];
            lda   = M;
            ldda  = magma_roundup( M, opts.align );  // multiple of 32 by default
            size  = lda*N;

            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( hA, double, lda *N );
            TESTING_MALLOC_DEV( dA, double, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &size, hA );
            
            // up to 25% of matrix is NAN, and
            // up to 25% of matrix is INF.
            magma_int_t cnt_nan = (magma_int_t)( (rand() / ((double)RAND_MAX)) * 0.25 * M*N );
            magma_int_t cnt_inf = (magma_int_t)( (rand() / ((double)RAND_MAX)) * 0.25 * M*N );
            magma_int_t total = cnt_nan + cnt_inf;
            assert( cnt_nan >= 0 );
            assert( cnt_inf >= 0 );
            assert( total <= M*N );
            
            // fill in indices
            TESTING_MALLOC_CPU( ii, magma_int_t, size );
            TESTING_MALLOC_CPU( jj, magma_int_t, size );
            for( cnt=0; cnt < size; ++cnt ) {
                ii[cnt] = cnt % M;
                jj[cnt] = cnt / M;
            }
            // shuffle indices
            for( cnt=0; cnt < total; ++cnt ) {
                i = magma_int_t( rand() / ((double)RAND_MAX) * size );
                tmp=ii[cnt];  ii[cnt]=ii[i];  ii[i]=tmp;
                tmp=jj[cnt];  jj[cnt]=jj[i];  jj[i]=tmp;
            }
            // fill in NAN and INF
            // for uplo, count NAN and INF in triangular portion of A
            magma_int_t c_nan=0;
            magma_int_t c_inf=0;
            for( cnt=0; cnt < cnt_nan; ++cnt ) {
                i = ii[cnt];
                j = jj[cnt];
                *hA(i,j) = MAGMA_D_NAN;
                if ( uplo[iuplo] == MagmaLower && i >= j ) { c_nan++; }
                if ( uplo[iuplo] == MagmaUpper && i <= j ) { c_nan++; }
            }
            for( cnt=cnt_nan; cnt < cnt_nan + cnt_inf; ++cnt ) {
                i = ii[cnt];
                j = jj[cnt];
                *hA(i,j) = MAGMA_D_INF;
                if ( uplo[iuplo] == MagmaLower && i >= j ) { c_inf++; }
                if ( uplo[iuplo] == MagmaUpper && i <= j ) { c_inf++; }
            }
            if ( uplo[iuplo] == MagmaLower || uplo[iuplo] == MagmaUpper ) {
                cnt_nan = c_nan;
                cnt_inf = c_inf;
                total = cnt_nan + cnt_inf;
            }
            
            //printf( "nan %g + %gi\n", MAGMA_D_REAL( MAGMA_D_NAN ), MAGMA_D_REAL( MAGMA_D_NAN ) );
            //printf( "inf %g + %gi\n", MAGMA_D_REAL( MAGMA_D_INF ), MAGMA_D_REAL( MAGMA_D_INF ) );
            //magma_dprint( M, N, hA, lda );
            
            magma_dsetmatrix( M, N, hA, lda, dA, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_int_t c_cpu_nan=-1, c_cpu_inf=-1;
            magma_int_t c_gpu_nan=-1, c_gpu_inf=-1;
            
            magma_int_t c_cpu = magma_dnan_inf    ( uplo[iuplo], M, N, hA, lda,  &c_cpu_nan, &c_cpu_inf );
            magma_int_t c_gpu = magma_dnan_inf_gpu( uplo[iuplo], M, N, dA, ldda, &c_gpu_nan, &c_gpu_inf );
            
            magma_int_t c_cpu2 = magma_dnan_inf    ( uplo[iuplo], M, N, hA, lda,  NULL, NULL );
            magma_int_t c_gpu2 = magma_dnan_inf_gpu( uplo[iuplo], M, N, dA, ldda, NULL, NULL );
            
            /* =====================================================================
               Check the result
               =================================================================== */
            bool okay = ( c_cpu == c_gpu )
                     && ( c_cpu == c_cpu2 )
                     && ( c_gpu == c_gpu2 )
                     && ( c_cpu == c_cpu_nan + c_cpu_inf )
                     && ( c_gpu == c_gpu_nan + c_gpu_inf )
                     && ( c_cpu_nan == cnt_nan )
                     && ( c_cpu_inf == cnt_inf )
                     && ( c_gpu_nan == cnt_nan )
                     && ( c_gpu_inf == cnt_inf );
            
            printf( "%4c %5d %5d   %10d + %-10d   %10d + %-10d   %10d + %-10d  %s\n",
                    lapacke_uplo_const( uplo[iuplo] ), (int) M, (int) N,
                    (int) c_cpu_nan, (int) c_cpu_inf,
                    (int) c_gpu_nan, (int) c_gpu_inf,
                    (int) cnt_nan,   (int) cnt_inf,
                    (okay ? "ok" : "failed"));
            status += ! okay;
            
            TESTING_FREE_CPU( hA );
            TESTING_FREE_DEV( dA );
            
            TESTING_FREE_CPU( ii );
            TESTING_FREE_CPU( jj );
        }
      }
      printf( "\n" );
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
