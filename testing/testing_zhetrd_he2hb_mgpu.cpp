/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


#if defined(USEMKL)
#include <mkl_service.h>
#endif
#if defined(USEACML)
#include <omp.h>
#endif

#define PRECISION_z

#if defined(PRECISION_z) || defined(PRECISION_d)
extern "C" void cmp_vals(int n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);
extern "C" void zcheck_eig_(char *JOBZ, int  *MATYPE, int  *N, int  *NB,
                       magmaDoubleComplex* A, int  *LDA, double *AD, double *AE, double *D1, double *EIG,
                    magmaDoubleComplex *Z, int  *LDZ, magmaDoubleComplex *WORK, double *RWORK, double *RESU);
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhetrd_he2hb
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t gflops, gpu_time, gpu_perf;
    magmaDoubleComplex *h_A, *h_R, *h_work;
    magmaDoubleComplex *tau;
    double *D, *E;
    magma_int_t N, n2, lda, ldda, lwork, ldt, info, nstream;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    // TODO add these options to parse_opts
    magma_int_t NE      = 0;
    magma_int_t distblk = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    magma_int_t WANTZ = (opts.jobz == MagmaVec);
    double tol = opts.tolerance * lapackf77_dlamch("E");
    if (opts.nb == 0)
        opts.nb = 64; //magma_get_zhetrd_he2hb_nb(N);

    if (NE < 1)
        NE = N; //64; //magma_get_zhetrd_he2hb_nb(N);

    nstream = max(3, opts.ngpu+2);
    magma_queue_t streams[MagmaMaxGPUs][20];
    magmaDoubleComplex_ptr da[MagmaMaxGPUs], dT1[MagmaMaxGPUs];
    if ((distblk == 0) || (distblk < opts.nb))
        distblk = max(256, opts.nb);
    printf("voici ngpu %d distblk %d NB %d nstream %d\n ",
           (int) opts.ngpu, (int) distblk, (int) opts.nb, (int) nstream);

    for( magma_int_t dev = 0; dev < opts.ngpu; ++dev ) {
        magma_setdevice( dev );
        for( int i = 0; i < nstream; ++i ) {
            magma_queue_create( &streams[dev][i] );
        }
    }
    magma_setdevice( 0 );

    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N     = opts.nsize[itest];
            lda   = N;
            ldt   = N;
            ldda  = ((N+31)/32)*32;
            n2    = N*lda;
            /* We suppose the magma NB is bigger than lapack NB */
            lwork = N*opts.nb;
            //gflops = ....?

            /* Allocate host memory for the matrix */
            TESTING_MALLOC_CPU( tau,    magmaDoubleComplex, N-1   );

            TESTING_MALLOC_PIN( h_A,    magmaDoubleComplex, lda*N );
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, lda*N );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork );
            TESTING_MALLOC_PIN( D, double, N );
            TESTING_MALLOC_PIN( E, double, N );

            for( magma_int_t dev = 0; dev < opts.ngpu; ++dev ) {
                magma_int_t mlocal = ((N / distblk) / opts.ngpu + 1) * distblk;
                magma_setdevice( dev );
                TESTING_MALLOC_DEV( da[dev],  magmaDoubleComplex, ldda*mlocal );
                TESTING_MALLOC_DEV( dT1[dev], magmaDoubleComplex, N*opts.nb        );
            }
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hermitian( N, h_A, lda );

            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            /* Copy the matrix to the GPU */
            magma_zsetmatrix_1D_col_bcyclic( N, N, h_R, lda, da, ldda, opts.ngpu, distblk);
            //magmaDoubleComplex_ptr dabis;
            //TESTING_MALLOC_DEV( dabis,  magmaDoubleComplex, ldda*N );
            //magma_zsetmatrix(N, N, h_R, lda, dabis, ldda);

            for (int count=0; count < 1; ++count) {
                magma_setdevice(0);
                gpu_time = magma_wtime();
                if (opts.version == 30) {
                    magma_zhetrd_he2hb_mgpu_spec(
                        opts.uplo, N, opts.nb, h_R, lda, tau, h_work, lwork,
                        da, ldda, dT1, opts.nb, opts.ngpu, distblk,
                        streams, nstream, opts.nthread, &info);
                } else {
                    nstream = 3;
                    magma_zhetrd_he2hb_mgpu(
                        opts.uplo, N, opts.nb, h_R, lda, tau, h_work, lwork,
                        da, ldda, dT1, opts.nb, opts.ngpu, distblk,
                        streams, nstream, opts.nthread, &info);
                }
                // magma_zhetrd_he2hb(opts.uplo, N, opts.nb, h_R, lda, tau, h_work, lwork, dT1[0], &info);
                gpu_time = magma_wtime() - gpu_time;
                printf("  Finish BAND  N %d  NB %d  dist %d  ngpu %d version %d timing= %f\n",
                       N, opts.nb, distblk, opts.ngpu, opts.version, gpu_time);
            }
            magma_setdevice(0);

            for( magma_int_t dev = 0; dev < opts.ngpu; ++dev ) {
                magma_setdevice(dev);
                magma_device_sync();
            }
            magma_setdevice(0);
            magmablasSetKernelStream( NULL );

            // todo neither of these is declared in headers
            // magma_zhetrd_bhe2trc_v5(opts.nthread, WANTZ, opts.uplo, NE, N, opts.nb, h_R, lda, D, E, dT1[0], ldt);
            // magma_zhetrd_bhe2trc(opts.nthread, WANTZ, opts.uplo, NE, N, opts.nb, h_R, lda, D, E, dT1[0], ldt);
            
            // todo where is this timer started?
            // gpu_time = magma_wtime() - gpu_time;
            
            // todo what are the gflops?
            gpu_perf = gflops / gpu_time;
            
            if (info != 0)
                printf("magma_zhetrd_he2hb returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
#if defined(CHECKEIG)
#if defined(PRECISION_z) || defined(PRECISION_d)
            if ( opts.check ) {
                printf("  Total N %5d  flops %6.2f  timing %6.2f seconds\n", (int) N, gpu_perf, gpu_time );
                char JOBZ;
                if (WANTZ == 0)
                    JOBZ = 'N';
                else
                    JOBZ = 'V';
                double nrmI=0.0, nrm1=0.0, nrm2=0.0;
                int    lwork2 = 256*N;
                magmaDoubleComplex *work2, *AINIT;
                double *rwork2, *D2;
                // TODO free this memory !
                magma_zmalloc_cpu( &work2, lwork2 );
                magma_dmalloc_cpu( &rwork2, N );
                magma_dmalloc_cpu( &D2, N );
                magma_zmalloc_cpu( &AINIT, N*lda );
                memcpy(AINIT, h_A, N*lda*sizeof(magmaDoubleComplex));
                
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                int nt = min(12, opts.nthread);

                #if defined(USEMKL)
                mkl_set_num_threads(nt);
                #endif
                #if defined(USEACML)
                omp_set_num_threads(nt);
                #endif

                #if defined(PRECISION_z) || defined (PRECISION_c)
                lapackf77_zheev( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, rwork2, &info );
                #else
                lapackf77_dsyev( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, &info );
                #endif
                ///* call eigensolver for our resulting tridiag [D E] and for Q */
                //dstedc_withZ('V', N, D, E, h_R, lda);
                ////dsterf_( &N, D, E, &info);
                
                cpu_time = magma_wtime() - cpu_time;
                printf("  Finish CHECK - EIGEN   timing= %f  threads %d\n", cpu_time, nt);

                /* compare result */
                cmp_vals(N, D2, D, &nrmI, &nrm1, &nrm2);

                magmaDoubleComplex *WORKAJETER;
                double *RWORKAJETER, *RESU;
                // TODO free this memory !
                magma_zmalloc_cpu( &WORKAJETER, (2* N * N + N)  );
                magma_dmalloc_cpu( &RWORKAJETER, N  );
                magma_dmalloc_cpu( &RESU, 10 );
                int MATYPE;
                memset(RESU, 0, 10*sizeof(double));

                MATYPE=3;
                double NOTHING=0.0;
                cpu_time = magma_wtime();
                // check results
                zcheck_eig_( lapack_vec_const(opts.jobz), &MATYPE, &N, &opts.nb,
                             AINIT, &lda, &NOTHING, &NOTHING, D2, D,
                             h_R, &lda, WORKAJETER, RWORKAJETER, RESU );
                cpu_time = magma_wtime() - cpu_time;
                printf("  Finish CHECK - results timing= %f\n", cpu_time);
                #if defined(USEMKL)
                mkl_set_num_threads(1);
                #endif
                #if defined(USEACML)
                omp_set_num_threads(1);
                #endif

                printf("\n");
                printf(" ================================================================================================================\n");
                printf("   ==> INFO voici  threads=%d    N=%d    NB=%d   WANTZ=%d\n", (int) opts.nthread, (int) N, (int) opts.nb, (int) WANTZ);
                printf(" ================================================================================================================\n");
                printf("            DSBTRD                : %15s \n", "STATblgv9withQ    ");
                printf(" ================================================================================================================\n");
                if (WANTZ > 0)
                    printf(" | A - U S U' | / ( |A| n ulp )   : %15.3E   \n", RESU[0]);
                if (WANTZ > 0)
                    printf(" | I - U U' | / ( n ulp )         : %15.3E   \n", RESU[1]);
                printf(" | D1 - EVEIGS | / (|D| ulp)      : %15.3E   \n",  RESU[2]);
                printf(" max | D1 - EVEIGS |              : %15.3E   \n",  RESU[6]);
                printf(" ================================================================================================================\n\n\n");

                printf(" ****************************************************************************************************************\n");
                printf(" * Hello here are the norm  Infinite (max)=%8.2e  norm one (sum)=%8.2e   norm2(sqrt)=%8.2e *\n", nrmI, nrm1, nrm2);
                printf(" ****************************************************************************************************************\n\n");
            }
#endif  // PRECISION_z || PRECISION_d
#endif  // CHECKEIG

            printf("  Total N %5d  flops %6.2f        timing %6.2f seconds\n", (int) N, 0.0, gpu_time );
            printf("============================================================================\n\n\n");

            TESTING_FREE_CPU( tau    );

            TESTING_FREE_PIN( h_A    );
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            TESTING_FREE_PIN( D      );
            TESTING_FREE_PIN( E      );

            for( magma_int_t dev = 0; dev < opts.ngpu; ++dev ) {
                magma_setdevice( dev );
                TESTING_FREE_DEV( da[dev]  );
                TESTING_FREE_DEV( dT1[dev] );
            }
            magma_setdevice( 0 );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( magma_int_t dev = 0; dev < opts.ngpu; ++dev ) {
        for( int i = 0; i < nstream; ++i ) {
            magma_queue_destroy( streams[dev][i] );
        }
    }

    TESTING_FINALIZE();
    return status;
}
