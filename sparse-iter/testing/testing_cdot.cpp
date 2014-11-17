/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from testing_zdot.cpp normal z -> c, Sat Nov 15 19:54:24 2014
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing zdot
*/
int main(  int argc, char** argv )
{
    // set queue for old dense routines
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    magmablasGetKernelStream( &queue );

    TESTING_INIT();




        printf("#================================================================================================================================================\n");
        printf("\n");
        printf("            |                            runtime                             |                              GFLOPS\n");
        printf("#n num_vecs |  CUDOT       CUGEMV       MAGMAGEMV       MDOT       MDGM      |      CUDOT       CUGEMV      MAGMAGEMV       MDOT       MDGM      \n");
        printf("#------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("\n");




    for( magma_int_t num_vecs=5; num_vecs<6; num_vecs+=1 ) {

    for( magma_int_t n=10000; n<100000001; n=n+10000 ) {
           
            magma_c_sparse_matrix A, B, C, D, E, F, G, H, I, J, K, Z;
            magma_c_vector a,b,c,x, y, z, skp;
            int iters = 10;
            float computations = (2.* n * iters * num_vecs); 

            
            magmaFloatComplex one = MAGMA_C_MAKE(1.0, 0.0);
            magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
            magmaFloatComplex alpha;

            #define ENABLE_TIMER
            #ifdef ENABLE_TIMER
            real_Double_t mdot1, mdot2, mdgm1, mdgm2, magmagemv1, magmagemv2, cugemv1, cugemv2, cudot1, cudot2;
            real_Double_t mdot_time, mdgm_time, magmagemv_time, cugemv_time, cudot_time;
            #endif


            magma_c_vinit( &a, Magma_DEV, n*num_vecs, one, queue );
            magma_c_vinit( &b, Magma_DEV, num_vecs, one, queue );
            int min_ten = min(num_vecs, 15);
            magma_c_vinit( &x, Magma_DEV, min_ten*n, one, queue );
            magma_c_vinit( &y, Magma_DEV, min_ten*n, one, queue );
            magma_c_vinit( &skp, Magma_DEV, num_vecs, zero, queue );

            // warm up
            magma_cgemvmdot( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );

            // CUDOT
            #ifdef ENABLE_TIMER
            cudot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                for( int l=0; l<num_vecs; l++)
                    alpha = magma_cdotc(n, a.dval, 1, b.dval, 1);
            }
            #ifdef ENABLE_TIMER
            cudot2 = magma_sync_wtime( queue );
            cudot_time=cudot2-cudot1;
            #endif
            // CUGeMV
            #ifdef ENABLE_TIMER
            cugemv1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                magma_cgemv(MagmaTrans, n, num_vecs, one, a.dval, n, b.dval, 1, zero, skp.dval, 1);
                //h++;
            }
            #ifdef ENABLE_TIMER
            cugemv2 = magma_sync_wtime( queue );
            cugemv_time=cugemv2-cugemv1;
            #endif
            // MAGMAGeMV
            #ifdef ENABLE_TIMER
            magmagemv1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                magmablas_cgemv(MagmaTrans, n, num_vecs, one, a.dval, n, b.dval, 1, zero, skp.dval, 1);
                //h++;
            }
            #ifdef ENABLE_TIMER
            magmagemv2 = magma_sync_wtime( queue );
            magmagemv_time=magmagemv2-magmagemv1;
            #endif
            // MDOT
            #ifdef ENABLE_TIMER
            mdot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                //magma_cmdotc( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                magma_cmdotc( n, 2, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                magma_cmdotc( n, 2, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                magma_cmdotc( n, 1, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                //h++;
            }
            #ifdef ENABLE_TIMER
            mdot2 = magma_sync_wtime( queue );
            mdot_time=mdot2-mdot1;
            #endif
            // MDGM
            #ifdef ENABLE_TIMER
            mdgm1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                magma_cgemvmdot( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                //h++;
            }
            #ifdef ENABLE_TIMER
            mdgm2 = magma_sync_wtime( queue );
            mdgm_time=mdgm2-mdgm1;
            #endif

            //magma_cprint_gpu(num_vecs,1,skp.dval,num_vecs);

            //Chronometry  
            #ifdef ENABLE_TIMER
            printf("%d  %d  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e\n", 
                    n, num_vecs, 
                    cudot_time/iters, 
                    (cugemv_time)/iters, 
                    (magmagemv_time)/iters,
                    (mdot_time)/iters,
                    (mdgm_time)/iters,
                    (float)(computations)/(cudot_time*(1.e+09)), 
                    (float)(computations)/(cugemv_time*(1.e+09)),
                    (float)(computations)/(magmagemv_time*(1.e+09)),
                    (float)(computations)/(mdot_time*(1.e+09)),
                    (float)(computations)/(mdgm_time*(1.e+09)) );
            #endif

            magma_c_vfree(&a, queue );
            magma_c_vfree(&b, queue );
            magma_c_vfree(&x, queue );
            magma_c_vfree(&y, queue );
            magma_c_vfree(&skp, queue );



        }


  //  }
        printf("#================================================================================================================================================\n");
        printf("\n");
        printf("\n");
}

    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
