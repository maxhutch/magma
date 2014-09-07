/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s
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
#include "mkl_spblas.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing zdot
*/
int main( int argc, char** argv)
{
    TESTING_INIT();




        printf("#================================================================================================================================================\n");
        printf("\n");
        printf("            |                            runtime                             |                              GFLOPS\n");
        printf("#n num_vecs |  CUDOT       CUGEMV       MAGMAGEMV       MDOT       MDGM      |      CUDOT       CUGEMV      MAGMAGEMV       MDOT       MDGM      \n");
        printf("#------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("\n");




    for( magma_int_t num_vecs=5; num_vecs<6; num_vecs+=1 ){

    for( magma_int_t n=10000; n<100000001; n=n+10000 ){
           
            magma_z_sparse_matrix A, B, C, D, E, F, G, H, I, J, K, Z;
            magma_z_vector a,b,c,x, y, z, skp;
            int iters = 10;
            double computations = (2.* n * iters * num_vecs); 

            
            magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
            magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
            magmaDoubleComplex alpha;

            #define ENABLE_TIMER
            #ifdef ENABLE_TIMER
            double mdot1, mdot2, mdgm1, mdgm2, magmagemv1, magmagemv2, cugemv1, cugemv2, cudot1, cudot2;
            double mdot_time, mdgm_time, magmagemv_time, cugemv_time, cudot_time;
            #endif


            magma_z_vinit( &a, Magma_DEV, n*num_vecs, one );
            magma_z_vinit( &b, Magma_DEV, num_vecs, one );
            int min_ten = min(num_vecs, 15);
            magma_z_vinit( &x, Magma_DEV, min_ten*n, one );
            magma_z_vinit( &y, Magma_DEV, min_ten*n, one );
            magma_z_vinit( &skp, Magma_DEV, num_vecs, zero );

            // warm up
            magma_zgemvmdot( n, num_vecs, a.val, b.val, x.val, y.val, skp.val );

            // CUDOT
            #ifdef ENABLE_TIMER
            magma_device_sync(); cudot1=magma_wtime();
            #endif
            for( int h=0; h<iters; h++){
                for( int l=0; l<num_vecs; l++)
                    alpha = magma_zdotc(n, a.val, 1, b.val, 1);
            }
            #ifdef ENABLE_TIMER
            magma_device_sync(); cudot2=magma_wtime();
            cudot_time=cudot2-cudot1;
            #endif
            // CUGeMV
            #ifdef ENABLE_TIMER
            magma_device_sync(); cugemv1=magma_wtime();
            #endif
            for( int h=0; h<iters; h++){
                magma_zgemv(MagmaTrans, n, num_vecs, one, a.val, n, b.val, 1, zero, skp.val, 1);
                //h++;
            }
            #ifdef ENABLE_TIMER
            magma_device_sync(); cugemv2=magma_wtime();
            cugemv_time=cugemv2-cugemv1;
            #endif
            // MAGMAGeMV
            #ifdef ENABLE_TIMER
            magma_device_sync(); magmagemv1=magma_wtime();
            #endif
            for( int h=0; h<iters; h++){
                magmablas_zgemv(MagmaTrans, n, num_vecs, one, a.val, n, b.val, 1, zero, skp.val, 1);
                //h++;
            }
            #ifdef ENABLE_TIMER
            magma_device_sync(); magmagemv2=magma_wtime();
            magmagemv_time=magmagemv2-magmagemv1;
            #endif
            // MDOT
            #ifdef ENABLE_TIMER
            magma_device_sync(); mdot1=magma_wtime();
            #endif
            for( int h=0; h<iters; h++){
                //magma_zmdotc( n, num_vecs, a.val, b.val, x.val, y.val, skp.val );
                magma_zmdotc( n, 2, a.val, b.val, x.val, y.val, skp.val );
                magma_zmdotc( n, 2, a.val, b.val, x.val, y.val, skp.val );
                magma_zmdotc( n, 1, a.val, b.val, x.val, y.val, skp.val );
                //h++;
            }
            #ifdef ENABLE_TIMER
            magma_device_sync(); mdot2=magma_wtime();
            mdot_time=mdot2-mdot1;
            #endif
            // MDGM
            #ifdef ENABLE_TIMER
            magma_device_sync(); mdgm1=magma_wtime();
            #endif
            for( int h=0; h<iters; h++){
                magma_zgemvmdot( n, num_vecs, a.val, b.val, x.val, y.val, skp.val );
                //h++;
            }
            #ifdef ENABLE_TIMER
            magma_device_sync(); mdgm2=magma_wtime();
            mdgm_time=mdgm2-mdgm1;
            #endif

            //magma_zprint_gpu(num_vecs,1,skp.val,num_vecs);

            //Chronometry  
            #ifdef ENABLE_TIMER
            printf("%d  %d  %e  %e  %e  %e  %e  %e  %e  %e  %e  %e\n", 
                    n, num_vecs, 
                    cudot_time/iters, 
                    (cugemv_time)/iters, 
                    (magmagemv_time)/iters,
                    (mdot_time)/iters,
                    (mdgm_time)/iters,
                    (double)(computations)/(cudot_time*(1.e+09)), 
                    (double)(computations)/(cugemv_time*(1.e+09)),
                    (double)(computations)/(magmagemv_time*(1.e+09)),
                    (double)(computations)/(mdot_time*(1.e+09)),
                    (double)(computations)/(mdgm_time*(1.e+09)) );
            #endif

            magma_z_vfree(&a);
            magma_z_vfree(&b);
            magma_z_vfree(&x);
            magma_z_vfree(&y);
            magma_z_vfree(&skp);



        }


  //  }
        printf("#================================================================================================================================================\n");
        printf("\n");
        printf("\n");

    }

    TESTING_FINALIZE();
    return 0;
}
