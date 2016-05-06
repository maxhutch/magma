/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/testing/testing_zmdotc.cpp normal z -> d, Mon May  2 23:31:24 2016
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"
#include "magmasparse_internal.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing zdot
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    const double one  = MAGMA_D_MAKE(1.0, 0.0);
    const double zero = MAGMA_D_MAKE(0.0, 0.0);
    double alpha;

    TESTING_INIT();

    magma_d_matrix a={Magma_CSR}, b={Magma_CSR}, x={Magma_CSR}, y={Magma_CSR}, skp={Magma_CSR};

    printf("%%================================================================================================================================================\n");
    printf("\n");
    printf("%%            |     runtime            |       GFLOPS\n");
    printf("%% n num_vecs |  CUDOT    MAGMA MDOTC  |  CUDOT    MAGMA MDOTC\n");
    printf("%%------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("\n");

    for( magma_int_t num_vecs=2; num_vecs < 9; num_vecs += 2 ) {
        for( magma_int_t n=1000000; n < 5000001; n += 1000000 ) {
            int iters = 10;
            double computations = ( n * iters * num_vecs);

            #define ENABLE_TIMER
            #ifdef ENABLE_TIMER
            real_Double_t mdot1, mdot2, cudot1, cudot2;
            real_Double_t mdot_time, cudot_time;
            #endif

            CHECK( magma_dvinit( &a, Magma_DEV, n, num_vecs, one, queue ));
            CHECK( magma_dvinit( &b, Magma_DEV, num_vecs, 1, one, queue ));
            int min_ten = min(num_vecs, 15);
            CHECK( magma_dvinit( &x, Magma_DEV, min_ten, n, one, queue ));
            CHECK( magma_dvinit( &y, Magma_DEV, min_ten, n, one, queue ));
            CHECK( magma_dvinit( &skp, Magma_DEV, num_vecs, 1, zero, queue ));

            // warm up
            CHECK( magma_dgemvmdot( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));

            // CUDOT
            #ifdef ENABLE_TIMER
            cudot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                for( int l=0; l<num_vecs/2; l++) {
                    alpha = magma_ddot( n, a.dval, 1, b.dval, 1, queue );
                }
            }
            #ifdef ENABLE_TIMER
            cudot2 = magma_sync_wtime( queue );
            cudot_time=cudot2-cudot1;
            #endif
          
            // MAGMA MDOTC
            #ifdef ENABLE_TIMER
            mdot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                if( num_vecs == 2 ){
                    magma_dmdotc1( n, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                }
                else if( num_vecs == 4 ){
                    magma_dmdotc1( n, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                    magma_dmdotc1( n, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                }
                else if( num_vecs == 6 ){
                    magma_dmdotc3( n, a.dval, b.dval, a.dval+n, b.dval+n, a.dval+2*n, b.dval+2*n, x.dval, y.dval, skp.dval, queue );
                }
                else if( num_vecs == 8 ){
                    magma_dmdotc3( n, a.dval, b.dval, a.dval+n, b.dval+n, a.dval+2*n, b.dval+2*n, x.dval, y.dval, skp.dval, queue );
                }
                else{
                    printf("error: not supported.\n");
                }
            }
            #ifdef ENABLE_TIMER
            mdot2 = magma_sync_wtime( queue );
            mdot_time=mdot2-mdot1;
            #endif
           
            //Chronometry
            #ifdef ENABLE_TIMER
            printf("%d  %d  %e  %e  %e  %e\n",
                    int(n), int(num_vecs),
                    cudot_time/iters,
                    (mdot_time)/iters,
                    computations/(cudot_time*1e9),
                    computations/(mdot_time*1e9));
            #endif

            magma_dmfree(&a, queue );
            magma_dmfree(&b, queue );
            magma_dmfree(&x, queue );
            magma_dmfree(&y, queue );
            magma_dmfree(&skp, queue );
        }

        printf("%%================================================================================================================================================\n");
        printf("\n");
        printf("\n");
    }
    
    // use alpha to silence compiler warnings
    if ( isnan( real( alpha ))) {
        info = -1;
    }

cleanup:
    magma_dmfree(&a, queue );
    magma_dmfree(&b, queue );
    magma_dmfree(&x, queue );
    magma_dmfree(&y, queue );
    magma_dmfree(&skp, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
