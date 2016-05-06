/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/testing/testing_zblas.cpp normal z -> d, Mon May  2 23:31:23 2016
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
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_INIT();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_int_t j, n=1000000, FLOPS;
    magma_int_t count = 100;
    
    double one = MAGMA_D_MAKE( 1.0, 0.0 );
    double two = MAGMA_D_MAKE( 2.0, 0.0 );

    magma_d_matrix a={Magma_CSR}, ad={Magma_CSR}, bd={Magma_CSR}, cd={Magma_CSR};
    CHECK( magma_dvinit( &a, Magma_CPU, n, 1, one, queue ));
    CHECK( magma_dvinit( &bd, Magma_DEV, n, 1, two, queue ));
    CHECK( magma_dvinit( &cd, Magma_DEV, n, 1, one, queue ));
    
    CHECK( magma_dmtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ));

    real_Double_t start, end, res;
    
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        res = magma_dnrm2( n, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA nrm2: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_dscal( n, two, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA scal: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_daxpy( n, one, ad.dval, 1, bd.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA axpy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_dcopy( n, bd.dval, 1, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA copy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        res = MAGMA_D_REAL( magma_ddot( n, ad.dval, 1, bd.dval, 1, queue ));
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA dotc: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );

    printf("%% tester BLAS:  ok\n");
    
    // use res to silence compiler warnings
    if ( isnan( real( res ))) {
        info = -1;
    }

cleanup:
    magma_dmfree( &a, queue);
    magma_dmfree(&ad, queue);
    magma_dmfree(&bd, queue);
    magma_dmfree(&cd, queue);
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
