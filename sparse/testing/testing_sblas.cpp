/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/testing/testing_zblas.cpp, normal z -> s, Sun Nov 20 20:20:46 2016
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_int_t j, n=1000000, FLOPS;
    magma_int_t count = 100;
    
    float one = MAGMA_S_MAKE( 1.0, 0.0 );
    float two = MAGMA_S_MAKE( 2.0, 0.0 );

    magma_s_matrix a={Magma_CSR}, ad={Magma_CSR}, bd={Magma_CSR}, cd={Magma_CSR};
    TESTING_CHECK( magma_svinit( &a, Magma_CPU, n, 1, one, queue ));
    TESTING_CHECK( magma_svinit( &bd, Magma_DEV, n, 1, two, queue ));
    TESTING_CHECK( magma_svinit( &cd, Magma_DEV, n, 1, one, queue ));
    
    TESTING_CHECK( magma_smtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ));

    real_Double_t start, end, res;
    
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        res = magma_snrm2( n, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA nrm2: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_sscal( n, two, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA scal: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_saxpy( n, one, ad.dval, 1, bd.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA axpy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        magma_scopy( n, bd.dval, 1, ad.dval, 1, queue );
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA copy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j < count; j++) {
        res = MAGMA_S_REAL( magma_sdot( n, ad.dval, 1, bd.dval, 1, queue ));
    }
    end = magma_sync_wtime( queue );
    printf( " > MAGMA dotc: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/count, FLOPS*count/1e9/(end-start) );

    printf("%% tester BLAS:  ok\n");
    
    // use res to silence compiler warnings
    if ( isnan( real( res ))) {
        info = -1;
    }

    magma_smfree( &a, queue);
    magma_smfree(&ad, queue);
    magma_smfree(&bd, queue);
    magma_smfree(&cd, queue);
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
