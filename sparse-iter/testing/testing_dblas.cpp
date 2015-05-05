/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from testing_zblas.cpp normal z -> d, Sun May  3 11:23:02 2015
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "common_magmasparse.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_INIT();
    magma_queue_t queue=NULL;
    magma_queue_create( &queue );
    magmablasSetKernelStream( queue );

    magma_int_t j, n=1000000, FLOPS;
    
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
    for (j=0; j<100; j++)
        res = magma_dnrm2(n, ad.dval, 1);
    end = magma_sync_wtime( queue );
    printf( " > MAGMA nrm2: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        magma_dscal( n, two, ad.dval, 1 );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA scal: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        magma_daxpy( n, one, ad.dval, 1, bd.dval, 1 );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA axpy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        magma_dcopy( n, bd.dval, 1, ad.dval, 1 );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA copy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        res = MAGMA_D_REAL( magma_ddot(n, ad.dval, 1, bd.dval, 1) );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA dotc: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );

    printf("# tester BLAS:  ok\n");


    magma_dmfree( &a, queue);
    magma_dmfree(&ad, queue);
    magma_dmfree(&bd, queue);
    magma_dmfree(&cd, queue);

    
cleanup:
    magma_dmfree( &a, queue);
    magma_dmfree(&ad, queue);
    magma_dmfree(&bd, queue);
    magma_dmfree(&cd, queue);
    magmablasSetKernelStream( NULL );
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
