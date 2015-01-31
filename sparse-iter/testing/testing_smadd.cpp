/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zmadd.cpp normal z -> s, Fri Jan 30 19:00:33 2015
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
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing csr matrix add
*/
int main(  int argc, char** argv )
{
    TESTING_INIT();
    
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );

    real_Double_t res;
    magma_s_sparse_matrix A, B, B2, C, A_d, B_d, C_d;

    float one = MAGMA_S_MAKE(1.0, 0.0);
    float zero = MAGMA_S_MAKE(0.0, 0.0);
    float mone = MAGMA_S_MAKE(-1.0, 0.0);

    magma_int_t i=1;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        magma_sm_5stencil(  laplace_size, &A, queue );
    } else {                        // file-matrix test
        magma_s_csr_mtx( &A,  argv[i], queue );
    }
    printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                        (int) A.num_rows,(int) A.num_cols,(int) A.nnz );
    i++;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        magma_sm_5stencil(  laplace_size, &B, queue );
    } else {                        // file-matrix test
        magma_s_csr_mtx( &B,  argv[i], queue );
    }
    printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                        (int) B.num_rows,(int) B.num_cols,(int) B.nnz );


    magma_s_mtransfer( A, &A_d, Magma_CPU, Magma_DEV, queue );
    magma_s_mtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue );

    magma_scuspaxpy( &one, A_d, &one, B_d, &C_d, queue );

    magma_s_mfree(&B_d, queue );

    magma_scuspaxpy( &mone, A_d, &one, C_d, &B_d, queue );
    
    magma_s_mtransfer( B_d, &B2, Magma_DEV, Magma_CPU, queue );

    magma_s_mfree(&A_d, queue );
    magma_s_mfree(&B_d, queue );
    magma_s_mfree(&C_d, queue );

    // check difference
    magma_smdiff( B, B2, &res, queue );
    printf("# ||A-B||_F = %8.2e\n", res);
    if ( res < .000001 )
        printf("# tester matrix add:  ok\n");
    else
        printf("# tester matrix add:  failed\n");

    magma_s_mfree(&A, queue ); 
    magma_s_mfree(&B, queue ); 
    magma_s_mfree(&B2, queue ); 

    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
