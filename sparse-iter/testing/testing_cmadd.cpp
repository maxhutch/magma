/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from sparse-iter/testing/testing_zmadd.cpp, normal z -> c, Tue Aug 30 09:39:21 2016
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
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing csr matrix add
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    real_Double_t res;
    magma_c_matrix A={Magma_CSR}, B={Magma_CSR}, B2={Magma_CSR}, 
    A_d={Magma_CSR}, B_d={Magma_CSR}, C_d={Magma_CSR};

    magmaFloatComplex one = MAGMA_C_MAKE(1.0, 0.0);
    magmaFloatComplex mone = MAGMA_C_MAKE(-1.0, 0.0);

    magma_int_t i=1;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        TESTING_CHECK( magma_cm_5stencil(  laplace_size, &A, queue ));
    } else {                        // file-matrix test
        TESTING_CHECK( magma_c_csr_mtx( &A,  argv[i], queue ));
    }
    printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
            (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
    i++;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        TESTING_CHECK( magma_cm_5stencil(  laplace_size, &B, queue ));
    } else {                        // file-matrix test
        TESTING_CHECK( magma_c_csr_mtx( &B,  argv[i], queue ));
    }
    printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
            (long long) B.num_rows, (long long) B.num_cols, (long long) B.nnz );


    TESTING_CHECK( magma_cmtransfer( A, &A_d, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_cmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));

    TESTING_CHECK( magma_ccuspaxpy( &one, A_d, &one, B_d, &C_d, queue ));

    magma_cmfree(&B_d, queue );

    TESTING_CHECK( magma_ccuspaxpy( &mone, A_d, &one, C_d, &B_d, queue ));
    
    TESTING_CHECK( magma_cmtransfer( B_d, &B2, Magma_DEV, Magma_CPU, queue ));

    magma_cmfree(&A_d, queue );
    magma_cmfree(&B_d, queue );
    magma_cmfree(&C_d, queue );

    // check difference
    TESTING_CHECK( magma_cmdiff( B, B2, &res, queue ));
    printf("%% ||A-B||_F = %8.2e\n", res);
    if ( res < .000001 )
        printf("%% tester matrix add:  ok\n");
    else
        printf("%% tester matrix add:  failed\n");

    magma_cmfree(&A, queue );
    magma_cmfree(&B, queue );
    magma_cmfree(&B2, queue );
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
