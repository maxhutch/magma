/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/testing/testing_zmadd.cpp normal z -> d, Mon May  2 23:31:25 2016
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
   -- testing csr matrix add
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_INIT();
    
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    real_Double_t res;
    magma_d_matrix A={Magma_CSR}, B={Magma_CSR}, B2={Magma_CSR}, 
    A_d={Magma_CSR}, B_d={Magma_CSR}, C_d={Magma_CSR};

    double one = MAGMA_D_MAKE(1.0, 0.0);
    double mone = MAGMA_D_MAKE(-1.0, 0.0);

    magma_int_t i=1;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        CHECK( magma_dm_5stencil(  laplace_size, &A, queue ));
    } else {                        // file-matrix test
        CHECK( magma_d_csr_mtx( &A,  argv[i], queue ));
    }
    printf("%% matrix info: %d-by-%d with %d nonzeros\n",
                        int(A.num_rows), int(A.num_cols), int(A.nnz) );
    i++;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        CHECK( magma_dm_5stencil(  laplace_size, &B, queue ));
    } else {                        // file-matrix test
        CHECK( magma_d_csr_mtx( &B,  argv[i], queue ));
    }
    printf("%% matrix info: %d-by-%d with %d nonzeros\n",
                        int(B.num_rows), int(B.num_cols), int(B.nnz) );


    CHECK( magma_dmtransfer( A, &A_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));

    CHECK( magma_dcuspaxpy( &one, A_d, &one, B_d, &C_d, queue ));

    magma_dmfree(&B_d, queue );

    CHECK( magma_dcuspaxpy( &mone, A_d, &one, C_d, &B_d, queue ));
    
    CHECK( magma_dmtransfer( B_d, &B2, Magma_DEV, Magma_CPU, queue ));

    magma_dmfree(&A_d, queue );
    magma_dmfree(&B_d, queue );
    magma_dmfree(&C_d, queue );

    // check difference
    CHECK( magma_dmdiff( B, B2, &res, queue ));
    printf("%% ||A-B||_F = %8.2e\n", res);
    if ( res < .000001 )
        printf("%% tester matrix add:  ok\n");
    else
        printf("%% tester matrix add:  failed\n");

cleanup:
    magma_dmfree(&A_d, queue );
    magma_dmfree(&B_d, queue );
    magma_dmfree(&C_d, queue );
    magma_dmfree(&A, queue );
    magma_dmfree(&B, queue );
    magma_dmfree(&B2, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
