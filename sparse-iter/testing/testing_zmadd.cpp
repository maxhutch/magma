/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> c d s
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
    magma_z_sparse_matrix A, B, B2, C, A_d, B_d, C_d;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex mone = MAGMA_Z_MAKE(-1.0, 0.0);

    magma_int_t i=1;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        magma_zm_5stencil(  laplace_size, &A, queue );
    } else {                        // file-matrix test
        magma_z_csr_mtx( &A,  argv[i], queue );
    }
    printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                        (int) A.num_rows,(int) A.num_cols,(int) A.nnz );
    i++;

    if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
        i++;
        magma_int_t laplace_size = atoi( argv[i] );
        magma_zm_5stencil(  laplace_size, &B, queue );
    } else {                        // file-matrix test
        magma_z_csr_mtx( &B,  argv[i], queue );
    }
    printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                        (int) B.num_rows,(int) B.num_cols,(int) B.nnz );


    magma_z_mtransfer( A, &A_d, Magma_CPU, Magma_DEV, queue );
    magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue );

    magma_zcuspaxpy( &one, A_d, &one, B_d, &C_d, queue );

    magma_z_mfree(&B_d, queue );

    magma_zcuspaxpy( &mone, A_d, &one, C_d, &B_d, queue );
    
    magma_z_mtransfer( B_d, &B2, Magma_DEV, Magma_CPU, queue );

    magma_z_mfree(&A_d, queue );
    magma_z_mfree(&B_d, queue );
    magma_z_mfree(&C_d, queue );

    // check difference
    magma_zmdiff( B, B2, &res, queue );
    printf("# ||A-B||_F = %8.2e\n", res);
    if ( res < .000001 )
        printf("# tester matrix add:  ok\n");
    else
        printf("# tester matrix add:  failed\n");

    magma_z_mfree(&A, queue ); 
    magma_z_mfree(&B, queue ); 
    magma_z_mfree(&B2, queue ); 

    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
