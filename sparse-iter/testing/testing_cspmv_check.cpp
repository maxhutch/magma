/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/testing/testing_zspmv_check.cpp normal z -> c, Mon May  2 23:31:24 2016
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
    TESTING_INIT();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    magmaFloatComplex one = MAGMA_C_MAKE(1.0, 0.0);
    magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
    magma_c_matrix A={Magma_CSR}, B_d={Magma_CSR};
    magma_c_matrix x={Magma_CSR}, b={Magma_CSR};

    int i=1;
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_cm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            CHECK( magma_c_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            int(A.num_rows), int(A.num_cols), int(A.nnz) );

        magma_int_t n = A.num_rows;
        CHECK( magma_cmtransfer( A, &B_d, Magma_CPU, Magma_DEV, queue ));

        // vectors and initial guess
        CHECK( magma_cvinit( &b, Magma_DEV, A.num_cols, 1, zero, queue ));
        CHECK( magma_cvinit( &x, Magma_DEV, A.num_cols, 1, one, queue ));
        
        CHECK( magma_cprint_vector( b, 90, 10, queue ));
        
            CHECK( magma_cprint_matrix( A, queue ));
            printf("\n\n\n");
            CHECK( magma_cprint_matrix( B_d, queue ));
        
        float res;
        res = magma_scnrm2(n, b.dval, 1, queue );
        printf("norm0: %f\n", res);
        
        CHECK( magma_c_spmv( one, B_d, x, zero, b, queue ));         //  b = A x

        CHECK( magma_cprint_vector( b, 0, 100, queue ));
        CHECK( magma_cprint_vector( b, b.num_rows-10, 10, queue ));

        res = magma_scnrm2( n, b.dval, 1, queue );
        printf("norm: %f\n", res);

        
        CHECK( magma_cresidual( B_d, x, b, &res, queue));
        printf("res: %f\n", res);


        magma_cmfree(&B_d, queue );

        magma_cmfree(&A, queue );
        
        magma_cmfree(&x, queue );
        magma_cmfree(&b, queue );

        i++;
    }

cleanup:
    magma_cmfree(&A, queue );
    magma_cmfree(&B_d, queue );
    magma_cmfree(&x, queue );
    magma_cmfree(&b, queue );
    
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
