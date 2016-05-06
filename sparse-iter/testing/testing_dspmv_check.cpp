/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/testing/testing_zspmv_check.cpp normal z -> d, Mon May  2 23:31:24 2016
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
    
    double one = MAGMA_D_MAKE(1.0, 0.0);
    double zero = MAGMA_D_MAKE(0.0, 0.0);
    magma_d_matrix A={Magma_CSR}, B_d={Magma_CSR};
    magma_d_matrix x={Magma_CSR}, b={Magma_CSR};

    int i=1;
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_dm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            CHECK( magma_d_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            int(A.num_rows), int(A.num_cols), int(A.nnz) );

        magma_int_t n = A.num_rows;
        CHECK( magma_dmtransfer( A, &B_d, Magma_CPU, Magma_DEV, queue ));

        // vectors and initial guess
        CHECK( magma_dvinit( &b, Magma_DEV, A.num_cols, 1, zero, queue ));
        CHECK( magma_dvinit( &x, Magma_DEV, A.num_cols, 1, one, queue ));
        
        CHECK( magma_dprint_vector( b, 90, 10, queue ));
        
            CHECK( magma_dprint_matrix( A, queue ));
            printf("\n\n\n");
            CHECK( magma_dprint_matrix( B_d, queue ));
        
        double res;
        res = magma_dnrm2(n, b.dval, 1, queue );
        printf("norm0: %f\n", res);
        
        CHECK( magma_d_spmv( one, B_d, x, zero, b, queue ));         //  b = A x

        CHECK( magma_dprint_vector( b, 0, 100, queue ));
        CHECK( magma_dprint_vector( b, b.num_rows-10, 10, queue ));

        res = magma_dnrm2( n, b.dval, 1, queue );
        printf("norm: %f\n", res);

        
        CHECK( magma_dresidual( B_d, x, b, &res, queue));
        printf("res: %f\n", res);


        magma_dmfree(&B_d, queue );

        magma_dmfree(&A, queue );
        
        magma_dmfree(&x, queue );
        magma_dmfree(&b, queue );

        i++;
    }

cleanup:
    magma_dmfree(&A, queue );
    magma_dmfree(&B_d, queue );
    magma_dmfree(&x, queue );
    magma_dmfree(&b, queue );
    
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
