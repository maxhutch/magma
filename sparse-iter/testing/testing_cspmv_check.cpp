/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zspmv_check.cpp normal z -> c, Fri Jan 30 19:00:33 2015
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
   -- testing any solver 
*/
int main(  int argc, char** argv )
{
    TESTING_INIT();
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;

    magmaFloatComplex one = MAGMA_C_MAKE(1.0, 0.0);
    magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
    magma_c_sparse_matrix A, B, B_d;
    magma_c_vector x, b;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_cm_5stencil(  laplace_size, &A, queue );
        } else {                        // file-matrix test
            magma_c_csr_mtx( &A,  argv[i], queue );
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        magma_int_t n = A.num_rows;
        magma_c_mtransfer( A, &B_d, Magma_CPU, Magma_DEV, queue );

        // vectors and initial guess
        magma_c_vinit( &b, Magma_DEV, A.num_cols, zero, queue );
        magma_c_vinit( &x, Magma_DEV, A.num_cols, one, queue );
        
        magma_c_vvisu( b, 90, 10, queue );
        
            magma_c_mvisu( A, queue );
            printf("\n\n\n");
            magma_c_mvisu( B_d, queue );
        
        float res;
        res = magma_scnrm2(n, b.dval, 1 );
        printf("norm0: %f\n", res);
        
        magma_c_spmv( one, B_d, x, zero, b, queue );                 //  b = A x

        magma_c_vvisu( b, 0, 100, queue );
        magma_c_vvisu( b, b.num_rows-10, 10, queue );

        res = magma_scnrm2(n, b.dval, 1 );
        printf("norm: %f\n", res);

        
        magma_cresidual( B_d, x, b, &res, queue);
        printf("res: %f\n", res);


        magma_c_mfree(&B_d, queue );

        magma_c_mfree(&A, queue ); 
        
        magma_c_vfree(&x, queue );
        magma_c_vfree(&b, queue );

        i++;
    }

    
    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
