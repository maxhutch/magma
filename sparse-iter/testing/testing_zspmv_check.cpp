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
   -- testing any solver 
*/
int main(  int argc, char** argv )
{
    TESTING_INIT();
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_zm_5stencil(  laplace_size, &A, queue );
        } else {                        // file-matrix test
            magma_z_csr_mtx( &A,  argv[i], queue );
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        magma_int_t n = A.num_rows;
        magma_z_mtransfer( A, &B_d, Magma_CPU, Magma_DEV, queue );

        // vectors and initial guess
        magma_z_vinit( &b, Magma_DEV, A.num_cols, zero, queue );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, one, queue );
        
        magma_z_vvisu( b, 90, 10, queue );
        
            magma_z_mvisu( A, queue );
            printf("\n\n\n");
            magma_z_mvisu( B_d, queue );
        
        double res;
        res = magma_dznrm2(n, b.dval, 1 );
        printf("norm0: %f\n", res);
        
        magma_z_spmv( one, B_d, x, zero, b, queue );                 //  b = A x

        magma_z_vvisu( b, 0, 100, queue );
        magma_z_vvisu( b, b.num_rows-10, 10, queue );

        res = magma_dznrm2(n, b.dval, 1 );
        printf("norm: %f\n", res);

        
        magma_zresidual( B_d, x, b, &res, queue);
        printf("res: %f\n", res);


        magma_z_mfree(&B_d, queue );

        magma_z_mfree(&A, queue ); 
        
        magma_z_vfree(&x, queue );
        magma_z_vfree(&b, queue );

        i++;
    }

    
    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
