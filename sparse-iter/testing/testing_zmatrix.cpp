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

    magma_zopts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;
    magma_zparse_opts( argc, argv, &zopts, &i, queue );


    real_Double_t res;
    magma_z_sparse_matrix Z, A, AT, A2, B, B_d;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_zm_5stencil(  laplace_size, &Z, queue );
        } else {                        // file-matrix test
            magma_z_csr_mtx( &Z,  argv[i], queue );
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) Z.num_rows,(int) Z.num_cols,(int) Z.nnz );

        // scale matrix
        magma_zmscale( &Z, zopts.scaling, queue );

        // remove nonzeros in matrix
        magma_zmcsrcompressor( &Z, queue );
        
        // convert to be non-symmetric
        magma_z_mconvert( Z, &A, Magma_CSR, Magma_CSRL, queue );
        
        // transpose
        magma_z_mtranspose( A, &AT, queue );

        // convert, copy back and forth to check everything works

        magma_z_mconvert( AT, &B, Magma_CSR, zopts.output_format, queue );
        magma_z_mfree(&AT, queue );
        magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue );
        magma_z_mfree(&B, queue );
        magma_zmcsrcompressor_gpu( &B_d, queue );
        magma_z_mtransfer( B_d, &B, Magma_DEV, Magma_CPU, queue );
        magma_z_mfree(&B_d, queue );
        magma_z_mconvert( B, &AT, zopts.output_format,Magma_CSR, queue );      
        magma_z_mfree(&B, queue );

        // transpose back
        magma_z_mtranspose( AT, &A2, queue );
        magma_z_mfree(&AT, queue );
        magma_zmdiff( A, A2, &res, queue);
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester:  ok\n");
        else
            printf("# tester:  failed\n");

        magma_z_mfree(&A, queue ); 
        magma_z_mfree(&A2, queue );
        magma_z_mfree(&Z, queue ); 

        i++;
    }
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
