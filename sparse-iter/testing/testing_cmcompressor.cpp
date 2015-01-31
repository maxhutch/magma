/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zmcompressor.cpp normal z -> c, Fri Jan 30 19:00:33 2015
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

    magma_copts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );

    int i=1;
    real_Double_t start, end;
    magma_cparse_opts( argc, argv, &zopts, &i, queue );


    real_Double_t res;
    magma_c_sparse_matrix A, AT, A2, B, B_d;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

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

        // scale matrix
        magma_cmscale( &A, zopts.scaling, queue );

        // remove nonzeros in matrix
        start = magma_sync_wtime( queue ); 
        for (int j=0; j<10; j++)
            magma_cmcsrcompressor( &A, queue );
        end = magma_sync_wtime( queue ); 
        printf( " > MAGMA CPU: %.2e seconds.\n", (end-start)/10 );
        // transpose
        magma_c_mtranspose( A, &AT, queue );

        // convert, copy back and forth to check everything works
        magma_c_mconvert( AT, &B, Magma_CSR, Magma_CSR, queue );
        magma_c_mfree(&AT, queue ); 
        magma_c_mtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue );
        magma_c_mfree(&B, queue );

        start = magma_sync_wtime( queue ); 
        for (int j=0; j<10; j++)
            magma_cmcsrcompressor_gpu( &B_d, queue );
        end = magma_sync_wtime( queue ); 
        printf( " > MAGMA GPU: %.2e seconds.\n", (end-start)/10 );


        magma_c_mtransfer( B_d, &B, Magma_DEV, Magma_CPU, queue );
        magma_c_mfree(&B_d, queue );
        magma_c_mconvert( B, &AT, Magma_CSR, Magma_CSR, queue );      
        magma_c_mfree(&B, queue );

        // transpose back
        magma_c_mtranspose( AT, &A2, queue );
        magma_c_mfree(&AT, queue ); 
        magma_cmdiff( A, A2, &res, queue );
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester matrix compressor:  ok\n");
        else
            printf("# tester matrix compressor:  failed\n");

        magma_c_mfree(&A, queue ); 
        magma_c_mfree(&A2, queue ); 

        i++;
    }
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
