/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from testing_zmcompressor.cpp normal z -> c, Sun May  3 11:23:02 2015
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
#include "magma_lapack.h"
#include "testings.h"
#include "common_magmasparse.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_INIT();

    magma_copts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );

    real_Double_t res;
    magma_c_matrix A={Magma_CSR}, AT={Magma_CSR}, A2={Magma_CSR}, 
    B={Magma_CSR}, B_d={Magma_CSR};
    
    int i=1;
    real_Double_t start, end;
    CHECK( magma_cparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_cm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            CHECK( magma_c_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        CHECK( magma_cmscale( &A, zopts.scaling, queue ));

        // remove nonzeros in matrix
        start = magma_sync_wtime( queue );
        for (int j=0; j<10; j++)
            CHECK( magma_cmcsrcompressor( &A, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA CPU: %.2e seconds.\n", (end-start)/10 );
        // transpose
        CHECK( magma_cmtranspose( A, &AT, queue ));

        // convert, copy back and forth to check everything works
        CHECK( magma_cmconvert( AT, &B, Magma_CSR, Magma_CSR, queue ));
        magma_cmfree(&AT, queue );
        CHECK( magma_cmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));
        magma_cmfree(&B, queue );

        start = magma_sync_wtime( queue );
        for (int j=0; j<10; j++)
            CHECK( magma_cmcsrcompressor_gpu( &B_d, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA GPU: %.2e seconds.\n", (end-start)/10 );


        CHECK( magma_cmtransfer( B_d, &B, Magma_DEV, Magma_CPU, queue ));
        magma_cmfree(&B_d, queue );
        CHECK( magma_cmconvert( B, &AT, Magma_CSR, Magma_CSR, queue ));
        magma_cmfree(&B, queue );

        // transpose back
        CHECK( magma_cmtranspose( AT, &A2, queue ));
        magma_cmfree(&AT, queue );
        CHECK( magma_cmdiff( A, A2, &res, queue ));
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester matrix compressor:  ok\n");
        else
            printf("# tester matrix compressor:  failed\n");

        magma_cmfree(&A, queue );
        magma_cmfree(&A2, queue );

        i++;
    }
    
cleanup:
    magma_cmfree(&AT, queue );
    magma_cmfree(&B, queue );
    magma_cmfree(&A, queue );
    magma_cmfree(&A2, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
