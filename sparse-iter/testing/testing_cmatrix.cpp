/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from testing_zmatrix.cpp normal z -> c, Sun May  3 11:23:02 2015
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
    magma_c_matrix Z={Magma_CSR}, A={Magma_CSR}, AT={Magma_CSR}, 
    A2={Magma_CSR}, B={Magma_CSR}, B_d={Magma_CSR};
    
    int i=1;
    CHECK( magma_cparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_cm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            CHECK( magma_c_csr_mtx( &Z,  argv[i], queue ));
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) Z.num_rows,(int) Z.num_cols,(int) Z.nnz );

        // scale matrix
        CHECK( magma_cmscale( &Z, zopts.scaling, queue ));

        // remove nonzeros in matrix
        CHECK( magma_cmcsrcompressor( &Z, queue ));
        
        // convert to be non-symmetric
        CHECK( magma_cmconvert( Z, &A, Magma_CSR, Magma_CSRL, queue ));
        
        // transpose
        CHECK( magma_cmtranspose( A, &AT, queue ));

        // convert, copy back and forth to check everything works

        CHECK( magma_cmconvert( AT, &B, Magma_CSR, zopts.output_format, queue ));
        magma_cmfree(&AT, queue );
        CHECK( magma_cmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));
        magma_cmfree(&B, queue );
        CHECK( magma_cmcsrcompressor_gpu( &B_d, queue ));
        CHECK( magma_cmtransfer( B_d, &B, Magma_DEV, Magma_CPU, queue ));
        magma_cmfree(&B_d, queue );
        CHECK( magma_cmconvert( B, &AT, zopts.output_format,Magma_CSR, queue ));
        magma_cmfree(&B, queue );

        // transpose back
        CHECK( magma_cmtranspose( AT, &A2, queue ));
        magma_cmfree(&AT, queue );
        CHECK( magma_cmdiff( A, A2, &res, queue));
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester:  ok\n");
        else
            printf("# tester:  failed\n");

        magma_cmfree(&A, queue );
        magma_cmfree(&A2, queue );
        magma_cmfree(&Z, queue );

        i++;
    }

cleanup:
    magma_cmfree(&AT, queue );
    magma_cmfree(&A, queue );
    magma_cmfree(&B, queue );
    magma_cmfree(&B_d, queue );
    magma_cmfree(&A2, queue );
    magma_cmfree(&Z, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
