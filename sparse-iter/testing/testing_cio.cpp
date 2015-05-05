/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from testing_zio.cpp normal z -> c, Sun May  3 11:23:02 2015
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
    magma_c_matrix A={Magma_CSR}, A2={Magma_CSR}, 
    A3={Magma_CSR}, A4={Magma_CSR}, A5={Magma_CSR};
    
    int i=1;
    CHECK( magma_cparse_opts( argc, argv, &zopts, &i, queue ));

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_cm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            CHECK( magma_c_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // filename for temporary matrix storage
        const char *filename = "testmatrix.mtx";

        // write to file
        CHECK( magma_cwrite_csrtomtx( A, filename, queue ));
        // read from file
        CHECK( magma_c_csr_mtx( &A2, filename, queue ));

        // delete temporary matrix
        unlink( filename );
                
        //visualize
        printf("A2:\n");
        CHECK( magma_cprint_matrix( A2, queue ));
        
        //visualize
        CHECK( magma_cmconvert(A2, &A4, Magma_CSR, Magma_CSRL, queue ));
        printf("A4:\n");
        CHECK( magma_cprint_matrix( A4, queue ));
        CHECK( magma_cmconvert(A4, &A5, Magma_CSR, Magma_ELL, queue ));
        printf("A5:\n");
        CHECK( magma_cprint_matrix( A5, queue ));

        // pass it to another application and back
        magma_int_t m, n;
        magma_index_t *row, *col;
        magmaFloatComplex *val=NULL;
        CHECK( magma_ccsrget( A2, &m, &n, &row, &col, &val, queue ));
        CHECK( magma_ccsrset( m, n, row, col, val, &A3, queue ));

        CHECK( magma_cmdiff( A, A2, &res, queue ));
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester IO:  ok\n");
        else
            printf("# tester IO:  failed\n");

        CHECK( magma_cmdiff( A, A3, &res, queue ));
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester matrix interface:  ok\n");
        else
            printf("# tester matrix interface:  failed\n");

        magma_cmfree(&A, queue );
        magma_cmfree(&A2, queue );
        magma_cmfree(&A4, queue );
        magma_cmfree(&A5, queue );


        i++;
    }
    
cleanup:
    magma_cmfree(&A, queue );
    magma_cmfree(&A2, queue );
    magma_cmfree(&A4, queue );
    magma_cmfree(&A5, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
