/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/testing/testing_zio.cpp, normal z -> s, Sun Nov 20 20:20:46 2016
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>  // unlink
// on Windows, _unlink is in stdio.h; testings.h defines unlink as _unlink
#endif

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_sopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    real_Double_t res;
    magma_s_matrix A={Magma_CSR}, A2={Magma_CSR}, 
    A3={Magma_CSR}, A4={Magma_CSR}, A5={Magma_CSR};
    
    int i=1;
    TESTING_CHECK( magma_sparse_opts( argc, argv, &zopts, &i, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_sm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_s_csr_mtx( &A,  argv[i], queue ));
        }

        printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );

        // filename for temporary matrix storage
        const char *filename = "testmatrix.mtx";

        // write to file
        TESTING_CHECK( magma_swrite_csrtomtx( A, filename, queue ));
        // read from file
        TESTING_CHECK( magma_s_csr_mtx( &A2, filename, queue ));

        // delete temporary matrix
        unlink( filename );
                
        //visualize
        printf("A2:\n");
        TESTING_CHECK( magma_sprint_matrix( A2, queue ));
        
        //visualize
        TESTING_CHECK( magma_smconvert(A2, &A4, Magma_CSR, Magma_CSRL, queue ));
        printf("A4:\n");
        TESTING_CHECK( magma_sprint_matrix( A4, queue ));
        TESTING_CHECK( magma_smconvert(A4, &A5, Magma_CSR, Magma_ELL, queue ));
        printf("A5:\n");
        TESTING_CHECK( magma_sprint_matrix( A5, queue ));

        // pass it to another application and back
        magma_int_t m, n;
        magma_index_t *row, *col;
        float *val=NULL;
        TESTING_CHECK( magma_scsrget( A2, &m, &n, &row, &col, &val, queue ));
        TESTING_CHECK( magma_scsrset( m, n, row, col, val, &A3, queue ));

        TESTING_CHECK( magma_smdiff( A, A2, &res, queue ));
        printf("%% ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester IO:  ok\n");
        else
            printf("%% tester IO:  failed\n");

        TESTING_CHECK( magma_smdiff( A, A3, &res, queue ));
        printf("%% ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester matrix interface:  ok\n");
        else
            printf("%% tester matrix interface:  failed\n");

        magma_smfree(&A, queue );
        magma_smfree(&A2, queue );
        magma_smfree(&A4, queue );
        magma_smfree(&A5, queue );

        i++;
    }
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
