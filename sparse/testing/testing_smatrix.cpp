/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/testing/testing_zmatrix.cpp, normal z -> s, Sun Nov 20 20:20:46 2016
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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
    magma_s_matrix Z={Magma_CSR}, A={Magma_CSR}, AT={Magma_CSR}, 
    A2={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    
    magma_index_t *comm_i=NULL;
    float *comm_v=NULL;
    magma_int_t start, end;
    
    int i=1;
    TESTING_CHECK( magma_sparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_sm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_s_csr_mtx( &Z,  argv[i], queue ));
        }

        printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
                (long long) Z.num_rows, (long long) Z.num_cols, (long long) Z.nnz );
        
        // slice matrix
        TESTING_CHECK( magma_index_malloc_cpu( &comm_i, Z.num_rows ) );
        TESTING_CHECK( magma_smalloc_cpu( &comm_v, Z.num_rows ) );
        
        TESTING_CHECK( magma_smslice( 1, 0, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_sprint_matrix( A2, queue );
        magma_sprint_matrix( AT, queue );
        magma_sprint_matrix( B, queue );
        magma_smfree(&A2, queue );
        magma_smfree(&AT, queue );
        magma_smfree(&B, queue );

        TESTING_CHECK( magma_smslice( 9, 0, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_sprint_matrix( A2, queue );
        magma_sprint_matrix( AT, queue );
        magma_sprint_matrix( B, queue );
        magma_smfree(&A2, queue );
        magma_smfree(&AT, queue );
        magma_smfree(&B, queue );
        
        TESTING_CHECK( magma_smslice( 9, 1, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_sprint_matrix( A2, queue );
        magma_sprint_matrix( AT, queue );
        magma_sprint_matrix( B, queue );
        magma_smfree(&A2, queue );
        magma_smfree(&AT, queue );
        magma_smfree(&B, queue );

        TESTING_CHECK( magma_smslice( 9, 8, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_sprint_matrix( A2, queue );
        magma_sprint_matrix( AT, queue );
        magma_sprint_matrix( B, queue );
        magma_smfree(&A2, queue );
        magma_smfree(&AT, queue );
        magma_smfree(&B, queue );
        
        
        // scale matrix
        TESTING_CHECK( magma_smscale( &Z, zopts.scaling, queue ));

        // remove nonzeros in matrix
        TESTING_CHECK( magma_smcsrcompressor( &Z, queue ));
        
        // convert to be non-symmetric
        TESTING_CHECK( magma_smconvert( Z, &A, Magma_CSR, Magma_CSRL, queue ));
        
        // transpose
        TESTING_CHECK( magma_smtranspose( A, &AT, queue ));

        // convert, copy back and forth to check everything works

        TESTING_CHECK( magma_smconvert( AT, &B, Magma_CSR, zopts.output_format, queue ));
        magma_smfree(&AT, queue );
        TESTING_CHECK( magma_smtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ));
        magma_smfree(&B, queue );
        TESTING_CHECK( magma_smcsrcompressor_gpu( &dB, queue ));
        TESTING_CHECK( magma_smtransfer( dB, &B, Magma_DEV, Magma_CPU, queue ));
        magma_smfree(&dB, queue );
        TESTING_CHECK( magma_smconvert( B, &AT, zopts.output_format,Magma_CSR, queue ));
        magma_smfree(&B, queue );

        // transpose back
        TESTING_CHECK( magma_smtranspose( AT, &A2, queue ));
        magma_smfree(&AT, queue );
        TESTING_CHECK( magma_smdiff( A, A2, &res, queue));
        printf("%% ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester:  ok\n");
        else
            printf("%% tester:  failed\n");
        
        magma_free_cpu( comm_i );
        magma_free_cpu( comm_v );
        comm_i=NULL;
        comm_v=NULL;
        magma_smfree(&A, queue );
        magma_smfree(&A2, queue );
        magma_smfree(&Z, queue );

        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
