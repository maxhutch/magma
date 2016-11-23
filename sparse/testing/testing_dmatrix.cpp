/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/testing/testing_zmatrix.cpp, normal z -> d, Sun Nov 20 20:20:46 2016
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

    magma_dopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    real_Double_t res;
    magma_d_matrix Z={Magma_CSR}, A={Magma_CSR}, AT={Magma_CSR}, 
    A2={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    
    magma_index_t *comm_i=NULL;
    double *comm_v=NULL;
    magma_int_t start, end;
    
    int i=1;
    TESTING_CHECK( magma_dparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_dm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_d_csr_mtx( &Z,  argv[i], queue ));
        }

        printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
                (long long) Z.num_rows, (long long) Z.num_cols, (long long) Z.nnz );
        
        // slice matrix
        TESTING_CHECK( magma_index_malloc_cpu( &comm_i, Z.num_rows ) );
        TESTING_CHECK( magma_dmalloc_cpu( &comm_v, Z.num_rows ) );
        
        TESTING_CHECK( magma_dmslice( 1, 0, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_dprint_matrix( A2, queue );
        magma_dprint_matrix( AT, queue );
        magma_dprint_matrix( B, queue );
        magma_dmfree(&A2, queue );
        magma_dmfree(&AT, queue );
        magma_dmfree(&B, queue );

        TESTING_CHECK( magma_dmslice( 9, 0, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_dprint_matrix( A2, queue );
        magma_dprint_matrix( AT, queue );
        magma_dprint_matrix( B, queue );
        magma_dmfree(&A2, queue );
        magma_dmfree(&AT, queue );
        magma_dmfree(&B, queue );
        
        TESTING_CHECK( magma_dmslice( 9, 1, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_dprint_matrix( A2, queue );
        magma_dprint_matrix( AT, queue );
        magma_dprint_matrix( B, queue );
        magma_dmfree(&A2, queue );
        magma_dmfree(&AT, queue );
        magma_dmfree(&B, queue );

        TESTING_CHECK( magma_dmslice( 9, 8, Z, &A2, &AT, &B, comm_i, comm_v, &start, &end, queue ) );    
        magma_dprint_matrix( A2, queue );
        magma_dprint_matrix( AT, queue );
        magma_dprint_matrix( B, queue );
        magma_dmfree(&A2, queue );
        magma_dmfree(&AT, queue );
        magma_dmfree(&B, queue );
        
        
        // scale matrix
        TESTING_CHECK( magma_dmscale( &Z, zopts.scaling, queue ));

        // remove nonzeros in matrix
        TESTING_CHECK( magma_dmcsrcompressor( &Z, queue ));
        
        // convert to be non-symmetric
        TESTING_CHECK( magma_dmconvert( Z, &A, Magma_CSR, Magma_CSRL, queue ));
        
        // transpose
        TESTING_CHECK( magma_dmtranspose( A, &AT, queue ));

        // convert, copy back and forth to check everything works

        TESTING_CHECK( magma_dmconvert( AT, &B, Magma_CSR, zopts.output_format, queue ));
        magma_dmfree(&AT, queue );
        TESTING_CHECK( magma_dmtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ));
        magma_dmfree(&B, queue );
        TESTING_CHECK( magma_dmcsrcompressor_gpu( &dB, queue ));
        TESTING_CHECK( magma_dmtransfer( dB, &B, Magma_DEV, Magma_CPU, queue ));
        magma_dmfree(&dB, queue );
        TESTING_CHECK( magma_dmconvert( B, &AT, zopts.output_format,Magma_CSR, queue ));
        magma_dmfree(&B, queue );

        // transpose back
        TESTING_CHECK( magma_dmtranspose( AT, &A2, queue ));
        magma_dmfree(&AT, queue );
        TESTING_CHECK( magma_dmdiff( A, A2, &res, queue));
        printf("%% ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester:  ok\n");
        else
            printf("%% tester:  failed\n");
        
        magma_free_cpu( comm_i );
        magma_free_cpu( comm_v );
        comm_i=NULL;
        comm_v=NULL;
        magma_dmfree(&A, queue );
        magma_dmfree(&A2, queue );
        magma_dmfree(&Z, queue );

        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
