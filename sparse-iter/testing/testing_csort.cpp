/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/testing/testing_zsort.cpp normal z -> c, Mon May  2 23:31:24 2016
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>


// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"
#include "magmasparse_internal.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_INIT();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_int_t i, n=100;
    magma_index_t *x=NULL;
    magmaFloatComplex *y=NULL;
    
    magma_c_matrix A={Magma_CSR};

    CHECK( magma_index_malloc_cpu( &x, n ));
    printf("unsorted:\n");
    srand(time(NULL));
    for(i = 0; i < n; i++ ){
        int r = rand()%100;
        x[i] = r;
        printf("%d  ", x[i]);
    }
    printf("\n\n");
    
    printf("sorting...");
    CHECK( magma_cindexsort(x, 0, n-1, queue ));
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(i = 0; i < n; i++ ){
        printf("%d  ", x[i]);
    }
    printf("\n\n");

    magma_free_cpu( x );
    
    
    CHECK( magma_cmalloc_cpu( &y, n ));
    printf("unsorted:\n");
    srand(time(NULL));
    for(i = 0; i < n; i++ ){
        float r = (float) rand()/(float) 10.;
        y[i] = MAGMA_C_MAKE( r, 0.0);
        if(i%5==0)
            y[i] = - y[i];
        printf("%2.2f + %2.2f  ", MAGMA_C_REAL(y[i]), MAGMA_C_IMAG(y[i]) );
    }
    printf("\n\n");
    
    printf("sorting...");
    CHECK( magma_csort(y, 0, n-1, queue ));
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(i = 0; i < n; i++ ){
        printf("%2.2f + %2.2f  ", MAGMA_C_REAL(y[i]), MAGMA_C_IMAG(y[i]) );
    }
    printf("\n\n");

    magma_free_cpu( y );
    
    i=1;
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_cm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            CHECK( magma_c_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            int(A.num_rows), int(A.num_cols), int(A.nnz) );
    
        CHECK( magma_index_malloc_cpu( &x, A.num_rows*10 ));
        magma_int_t num_ind = 0;

        CHECK( magma_cdomainoverlap( A.num_rows, &num_ind, A.row, A.col, x, queue ));
                printf("domain overlap indices:\n");
        for(magma_int_t j = 0; j<num_ind; j++ ){
            printf("%d  ", int(x[j]) );
        }
        printf("\n\n");
        magma_free_cpu( x );
        magma_cmfree(&A, queue);
        
        i++;
    }

cleanup:
    magma_cmfree(&A, queue );
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
