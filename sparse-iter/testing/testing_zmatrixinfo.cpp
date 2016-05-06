/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

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
    TESTING_INIT();

    magma_zopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    magma_z_matrix Z={Magma_CSR};
    
    int i=1;
    CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));
    printf("matrixinfo = [ \n");
    printf("%%   size (n)   ||   nonzeros (nnz)   ||   nnz/n \n");
    printf("%%=============================================================%%\n");
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_zm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            CHECK( magma_z_csr_mtx( &Z,  argv[i], queue ));
        }

        printf("   %10d          %10d          %10d\n",
               int(Z.num_rows),  int(Z.nnz), int(Z.nnz/Z.num_rows) );

        magma_zmfree(&Z, queue );

        i++;
    }
    printf("%%=============================================================%%\n");
    printf("];\n");
        
cleanup:
    magma_zmfree(&Z, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
