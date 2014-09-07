/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from testing_zmatrix.cpp normal z -> d, Tue Sep  2 12:38:36 2014
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
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_dopts zopts;

    int i=1;
    magma_dparse_opts( argc, argv, &zopts, &i);


    real_Double_t res;
    magma_d_sparse_matrix A, AT, A2, B, B_d;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ){

        magma_d_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_dmscale( &A, zopts.scaling );

        // transpose
        magma_d_mtranspose( A, &AT );

        // convert, copy back and forth to check everything works
        magma_d_mconvert( AT, &B, Magma_CSR, zopts.output_format );
        magma_d_mfree(&AT); 
        magma_d_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );
        magma_d_mfree(&B);
        magma_d_mtransfer( B_d, &B, Magma_DEV, Magma_CPU );
        magma_d_mfree(&B_d);
        magma_d_mconvert( B, &AT, zopts.output_format,Magma_CSR );      
        magma_d_mfree(&B);

        // transpose back
        magma_d_mtranspose( AT, &A2 );
        magma_d_mfree(&AT); 
        magma_dmdiff( A, A2, &res);
        printf(" ||A-B||_F = %f\n", res);

        magma_d_mfree(&A); 
        magma_d_mfree(&A2); 

        i++;
    }

    TESTING_FINALIZE();
    return 0;
}
