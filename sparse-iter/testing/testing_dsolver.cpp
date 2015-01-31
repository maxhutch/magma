/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zsolver.cpp normal z -> d, Fri Jan 30 19:00:33 2015
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

    magma_dopts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;
    magma_dparse_opts( argc, argv, &zopts, &i, queue );


    double one = MAGMA_D_MAKE(1.0, 0.0);
    double zero = MAGMA_D_MAKE(0.0, 0.0);
    magma_d_sparse_matrix A, B, B_d;
    magma_d_vector x, b;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    if ( zopts.solver_par.solver != Magma_PCG &&
         zopts.solver_par.solver != Magma_PGMRES &&
         zopts.solver_par.solver != Magma_PBICGSTAB &&
         zopts.solver_par.solver != Magma_ITERREF )
    zopts.precond_par.solver = Magma_NONE;

    magma_dsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue );

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_dm_5stencil(  laplace_size, &A, queue );
        } else {                        // file-matrix test
            magma_d_csr_mtx( &A,  argv[i], queue );
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );


        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_rows;
        magma_deigensolverinfo_init( &zopts.solver_par, queue );

        // scale matrix
        magma_dmscale( &A, zopts.scaling, queue );

        magma_d_mconvert( A, &B, Magma_CSR, zopts.output_format, queue );
        magma_d_mtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue );

        // vectors and initial guess
        magma_d_vinit( &b, Magma_DEV, A.num_cols, one, queue );
        magma_d_vinit( &x, Magma_DEV, A.num_cols, one, queue );
        magma_d_spmv( one, B_d, x, zero, b, queue );                 //  b = A x
        magma_d_vfree(&x, queue );
        magma_d_vinit( &x, Magma_DEV, A.num_cols, zero, queue );

        magma_d_solver( B_d, b, &x, &zopts, queue );         

        magma_dsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );

        magma_d_mfree(&B_d, queue );
        magma_d_mfree(&B, queue );
        magma_d_mfree(&A, queue ); 
        magma_d_vfree(&x, queue );
        magma_d_vfree(&b, queue );

        i++;
    }

    magma_dsolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
