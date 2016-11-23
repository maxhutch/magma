/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/testing/testing_zsolver_rhs.cpp, normal z -> c, Sun Nov 20 20:20:46 2016
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

    magma_copts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    //Chronometry
    real_Double_t tempo1, tempo2, t_transfer = 0.0;
    
    magmaFloatComplex one = MAGMA_C_MAKE(1.0, 0.0);
    magmaFloatComplex zero = MAGMA_C_MAKE(0.0, 0.0);
    magma_c_matrix A={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    magma_c_matrix x={Magma_CSR}, x_h={Magma_CSR}, b_h={Magma_DENSE}, b={Magma_DENSE};
    
    int i=1;
    TESTING_CHECK( magma_cparse_opts( argc, argv, &zopts, &i, queue ));
    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    TESTING_CHECK( magma_csolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));
    // more iterations

    
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_cm_5stencil(  laplace_size, &A, queue ));
            TESTING_CHECK( magma_cvinit( &b_h, Magma_CPU, A.num_cols, 1, one, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_c_csr_mtx( &A,  argv[i], queue ));
            TESTING_CHECK( magma_cvread( &b_h, A.num_cols, argv[i+1], queue ));
            i++;
        }

        printf( "\n%% matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
        
        printf("matrixinfo = [\n");
        printf("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m\n");
        printf("%%=============================================================%%\n");
        printf("  %8lld  %8lld      %10lld        %10lld\n",
               (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz,
               (long long) (A.nnz/A.num_rows) );
        printf("%%=============================================================%%\n");
        printf("];\n");
        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_cols;
        TESTING_CHECK( magma_ceigensolverinfo_init( &zopts.solver_par, queue ));
        fflush(stdout);
        

        t_transfer = 0.0;
        zopts.precond_par.setuptime = 0.0;
        zopts.precond_par.runtime = 0.0;
        //TESTING_CHECK( magma_cvinit( &b_h, Magma_CPU, A.num_cols, 1, MAGMA_C_ONE, queue ));

        i++;
        tempo1 = magma_sync_wtime( queue );
        magma_c_vtransfer(b_h, &b, Magma_CPU, Magma_DEV, queue);
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;
        
        // scale matrix
        TESTING_CHECK( magma_cmscale( &A, zopts.scaling, queue ));
        
        // preconditioner
        if ( zopts.solver_par.solver != Magma_ITERREF ) {
            TESTING_CHECK( magma_c_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        }
        // make sure alignment is 1 for SELLP
        B.alignment = 1;
        B.blocksize = 256;
        TESTING_CHECK( magma_cmconvert( A, &B, Magma_CSR, zopts.output_format, queue ));
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_cmtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ));
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;
        
        TESTING_CHECK( magma_cvinit( &x, Magma_DEV, A.num_cols, 1, zero, queue ));
        
        info = magma_c_solver( dB, b, &x, &zopts, queue );
        if( info != 0 ) {
            printf("%%error: solver returned: %s (%lld).\n",
                magma_strerror( info ), (long long) info );
        }
        
        magma_cmfree(&x_h, queue );
        tempo1 = magma_sync_wtime( queue );
        magma_c_vtransfer(x, &x_h, Magma_DEV, Magma_CPU, queue);
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;  
        
        printf("data = [\n");
        magma_csolverinfo( &zopts.solver_par, &zopts.precond_par, queue );
        printf("];\n\n");
        
        printf("precond_info = [\n");
        printf("%%   setup  runtime\n");        
        printf("  %.6f  %.6f\n",
           zopts.precond_par.setuptime, zopts.precond_par.runtime );
        printf("];\n\n");
        
        printf("transfer_time = %.6f;\n\n", t_transfer);
        magma_cmfree(&x, queue );
        magma_cmfree(&b, queue );
        magma_cmfree(&dB, queue );
        magma_cmfree(&B, queue );
        magma_csolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
        fflush(stdout);
        
        magma_cmfree(&dB, queue );
        magma_cmfree(&B, queue );
        magma_cmfree(&A, queue );
        magma_cmfree(&x, queue );
        magma_cmfree(&x_h, queue );
        magma_cmfree(&b, queue );
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
