/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/testing/testing_zpreconditioner.cpp, normal z -> d, Sun Nov 20 20:20:46 2016
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
    
    double one = MAGMA_D_MAKE(1.0, 0.0);
    double zero = MAGMA_D_MAKE(0.0, 0.0);
    magma_d_matrix A={Magma_CSR}, B={Magma_CSR}, dB={Magma_CSR};
    magma_d_matrix x={Magma_CSR}, b={Magma_CSR}, t={Magma_CSR};
    magma_d_matrix x1={Magma_CSR}, x2={Magma_CSR};
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    
    int i=1;
    TESTING_CHECK( magma_dparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    TESTING_CHECK( magma_dsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_dm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_d_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n%% matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );


        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_rows;
        TESTING_CHECK( magma_deigensolverinfo_init( &zopts.solver_par, queue ));

        // scale matrix
        TESTING_CHECK( magma_dmscale( &A, zopts.scaling, queue ));

        TESTING_CHECK( magma_dmconvert( A, &B, Magma_CSR, zopts.output_format, queue ));
        TESTING_CHECK( magma_dmtransfer( B, &dB, Magma_CPU, Magma_DEV, queue ));

        // vectors and initial guess
        TESTING_CHECK( magma_dvinit( &b, Magma_DEV, A.num_cols, 1, one, queue ));
        TESTING_CHECK( magma_dvinit( &x, Magma_DEV, A.num_cols, 1, zero, queue ));
        TESTING_CHECK( magma_dvinit( &t, Magma_DEV, A.num_cols, 1, zero, queue ));
        TESTING_CHECK( magma_dvinit( &x1, Magma_DEV, A.num_cols, 1, zero, queue ));
        TESTING_CHECK( magma_dvinit( &x2, Magma_DEV, A.num_cols, 1, zero, queue ));
                        
        //preconditioner
        TESTING_CHECK( magma_d_precondsetup( dB, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        
        double residual;
        TESTING_CHECK( magma_dresidual( dB, b, x, &residual, queue ));
        zopts.solver_par.init_res = residual;
        printf("data = [\n");
        
        printf("%%runtime left preconditioner:\n");
        tempo1 = magma_sync_wtime( queue );
        info = magma_d_applyprecond_left( MagmaNoTrans, dB, b, &x1, &zopts.precond_par, queue ); 
        tempo2 = magma_sync_wtime( queue );
        if( info != 0 ){
            printf("error: preconditioner returned: %s (%lld).\n",
                    magma_strerror( info ), (long long) info );
        }
        TESTING_CHECK( magma_dresidual( dB, b, x1, &residual, queue ));
        printf("%.8e  %.8e\n", tempo2-tempo1, residual );
        
        printf("%%runtime right preconditioner:\n");
        tempo1 = magma_sync_wtime( queue );
        info = magma_d_applyprecond_right( MagmaNoTrans, dB, b, &x2, &zopts.precond_par, queue ); 
        tempo2 = magma_sync_wtime( queue );
        if( info != 0 ){
            printf("error: preconditioner returned: %s (%lld).\n",
                    magma_strerror( info ), (long long) info );
        }
        TESTING_CHECK( magma_dresidual( dB, b, x2, &residual, queue ));
        printf("%.8e  %.8e\n", tempo2-tempo1, residual );
        
        
        printf("];\n");
        
        info = magma_d_applyprecond_left( MagmaNoTrans, dB, b, &t, &zopts.precond_par, queue ); 
        info = magma_d_applyprecond_right( MagmaNoTrans, dB, t, &x, &zopts.precond_par, queue ); 

                
        TESTING_CHECK( magma_dresidual( dB, b, x, &residual, queue ));
        zopts.solver_par.final_res = residual;
        
        magma_dsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );

        magma_dmfree(&dB, queue );
        magma_dmfree(&B, queue );
        magma_dmfree(&A, queue );
        magma_dmfree(&x, queue );
        magma_dmfree(&x1, queue );
        magma_dmfree(&x2, queue );
        magma_dmfree(&b, queue );
        magma_dmfree(&t, queue );

        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
