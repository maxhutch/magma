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
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_matrix A={Magma_CSR}, B={Magma_CSR}, B_d={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR}, t={Magma_CSR};
    magma_z_matrix x1={Magma_CSR}, x2={Magma_CSR};
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    
    int i=1;
    CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
        }

        printf( "\n%% matrix info: %d-by-%d with %d nonzeros\n\n",
                            int(A.num_rows), int(A.num_cols), int(A.nnz) );


        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_rows;
        CHECK( magma_zeigensolverinfo_init( &zopts.solver_par, queue ));

        // scale matrix
        CHECK( magma_zmscale( &A, zopts.scaling, queue ));

        CHECK( magma_zmconvert( A, &B, Magma_CSR, zopts.output_format, queue ));
        CHECK( magma_zmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));

        // vectors and initial guess
        CHECK( magma_zvinit( &b, Magma_DEV, A.num_cols, 1, one, queue ));
        CHECK( magma_zvinit( &x, Magma_DEV, A.num_cols, 1, zero, queue ));
        CHECK( magma_zvinit( &t, Magma_DEV, A.num_cols, 1, zero, queue ));
        CHECK( magma_zvinit( &x1, Magma_DEV, A.num_cols, 1, zero, queue ));
        CHECK( magma_zvinit( &x2, Magma_DEV, A.num_cols, 1, zero, queue ));
                        
        //preconditioner
        CHECK( magma_z_precondsetup( B_d, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        
        double residual;
        CHECK( magma_zresidual( B_d, b, x, &residual, queue ));
        zopts.solver_par.init_res = residual;
        printf("data = [\n");
        
        printf("%%runtime left preconditioner:\n");
        tempo1 = magma_sync_wtime( queue );
        info = magma_z_applyprecond_left( MagmaNoTrans, B_d, b, &x1, &zopts.precond_par, queue ); 
        tempo2 = magma_sync_wtime( queue );
        if( info != 0 ){
            printf("error: preconditioner returned: %s (%d).\n",
                magma_strerror( info ), int(info) );
        }
        CHECK( magma_zresidual( B_d, b, x1, &residual, queue ));
        printf("%.8e  %.8e\n", tempo2-tempo1, residual );
        
        printf("%%runtime right preconditioner:\n");
        tempo1 = magma_sync_wtime( queue );
        info = magma_z_applyprecond_right( MagmaNoTrans, B_d, b, &x2, &zopts.precond_par, queue ); 
        tempo2 = magma_sync_wtime( queue );
        if( info != 0 ){
            printf("error: preconditioner returned: %s (%d).\n",
                magma_strerror( info ), int(info) );
        }
        CHECK( magma_zresidual( B_d, b, x2, &residual, queue ));
        printf("%.8e  %.8e\n", tempo2-tempo1, residual );
        
        
        printf("];\n");
        
        info = magma_z_applyprecond_left( MagmaNoTrans, B_d, b, &t, &zopts.precond_par, queue ); 
        info = magma_z_applyprecond_right( MagmaNoTrans, B_d, t, &x, &zopts.precond_par, queue ); 

                
        CHECK( magma_zresidual( B_d, b, x, &residual, queue ));
        zopts.solver_par.final_res = residual;
        
        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );

        magma_zmfree(&B_d, queue );
        magma_zmfree(&B, queue );
        magma_zmfree(&A, queue );
        magma_zmfree(&x, queue );
        magma_zmfree(&x1, queue );
        magma_zmfree(&x2, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&t, queue );

        i++;
    }

cleanup:
    magma_zmfree(&B_d, queue );
    magma_zmfree(&B, queue );
    magma_zmfree(&A, queue );
    magma_zmfree(&x, queue );
    magma_zmfree(&x1, queue );
    magma_zmfree(&x2, queue );
    magma_zmfree(&b, queue );
    magma_zmfree(&t, queue );
    magma_zsolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
