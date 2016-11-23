/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Hartwig Anzt

       @generated from sparse/src/zftjacobi.cpp, normal z -> s, Sun Nov 20 20:20:45 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )



    /**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    This routine takes the system matrix A and the RHS b as input.
    This is the fault-tolerant version of Jacobi according to ScalLA'15.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix M = D^(-1) * (L+U)
                
    @param[in]
    b           magma_s_matrix
                input RHS b
                
    @param[in,out]
    x           magma_s_matrix*
                iteration vector x

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sftjacobi(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // some useful variables
    real_Double_t tempo1, tempo2, runtime=0;
    float residual;
    
    // local variables
    float c_zero = MAGMA_S_ZERO;
    magma_int_t dofs = A.num_cols;
    magma_int_t k = solver_par->verbose;
    
    magma_s_matrix xkm2 = {Magma_CSR};
    magma_s_matrix xkm1 = {Magma_CSR};
    magma_s_matrix xk = {Magma_CSR};
    magma_s_matrix z = {Magma_CSR};
    magma_s_matrix c = {Magma_CSR};
    magma_int_t *flag_t = NULL;
    magma_int_t *flag_fp = NULL;
    
    float delta = 0.9;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    magma_s_matrix r={Magma_CSR}, d={Magma_CSR}, ACSR={Magma_CSR};
    
    CHECK( magma_smconvert(A, &ACSR, A.storage_type, Magma_CSR, queue ) );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    // solver setup
    CHECK( magma_svinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_sresidualvec( ACSR, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    //nom0 = residual;

    // Jacobi setup
    CHECK( magma_sjacobisetup_diagscal( ACSR, &d, queue ));
    magma_s_solver_par jacobiiter_par;
    if ( solver_par->verbose > 0 ) {
        jacobiiter_par.maxiter = solver_par->verbose;
    }
    else {
        jacobiiter_par.maxiter = solver_par->maxiter;
    }
    k = jacobiiter_par.maxiter;
    

    
    CHECK( magma_svinit( &xkm2, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_svinit( &xkm1, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_svinit( &xk, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_svinit( &z, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_svinit( &c, Magma_DEV, b.num_rows, 1, c_zero, queue ));

    CHECK( magma_imalloc( &flag_t, b.num_rows ));
    CHECK( magma_imalloc( &flag_fp, b.num_rows ));
    
    if ( solver_par->verbose != 0 ) {
        // k iterations for startup
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_sjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // save in xkm2
        magma_scopyvector( dofs, x->dval, 1, xkm2.dval, 1, queue ); 
        // two times k iterations for computing the contraction constants
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_sjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // save in xkm1
        tempo1 = magma_sync_wtime( queue );
        magma_scopyvector( dofs, x->dval, 1, xkm1.dval, 1, queue );
        CHECK( magma_sjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // compute contraction constants
        magma_scopyvector( dofs, x->dval, 1, xk.dval, 1, queue );
        magma_sftjacobicontractions( xkm2, xkm1, xk, &z, &c, queue );
    }
    
    // Jacobi iterator
    do {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter = solver_par->numiter+jacobiiter_par.maxiter;
        CHECK( magma_sjacobispmvupdate(jacobiiter_par.maxiter, ACSR, r, b, d, x, queue ));
        if( solver_par->verbose != 0 ){
            magma_sftjacobiupdatecheck( delta, &xk, x, &z, c, flag_t, flag_fp, queue ); 
        }
        solver_par->spmv_count=solver_par->spmv_count+k;
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        if ( solver_par->verbose > 0 ) {
            CHECK(  magma_sresidualvec( ACSR, b, *x, &r, &residual, queue));
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) residual;
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) runtime;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    solver_par->runtime = (real_Double_t) runtime;
    CHECK(  magma_sresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->final_res = residual;

    if ( solver_par->init_res > solver_par->final_res )
        info = MAGMA_SUCCESS;
    else
        info = MAGMA_DIVERGENCE;
    
cleanup:
    magma_smfree( &r, queue );
    magma_smfree( &d, queue );
    magma_smfree( &ACSR, queue );
    magma_smfree( &xkm2, queue );
    magma_smfree( &xkm1, queue ); 
    magma_smfree( &xk, queue );
    magma_smfree( &z, queue ); 
    magma_smfree( &c, queue ); 
    magma_free( flag_t );
    magma_free( flag_fp );
    
    solver_par->info = info;
    return info;
}   /* magma_sftjacobi */
