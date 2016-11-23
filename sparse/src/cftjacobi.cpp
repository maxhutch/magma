/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Hartwig Anzt

       @generated from sparse/src/zftjacobi.cpp, normal z -> c, Sun Nov 20 20:20:45 2016
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
    A           magma_c_matrix
                input matrix M = D^(-1) * (L+U)
                
    @param[in]
    b           magma_c_matrix
                input RHS b
                
    @param[in,out]
    x           magma_c_matrix*
                iteration vector x

    @param[in,out]
    solver_par  magma_c_solver_par*
                solver parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_c
    ********************************************************************/

extern "C" magma_int_t
magma_cftjacobi(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // some useful variables
    real_Double_t tempo1, tempo2, runtime=0;
    float residual;
    
    // local variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magma_int_t dofs = A.num_cols;
    magma_int_t k = solver_par->verbose;
    
    magma_c_matrix xkm2 = {Magma_CSR};
    magma_c_matrix xkm1 = {Magma_CSR};
    magma_c_matrix xk = {Magma_CSR};
    magma_c_matrix z = {Magma_CSR};
    magma_c_matrix c = {Magma_CSR};
    magma_int_t *flag_t = NULL;
    magma_int_t *flag_fp = NULL;
    
    float delta = 0.9;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    magma_c_matrix r={Magma_CSR}, d={Magma_CSR}, ACSR={Magma_CSR};
    
    CHECK( magma_cmconvert(A, &ACSR, A.storage_type, Magma_CSR, queue ) );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    // solver setup
    CHECK( magma_cvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_cresidualvec( ACSR, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    //nom0 = residual;

    // Jacobi setup
    CHECK( magma_cjacobisetup_diagscal( ACSR, &d, queue ));
    magma_c_solver_par jacobiiter_par;
    if ( solver_par->verbose > 0 ) {
        jacobiiter_par.maxiter = solver_par->verbose;
    }
    else {
        jacobiiter_par.maxiter = solver_par->maxiter;
    }
    k = jacobiiter_par.maxiter;
    

    
    CHECK( magma_cvinit( &xkm2, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_cvinit( &xkm1, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_cvinit( &xk, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_cvinit( &z, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_cvinit( &c, Magma_DEV, b.num_rows, 1, c_zero, queue ));

    CHECK( magma_imalloc( &flag_t, b.num_rows ));
    CHECK( magma_imalloc( &flag_fp, b.num_rows ));
    
    if ( solver_par->verbose != 0 ) {
        // k iterations for startup
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_cjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // save in xkm2
        magma_ccopyvector( dofs, x->dval, 1, xkm2.dval, 1, queue ); 
        // two times k iterations for computing the contraction constants
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_cjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // save in xkm1
        tempo1 = magma_sync_wtime( queue );
        magma_ccopyvector( dofs, x->dval, 1, xkm1.dval, 1, queue );
        CHECK( magma_cjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // compute contraction constants
        magma_ccopyvector( dofs, x->dval, 1, xk.dval, 1, queue );
        magma_cftjacobicontractions( xkm2, xkm1, xk, &z, &c, queue );
    }
    
    // Jacobi iterator
    do {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter = solver_par->numiter+jacobiiter_par.maxiter;
        CHECK( magma_cjacobispmvupdate(jacobiiter_par.maxiter, ACSR, r, b, d, x, queue ));
        if( solver_par->verbose != 0 ){
            magma_cftjacobiupdatecheck( delta, &xk, x, &z, c, flag_t, flag_fp, queue ); 
        }
        solver_par->spmv_count=solver_par->spmv_count+k;
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        if ( solver_par->verbose > 0 ) {
            CHECK(  magma_cresidualvec( ACSR, b, *x, &r, &residual, queue));
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) residual;
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) runtime;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    solver_par->runtime = (real_Double_t) runtime;
    CHECK(  magma_cresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->final_res = residual;

    if ( solver_par->init_res > solver_par->final_res )
        info = MAGMA_SUCCESS;
    else
        info = MAGMA_DIVERGENCE;
    
cleanup:
    magma_cmfree( &r, queue );
    magma_cmfree( &d, queue );
    magma_cmfree( &ACSR, queue );
    magma_cmfree( &xkm2, queue );
    magma_cmfree( &xkm1, queue ); 
    magma_cmfree( &xk, queue );
    magma_cmfree( &z, queue ); 
    magma_cmfree( &c, queue ); 
    magma_free( flag_t );
    magma_free( flag_fp );
    
    solver_par->info = info;
    return info;
}   /* magma_cftjacobi */
