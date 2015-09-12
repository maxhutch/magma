/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @author Hartwig Anzt

       @generated from zjacobidomainoverlap.cpp normal z -> s, Fri Sep 11 18:29:44 2015
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method allowing for
    domain overlap.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in]
    b           magma_s_matrix
                RHS b

    @param[in,out]
    x           magma_s_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgesv
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobidomainoverlap(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // some useful variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE,
                                                c_mone = MAGMA_S_NEG_ONE;
    magma_int_t dofs = A.num_rows*b.num_cols;
    //float nom0 = 0.0;
    // generate the domain overlap
    magma_int_t num_ind = 0;
    magma_index_t *indices={0}, *hindices={0};
    
    magma_s_matrix r={Magma_CSR}, d={Magma_CSR};
    magma_s_matrix hA={Magma_CSR};
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    real_Double_t tempo1, tempo2;
    float residual;
    CHECK( magma_sresidual( A, b, *x, &residual, queue ));
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;



    CHECK( magma_svinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_s_spmv( c_one, A, *x, c_zero, r, queue ));          // r = A x
    magma_saxpy(dofs,  c_mone, b.dval, 1, r.dval, 1);           // r = r - b
    //nom0 = magma_snrm2(dofs, r.dval, 1);                      // den = || r ||

    // Jacobi setup
    CHECK( magma_sjacobisetup_diagscal( A, &d, queue ));
    magma_s_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;
    

    CHECK( magma_index_malloc_cpu( &hindices, A.num_rows*10 ));
    CHECK( magma_index_malloc( &indices, A.num_rows*10 ));

    CHECK( magma_smtransfer( A, &hA, Magma_DEV, Magma_CPU, queue ));
    CHECK( magma_sdomainoverlap( hA.num_rows, &num_ind, hA.row, hA.col, hindices, queue ));
    /*num_ind = A.num_rows*5;
    for(magma_int_t i=0; i<num_ind; i++){
        hindices[i] = i%A.num_rows;
    }
*/
   
    CHECK( magma_index_malloc( &indices, num_ind ));
    magma_index_setvector( num_ind, hindices, 1, indices, 1 );
    
    
    
    tempo1 = magma_sync_wtime( queue );

    // Jacobi iterator
    CHECK( magma_sjacobispmvupdateselect(jacobiiter_par.maxiter, num_ind, indices,
                                                        A, r, b, d, r, x, queue ));
    
    //magma_sjacobispmvupdate(jacobiiter_par.maxiter, A, r, b, d, x, queue );

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    CHECK(  magma_sresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        info = MAGMA_SUCCESS;
    else
        info = MAGMA_DIVERGENCE;
    
    
cleanup:
    magma_free_cpu( hindices );
    magma_smfree( &r, queue );
    magma_smfree( &d, queue );
    magma_smfree(&hA, queue );
    magma_free( indices );
    
    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_sjacobi */
