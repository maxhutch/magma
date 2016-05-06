/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/magma_z_precond_wrapper.cpp normal z -> s, Mon May  2 23:31:04 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"




/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                sparse matrix A

    @param[in]
    b           magma_s_matrix
                input vector b

    @param[in]
    x           magma_s_matrix*
                output vector x

    @param[in,out]
    precond     magma_s_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_s_precond(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set up precond parameters as solver parameters
    magma_s_solver_par psolver_par;
    psolver_par.rtol = precond->rtol;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;
    magma_s_preconditioner pprecond;
    pprecond.solver = Magma_NONE;
    pprecond.maxiter = 3;

    switch( precond->solver ) {
        case  Magma_CG:
                CHECK( magma_scg_res( A, b, x, &psolver_par, queue )); break;
        case  Magma_BICGSTAB:
                CHECK( magma_sbicgstab( A, b, x, &psolver_par, queue )); break;
        case  Magma_GMRES:
                CHECK( magma_sfgmres( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_JACOBI:
                CHECK( magma_sjacobi( A, b, x, &psolver_par, queue )); break;
        case  Magma_BAITER:
                CHECK( magma_sbaiter( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_IDR:
                CHECK( magma_sidr( A, b, x, &psolver_par, queue )); break;
        case  Magma_CGS:
                CHECK( magma_scgs( A, b, x, &psolver_par, queue )); break;
        case  Magma_QMR:
                CHECK( magma_sqmr( A, b, x, &psolver_par, queue )); break;
        case  Magma_TFQMR:
                CHECK( magma_stfqmr( A, b, x, &psolver_par, queue )); break;
        case  Magma_BAITERO:
                CHECK( magma_sbaiter_overlap( A, b, x, &psolver_par, &pprecond, queue )); break;
        default:
                CHECK( magma_scg_res( A, b, x, &psolver_par, queue )); break;
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    For a given input matrix M and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is preprocessed.
    E.g. for Jacobi: the scaling-vetor, for ILU the factorization.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                sparse matrix M

    @param[in]
    b           magma_s_matrix
                input vector y

    @param[in,out]
    precond     magma_s_preconditioner
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_s_precondsetup(
    magma_s_matrix A, magma_s_matrix b,
    magma_s_solver_par *solver,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    
    tempo1 = magma_sync_wtime( queue );
    
    if( A.num_rows != A.num_cols ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no preconditioner.\n");
        precond->solver = Magma_NONE;
    } 
    
    if ( precond->solver == Magma_JACOBI ) {
        info = magma_sjacobisetup_diagscal( A, &(precond->d), queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //info = magma_spastixsetup( A, b, precond, queue );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        info = magma_scumilusetup( A, precond, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        info = magma_scumiccsetup( A, precond, queue );
    }
    else if ( precond->solver == Magma_AICC ) {
        info = magma_sitericsetup( A, b, precond, queue );
    }
    else if ( precond->solver == Magma_AICT ) {
        #ifdef _OPENMP
            info = magma_siterictsetup( A, b, precond, queue );
            precond->solver = Magma_AICC; // handle as AICC
        #else
            printf( "error: preconditioner requires OpenMP.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        #endif
    }
    else if ( precond->solver == Magma_AILU ) {
        info = magma_siterilusetup( A, b, precond, queue );
    }
    else if ( precond->solver == Magma_CUSTOMIC ) {
        info = magma_scustomicsetup( A, b, precond, queue );
        precond->solver = Magma_AICC; // handle as AICC
    }
    else if ( precond->solver == Magma_CUSTOMILU ) {
        info = magma_scustomilusetup( A, b, precond, queue );
        precond->solver = Magma_AILU; // handle as AILU
    }
    else if ( precond->solver == Magma_NONE ) {
        info = MAGMA_SUCCESS;
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    if( 
        ( solver->solver == Magma_PQMR  || 
          solver->solver == Magma_PBICG ||
          solver->solver == Magma_LSQR ) &&
        ( precond->solver == Magma_ILU      || 
            precond->solver == Magma_AILU   || 
            precond->solver == Magma_ICC    || 
            precond->solver == Magma_AICC ) ) {  // also prepare the transpose
        info = magma_scumilusetup_transpose( A, precond, queue );
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->setuptime += tempo2-tempo1;
    
    return info;
}



/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is applied.
    E.g. for Jacobi: the scaling-vetor, for ILU the triangular solves.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                sparse matrix A

    @param[in]
    b           magma_s_matrix
                input vector b

    @param[in,out]
    x           magma_s_matrix*
                output vector x

    @param[in]
    precond     magma_s_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_s_applyprecond(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_s_matrix tmp={Magma_CSR};

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_sjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //CHECK( magma_sapplypastix( b, x, precond, queue ));
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        CHECK( magma_svinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_S_ZERO, queue ));
    }
    else if ( precond->solver == Magma_ICC ) {
        CHECK( magma_svinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_S_ZERO, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_smfree( &tmp, queue );
    return info;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective left preconditioner
    is applied.
    E.g. for Jacobi: the scaling-vetor, for ILU the left triangular solve.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                sparse matrix A

    @param[in]
    b           magma_s_matrix
                input vector b

    @param[in,out]
    x           magma_s_matrix*
                output vector x

    @param[in]
    precond     magma_s_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_s_applyprecond_left(
    magma_trans_t trans,
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if( trans == MagmaNoTrans ) {
        if ( precond->solver == Magma_JACOBI ) {
            CHECK( magma_sjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
            CHECK( magma_sapplycumilu_l( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->L, b, precond->d, precond->work1, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_AICC ) && precond->maxiter >= 50 )  {
            CHECK( magma_sapplycumicc_l( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_AICC ) && precond->maxiter < 50 )  {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->L, b, precond->d, precond->work1, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_sapplycustomprecond_l( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED; 
        }
    } else if ( trans == MagmaTrans ){
        if ( precond->solver == Magma_JACOBI ) {
            CHECK( magma_sjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
            CHECK( magma_sapplycumilu_l_transpose( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->LT, b, precond->d, precond->work1, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_AICC ) && precond->maxiter >= 50 )  {
            CHECK( magma_sapplycumicc_l( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_AICC ) && precond->maxiter < 50 )  {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->LT, b, precond->d, precond->work1, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_sapplycustomprecond_l( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED; 
        }
    } else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
cleanup:
    return info;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective right-preconditioner
    is applied.
    E.g. for Jacobi: the scaling-vetor, for ILU the right triangular solve.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                sparse matrix A

    @param[in]
    b           magma_s_matrix
                input vector b

    @param[in,out]
    x           magma_s_matrix*
                output vector x

    @param[in]
    precond     magma_s_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_s_applyprecond_right(
    magma_trans_t trans,
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    if( trans == MagmaNoTrans ) {
        if ( precond->solver == Magma_JACOBI ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );    // x = b
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
            CHECK( magma_sapplycumilu_r( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->U, b, precond->d2, precond->work2, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate_bw(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
        }
    
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_AICC ) && precond->maxiter >= 50 ) {
            CHECK( magma_sapplycumicc_r( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                   precond->solver == Magma_AICC ) && precond->maxiter < 50 ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->U, b, precond->d2, precond->work2, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate_bw(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_sapplycustomprecond_r( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    } else if ( trans == MagmaTrans ){
        if ( precond->solver == Magma_JACOBI ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );    // x = b
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
            CHECK( magma_sapplycumilu_r_transpose( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->UT, b, precond->d2, precond->work2, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate_bw(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
        }
    
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_AICC ) && precond->maxiter >= 50 ) {
            CHECK( magma_sapplycumicc_r( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                   precond->solver == Magma_AICC ) && precond->maxiter < 50 ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols, queue );
            magma_s_solver_par solver_par;
            solver_par.maxiter = precond->maxiter;
            magma_sjacobiiter_sys( precond->UT, b, precond->d2, precond->work2, x, &solver_par, queue );
            // CHECK( magma_sjacobispmvupdate_bw(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_scopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_sapplycustomprecond_r( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
        
cleanup:
    return info;
}
