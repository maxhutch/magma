/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/src/magma_z_precond_wrapper.cpp, normal z -> c, Sun Nov 20 20:20:46 2016
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
    A           magma_c_matrix
                sparse matrix A

    @param[in]
    b           magma_c_matrix
                input vector b

    @param[in]
    x           magma_c_matrix*
                output vector x

    @param[in,out]
    precond     magma_c_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_precond(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set up precond parameters as solver parameters
    magma_c_solver_par psolver_par;
    psolver_par.rtol = precond->rtol;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;
    magma_c_preconditioner pprecond;
    pprecond.solver = Magma_NONE;
    pprecond.maxiter = 3;

    switch( precond->solver ) {
        case  Magma_CG:
                CHECK( magma_ccg_res( A, b, x, &psolver_par, queue )); break;
        case  Magma_BICGSTAB:
                CHECK( magma_cbicgstab( A, b, x, &psolver_par, queue )); break;
        case  Magma_GMRES:
                CHECK( magma_cfgmres( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_JACOBI:
                CHECK( magma_cjacobi( A, b, x, &psolver_par, queue )); break;
        case  Magma_BAITER:
                CHECK( magma_cbaiter( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_IDR:
                CHECK( magma_cidr( A, b, x, &psolver_par, queue )); break;
        case  Magma_CGS:
                CHECK( magma_ccgs( A, b, x, &psolver_par, queue )); break;
        case  Magma_QMR:
                CHECK( magma_cqmr( A, b, x, &psolver_par, queue )); break;
        case  Magma_TFQMR:
                CHECK( magma_ctfqmr( A, b, x, &psolver_par, queue )); break;
        case  Magma_BAITERO:
                CHECK( magma_cbaiter_overlap( A, b, x, &psolver_par, &pprecond, queue )); break;
        default:
                CHECK( magma_ccg_res( A, b, x, &psolver_par, queue )); break;
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
    A           magma_c_matrix
                sparse matrix M

    @param[in]
    b           magma_c_matrix
                input vector y
    
    @param[in]
    solver      magma_c_solver_par
                solver structure using the preconditioner
                
    @param[in,out]
    precond     magma_c_preconditioner
                preconditioner
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_precondsetup(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_solver_par *solver,
    magma_c_preconditioner *precond,
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
        info = magma_cjacobisetup_diagscal( A, &(precond->d), queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //info = magma_cpastixsetup( A, b, precond, queue );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    // ILU and related
    else if ( precond->solver == Magma_ILU ) {
        if ( precond->trisolver == Magma_ISAI ||
             precond->trisolver == Magma_JACOBI ||
             precond->trisolver == Magma_VBJACOBI ){
            info = magma_ccumilusetup( A, precond, queue );
            info = magma_ciluisaisetup( A, b, precond, queue );
        } else {
            info = magma_ccumilusetup( A, precond, queue );
        }
    }
    else if ( precond->solver == Magma_PARILU ) {
        info = magma_cparilusetup( A, b, precond, queue );
        if ( precond->trisolver == Magma_ISAI ||
             precond->trisolver == Magma_JACOBI ||
             precond->trisolver == Magma_VBJACOBI ){
            info = magma_ciluisaisetup( A, b, precond, queue );
        }
    }
    else if ( precond->solver == Magma_PARILUT ) {
        #ifdef _OPENMP
            info = magma_cparilutsetup( A, b, precond, queue );
            if ( precond->trisolver == Magma_ISAI  ||
                 precond->trisolver == Magma_JACOBI ||
                 precond->trisolver == Magma_VBJACOBI ){
                info = magma_ciluisaisetup( A, b, precond, queue );
            }
            precond->solver = Magma_PARILU; // handle as PARILU
        #else
            printf( "error: preconditioner requires OpenMP.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        #endif
    }
    else if ( precond->solver == Magma_CUSTOMILU ) {
        info = magma_ccustomilusetup( A, b, precond, queue );
        precond->solver = Magma_PARILU; // handle as PARILU
    }
    // symmetric: Cholesky variant
    else if ( precond->solver == Magma_ICC ) {
        if ( precond->trisolver == Magma_ISAI  ||
             precond->trisolver == Magma_JACOBI ||
             precond->trisolver == Magma_VBJACOBI ){
            info = magma_ccumiccsetup( A, precond, queue );
            info = magma_cicisaisetup( A, b, precond, queue );
        } else {
            info = magma_ccumiccsetup( A, precond, queue );
        }
    }
    else if ( precond->solver == Magma_PARIC ) {
        info = magma_cparicsetup( A, b, precond, queue );
    }
    else if ( precond->solver == Magma_PARICT ) {
        #ifdef _OPENMP
            info = magma_cparilutsetup( A, b, precond, queue );
            precond->solver = Magma_PARILU; // handle as PARIC
            precond->trisolver = Magma_CUSOLVE; // for now only allow cusolve
            printf( "%% warning: only PARILUT supported.\n" );
        #else
            printf( "error: preconditioner requires OpenMP.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        #endif
    }
    else if ( precond->solver == Magma_CUSTOMIC ) {
        info = magma_ccustomicsetup( A, b, precond, queue );
        precond->solver = Magma_PARIC; // handle as PARIC
    }
    // none case
    else if ( precond->solver == Magma_NONE ) {
        info = MAGMA_SUCCESS;
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    if( 
        ( solver->solver == Magma_PQMR  || 
          solver->solver == Magma_PQMRMERGE  || 
          solver->solver == Magma_PBICG ||
          solver->solver == Magma_LSQR ) &&
        ( precond->solver == Magma_ILU      || 
            precond->solver == Magma_PARILU   || 
            precond->solver == Magma_ICC    || 
            precond->solver == Magma_PARIC ) ) {  // also prepare the transpose
        info = magma_ccumilusetup_transpose( A, precond, queue );
        if( info == 0 && 
            ( precond->trisolver == Magma_ISAI  ||
              precond->trisolver == Magma_JACOBI ||
              precond->trisolver == Magma_VBJACOBI ) )
        {
                // simple solution: copy
                info = info + magma_cmtranspose( precond->LD, &precond->LDT, queue );
                info = info + magma_cmtranspose( precond->UD, &precond->UDT, queue );
                // info = magma_ciluisaisetup_t( A, b, precond, queue );
        }
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->setuptime = tempo2-tempo1;
    
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
    A           magma_c_matrix
                sparse matrix A

    @param[in]
    b           magma_c_matrix
                input vector b

    @param[in,out]
    x           magma_c_matrix*
                output vector x

    @param[in]
    precond     magma_c_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_applyprecond(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix tmp={Magma_CSR};

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_cjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //CHECK( magma_capplypastix( b, x, precond, queue ));
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        CHECK( magma_cvinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_C_ZERO, queue ));
    }
    else if ( precond->solver == Magma_ICC ) {
        CHECK( magma_cvinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_C_ZERO, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_cmfree( &tmp, queue );
    //magmablasSetKernelStream( orig_queue );
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
    trans       magma_trans_t
                mode of the preconditioner: MagmaTrans or MagmaNoTrans
                
    @param[in]
    A           magma_c_matrix
                sparse matrix A

    @param[in]
    b           magma_c_matrix
                input vector b

    @param[in,out]
    x           magma_c_matrix*
                output vector x

    @param[in]
    precond     magma_c_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_applyprecond_left(
    magma_trans_t trans,
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
        //Chronometry
    real_Double_t tempo1, tempo2;
    
    tempo1 = magma_sync_wtime( queue );

    magma_copts zopts;
    zopts.solver_par.solver = precond->trisolver;
    zopts.solver_par.maxiter = precond->maxiter;
    zopts.solver_par.verbose = 0;
    zopts.solver_par.version = 0;
    zopts.solver_par.restart = 50;
    zopts.solver_par.atol = 1e-16;
    zopts.solver_par.rtol = 1e-10;
    
    if( trans == MagmaNoTrans ) {
        if ( precond->solver == Magma_JACOBI ) {
            CHECK( magma_cjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumilu_l( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_SPTRSV ) ){
            // CHECK( magma_csptrsv( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_PARIC ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumicc_l( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_PARIC ) && 
                  ( precond->trisolver == Magma_ISAI ||
                    precond->trisolver == Magma_JACOBI ||
                    precond->trisolver == Magma_VBJACOBI ) ){
            CHECK( magma_cisai_l( b, x, precond, queue ) );
            // magma_c_spmv( MAGMA_C_ONE, precond->L, b,MAGMA_C_ZERO, *x, queue ); // SPAI
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_ISAI ||
                    precond->trisolver == Magma_JACOBI ||
                    precond->trisolver == Magma_VBJACOBI ) ){
            CHECK( magma_cisai_l( b, x, precond, queue ) );
            // magma_c_spmv( MAGMA_C_ONE, precond->L, b,MAGMA_C_ZERO, *x, queue ); // SPAI
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) ){
            magma_c_solver( precond->L, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_capplycustomprecond_l( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED; 
        }
    } else if ( trans == MagmaTrans ){
        if ( precond->solver == Magma_JACOBI ) {
            CHECK( magma_cjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumilu_l_transpose( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_PARIC ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumicc_l( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_ISAI ||
                    precond->trisolver == Magma_JACOBI ||
                    precond->trisolver == Magma_VBJACOBI ) ){
            CHECK( magma_cisai_l_t( b, x, precond, queue ) );
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) ){
            magma_c_solver( precond->L, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_capplycustomprecond_l( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED; 
        }
    } else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->runtime += tempo2-tempo1;
    
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
    trans       magma_trans_t
                mode of the preconditioner: MagmaTrans or MagmaNoTrans

    @param[in]
    A           magma_c_matrix
                sparse matrix A

    @param[in]
    b           magma_c_matrix
                input vector b

    @param[in,out]
    x           magma_c_matrix*
                output vector x

    @param[in]
    precond     magma_c_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_applyprecond_right(
    magma_trans_t trans,
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
        //Chronometry
    real_Double_t tempo1, tempo2;
    
    tempo1 = magma_sync_wtime( queue );
    
    magma_copts zopts;
    zopts.solver_par.solver = precond->trisolver;
    zopts.solver_par.maxiter = precond->maxiter;
    zopts.solver_par.verbose = 0;
    zopts.solver_par.version = 0;
    zopts.solver_par.restart = 50;
    zopts.solver_par.atol = 1e-16;
    zopts.solver_par.rtol = 1e-10;
    
    if( trans == MagmaNoTrans ) {
        if ( precond->solver == Magma_JACOBI ) {
            magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );    // x = b
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ) {
            CHECK( magma_capplycumilu_r( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_PARIC ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumicc_r( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_PARIC ) && 
                  ( precond->trisolver == Magma_ISAI ||
                    precond->trisolver == Magma_JACOBI ||
                    precond->trisolver == Magma_VBJACOBI ) ){
            CHECK( magma_cisai_r( b, x, precond, queue ) );
            // magma_c_spmv( MAGMA_C_ONE, precond->L, b,MAGMA_C_ZERO, *x, queue ); // SPAI
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_ISAI ||
                    precond->trisolver == Magma_JACOBI ||
                    precond->trisolver == Magma_VBJACOBI ) ){
            CHECK( magma_cisai_r( b, x, precond, queue ) );
            // magma_c_spmv( MAGMA_C_ONE, precond->L, b,MAGMA_C_ZERO, *x, queue ); // SPAI
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) ){
            magma_c_solver( precond->U, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
      //  else if ( precond->solver == Magma_ISAI ) {
      //      CHECK( magma_cisai_r( b, x, precond, queue ) );
      //      // magma_c_spmv( MAGMA_C_ONE, precond->U, b,MAGMA_C_ZERO, *x, queue ); // SPAI
      //  }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_capplycustomprecond_r( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    } else if ( trans == MagmaTrans ){
        if ( precond->solver == Magma_JACOBI ) {
            magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );    // x = b
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumilu_r_transpose( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ICC ||
                    precond->solver == Magma_PARIC ) && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_capplycumicc_r( b, x, precond, queue ));
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) && 
                  ( precond->trisolver == Magma_ISAI ||
                    precond->trisolver == Magma_JACOBI ||
                    precond->trisolver == Magma_VBJACOBI ) ){
            CHECK( magma_cisai_r_t( b, x, precond, queue ) );
            // magma_c_spmv( MAGMA_C_ONE, precond->U, b,MAGMA_C_ZERO, *x, queue ); // SPAI
        }
        else if ( ( precond->solver == Magma_ILU ||
                    precond->solver == Magma_PARILU ) ){
            magma_c_solver( precond->U, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_ccopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_capplycustomprecond_r( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->runtime += tempo2-tempo1;
        
cleanup:
    return info;
}
