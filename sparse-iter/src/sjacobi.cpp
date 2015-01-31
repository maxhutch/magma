/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Hartwig Anzt 

       @generated from zjacobi.cpp normal z -> s, Fri Jan 30 19:00:30 2015
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method.

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                input matrix A

    @param[in]
    b           magma_s_vector
                RHS b

    @param[in,out]
    x           magma_s_vector*
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
magma_sjacobi(
    magma_s_sparse_matrix A, 
    magma_s_vector b, 
    magma_s_vector *x,  
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    real_Double_t tempo1, tempo2;
    float residual;
    magma_sresidual( A, b, *x, &residual, queue );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // some useful variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE, 
                                                c_mone = MAGMA_S_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    float nom0;


    magma_s_sparse_matrix M;
    magma_s_vector c, r;
    magma_s_vinit( &r, Magma_DEV, dofs, c_zero, queue );
    magma_s_spmv( c_one, A, *x, c_zero, r, queue );                  // r = A x
    magma_saxpy(dofs,  c_mone, b.dval, 1, r.dval, 1);           // r = r - b
    nom0 = magma_snrm2(dofs, r.dval, 1);                      // den = || r ||

    // Jacobi setup
    magma_sjacobisetup( A, b, &M, &c, queue );
    magma_s_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;

    tempo1 = magma_sync_wtime( queue );

    // Jacobi iterator
    magma_sjacobiiter( M, c, x, &jacobiiter_par, queue ); 

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_sresidual( A, b, *x, &residual, queue );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        solver_par->info = MAGMA_SUCCESS;
    else
        solver_par->info = MAGMA_DIVERGENCE;

    magma_s_mfree( &M, queue );
    magma_s_vfree( &c, queue );
    magma_s_vfree( &r, queue );

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_sjacobi */






/**
    Purpose
    -------

    Prepares the Matrix M for the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    It returns the preconditioner Matrix M and a vector d
    containing the diagonal elements. 

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                input matrix A

    @param[in]
    M           magma_s_sparse_matrix*
                M = D^(-1) * (L+U)

    @param[in,out]
    d           magma_s_vector*
                vector with diagonal elements of A
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobisetup_matrix(
    magma_s_sparse_matrix A, 
    magma_s_sparse_matrix *M, magma_s_vector *d,
    magma_queue_t queue )
{
    magma_int_t i;

    magma_s_sparse_matrix A_h1, A_h2, B, C;
    magma_s_vector diag;
    magma_s_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_S_ZERO, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_s_mtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue );
        magma_s_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue );
    }
    else {
        magma_s_mtransfer( A, &B, A.memory_location, Magma_CPU, queue );
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = B.val[i];
                if ( MAGMA_S_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ) {
            B.val[i] = B.val[i] / diag.val[rowindex];
            if ( B.dcol[i]==rowindex ) {
                B.val[i] = MAGMA_S_MAKE( 0., 0. );
            }
        }
    }
    magma_s_csr_compressor(&B.val, &B.drow, &B.dcol, 
                           &C.val, &C.drow, &C.dcol, &B.num_rows, queue );  
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.drow[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if ( A.storage_type != Magma_CSR) {
        magma_s_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type, queue );
        magma_s_mtransfer( A_h2, M, Magma_CPU, A.memory_location, queue );
    }
    else {
        magma_s_mtransfer( C, M, Magma_CPU, A.memory_location, queue );
    }    
    magma_s_vtransfer( diag, d, Magma_CPU, A.memory_location, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_s_mfree( &A_h1, queue );
        magma_s_mfree( &A_h2, queue );   
    }
    magma_s_mfree( &B, queue );
    magma_s_mfree( &C, queue ); 

    magma_s_vfree( &diag, queue );
 
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------


    It returns a vector d
    containing the inverse diagonal elements. 

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                input matrix A

    @param[in,out]
    d           magma_s_vector*
                vector with diagonal elements
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobisetup_diagscal(
    magma_s_sparse_matrix A, magma_s_vector *d,
    magma_queue_t queue )
{
    magma_int_t i;

    magma_s_sparse_matrix A_h1, B;
    magma_s_vector diag;
    magma_s_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_S_ZERO, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_s_mtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue );
        magma_s_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue );
    }
    else {
        magma_s_mtransfer( A, &B, A.memory_location, Magma_CPU, queue );
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = 1.0/B.val[i];
                if ( MAGMA_S_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
    }
    magma_s_vtransfer( diag, d, Magma_CPU, A.memory_location, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_s_mfree( &A_h1, queue );
    }
    magma_s_mfree( &B, queue );
    magma_s_vfree( &diag, queue );
 
    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c

    Arguments
    ---------

    @param[in]
    b           magma_s_vector
                RHS b

    @param[in]
    d           magma_s_vector
                vector with diagonal entries

    @param[in]
    c           magma_s_vector*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobisetup_vector(
    magma_s_vector b, magma_s_vector d, 
    magma_s_vector *c,
    magma_queue_t queue )
{
    if ( b.memory_location == Magma_CPU ) {
        magma_s_vector diag, c_t, b_h;
        magma_s_vinit( &c_t, Magma_CPU, b.num_rows, MAGMA_S_ZERO, queue );

        magma_s_vtransfer( b, &b_h, b.memory_location, Magma_CPU, queue );
        magma_s_vtransfer( d, &diag, b.memory_location, Magma_CPU, queue );

        for( magma_int_t rowindex=0; rowindex<b.num_rows; rowindex++ ) {   
            c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

        }  
        magma_s_vtransfer( c_t, c, Magma_CPU, b.memory_location, queue ); 

        magma_s_vfree( &diag, queue );
        magma_s_vfree( &c_t, queue );
        magma_s_vfree( &b_h, queue );

        return MAGMA_SUCCESS;
    }
    else if ( b.memory_location == Magma_DEV ) {
        // fill vector
        magma_s_vector tmp;
        magma_s_vinit( &tmp, Magma_DEV, b.num_rows, MAGMA_S_ZERO, queue );
        magma_sjacobisetup_vector_gpu( 
                    b.num_rows, b, d, *c, &tmp, queue );
        magma_s_vfree( &tmp, queue );
        return MAGMA_SUCCESS;
    }

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                input matrix A

    @param[in]
    b           magma_s_vector
                RHS b

    @param[in]
    M           magma_s_sparse_matrix*
                M = D^(-1) * (L+U)

    @param[in]
    c           magma_s_vector*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobisetup(
    magma_s_sparse_matrix A, magma_s_vector b, 
    magma_s_sparse_matrix *M, magma_s_vector *c,
    magma_queue_t queue )
{
    magma_int_t i;

    magma_s_sparse_matrix A_h1, A_h2, B, C;
    magma_s_vector diag, c_t, b_h;
    magma_s_vinit( &c_t, Magma_CPU, A.num_rows, MAGMA_S_ZERO, queue );
    magma_s_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_S_ZERO, queue );
    magma_s_vtransfer( b, &b_h, A.memory_location, Magma_CPU, queue );

    if ( A.storage_type != Magma_CSR ) {
        magma_s_mtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue );
        magma_s_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue );
    }
    else {
        magma_s_mtransfer( A, &B, A.memory_location, Magma_CPU, queue );
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = B.val[i];
                if ( MAGMA_S_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                               (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ) {
            B.val[i] = B.val[i] / diag.val[rowindex];
            if ( B.dcol[i]==rowindex ) {
                B.val[i] = MAGMA_S_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

    }

    magma_s_csr_compressor(&B.val, &B.drow, &B.dcol, 
                           &C.val, &C.drow, &C.dcol, &B.num_rows, queue );  

    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.drow[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if ( A.storage_type != Magma_CSR) {
        A_h2.alignment = A.alignment;
        A_h2.blocksize = A.blocksize;
        magma_s_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type, queue );
        magma_s_mtransfer( A_h2, M, Magma_CPU, A.memory_location, queue );
    }
    else {
        magma_s_mtransfer( C, M, Magma_CPU, A.memory_location, queue );
    }     
    magma_s_vtransfer( c_t, c, Magma_CPU, A.memory_location, queue );

    if ( A.storage_type != Magma_CSR) {
        magma_s_mfree( &A_h1, queue );
        magma_s_mfree( &A_h2, queue );   
    }   
    magma_s_mfree( &B, queue );
    magma_s_mfree( &C, queue );  
    magma_s_vfree( &diag, queue );
    magma_s_vfree( &c_t, queue );
    magma_s_vfree( &b_h, queue );

    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    M           magma_s_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param[in]
    c           magma_s_vector
                c = D^(-1) * b

    @param[in,out]
    x           magma_s_vector*
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
magma_sjacobiiter(
    magma_s_sparse_matrix M, magma_s_vector c, magma_s_vector *x,  
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE, 
                                            c_mone = MAGMA_S_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_s_vector t, swap;
    magma_s_vinit( &t, Magma_DEV, dofs, c_zero, queue );


    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        magma_s_spmv( c_mone, M, *x, c_zero, t, queue );                // t = - M * x
        magma_saxpy( dofs, c_one , c.dval, 1 , t.dval, 1 );        // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = t;
        t = swap;        
        //magma_scopy( dofs, t.dval, 1 , x->dval, 1 );               // x = t
    }

    magma_s_vfree( &t, queue );

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_sjacobiiter */



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    M           magma_s_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param[in]
    c           magma_s_vector
                c = D^(-1) * b

    @param[in,out]
    x           magma_s_vector*
                iteration vector x

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in]
    solver_par  magma_s_precond_par*
                precond parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_sjacobiiter_precond(
    magma_s_sparse_matrix M, magma_s_vector *x, 
    magma_s_solver_par *solver_par, magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE, 
                                            c_mone = MAGMA_S_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_int_t num_vecs = x->num_rows / dofs;
    magma_s_vector swap;

    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        magma_s_spmv( c_mone, M, *x, c_zero, precond->work2, queue );   // t = - M * x

        magma_saxpy( num_vecs*dofs, c_one , 
                precond->work1.dval, 1 , precond->work2.dval, 1 ); // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = precond->work2;
        precond->work2 = swap;        
        //magma_scopy( dofs, t.dval, 1 , x->dval, 1 );               // x = t
    }

    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_sjacobiiter */
