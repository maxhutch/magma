/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Hartwig Anzt 

       @generated from zjacobi.cpp normal z -> s, Fri Jul 18 17:34:29 2014
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

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

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    b           magma_s_vector
                RHS b

    @param
    x           magma_s_vector*
                solution approximation

    @param
    solver_par  magma_s_solver_par*
                solver parameters

    @ingroup magmasparse_sgesv
    ********************************************************************/

magma_int_t
magma_sjacobi( magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector *x,  
           magma_s_solver_par *solver_par )
{
    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = 0;

    real_Double_t tempo1, tempo2;
    float residual;
    magma_sresidual( A, b, *x, &residual );
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
    magma_s_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_s_spmv( c_one, A, *x, c_zero, r );                  // r = A x
    magma_saxpy(dofs,  c_mone, b.val, 1, r.val, 1);           // r = r - b
    nom0 = magma_snrm2(dofs, r.val, 1);                      // den = || r ||

    // Jacobi setup
    magma_sjacobisetup( A, b, &M, &c );
    magma_s_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;

    magma_device_sync(); tempo1=magma_wtime();

    // Jacobi iterator
    magma_sjacobiiter( M, c, x, &jacobiiter_par ); 

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_sresidual( A, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    magma_s_mfree( &M );
    magma_s_vfree( &c );
    magma_s_vfree( &r );

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

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    b           magma_s_vector
                RHS b

    @param
    m           magma_s_sparse_matrix*
                M = D^(-1) * (L+U)

    @param
    d           magma_s_vector*
                vector with diagonal elements

    @ingroup magmasparse_s
    ********************************************************************/

magma_int_t
magma_sjacobisetup_matrix( magma_s_sparse_matrix A, magma_s_vector b, 
                    magma_s_sparse_matrix *M, magma_s_vector *d ){

    magma_int_t i;

    magma_s_sparse_matrix A_h1, A_h2, B, C;
    magma_s_vector diag, b_h;
    magma_s_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_S_ZERO );
    magma_s_vtransfer( b, &b_h, A.memory_location, Magma_CPU);

    if( A.storage_type != Magma_CSR){
        magma_s_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_s_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_s_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_S_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_S_MAKE( 0., 0. );
            }
        }
    }
    magma_s_csr_compressor(&B.val, &B.row, &B.col, 
                           &C.val, &C.row, &C.col, &B.num_rows );  
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.row[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if( A.storage_type != Magma_CSR){
        magma_s_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
        magma_s_mtransfer( A_h2, M, Magma_CPU, A.memory_location);
    }
    else{
        magma_s_mtransfer( C, M, Magma_CPU, A.memory_location);
    }    
    magma_s_vtransfer( diag, d, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_s_mfree( &A_h1 );
        magma_s_mfree( &A_h2 );   
    }
    magma_s_mfree( &B );
    magma_s_mfree( &C ); 

    magma_s_vfree( &diag);
    magma_s_vfree( &b_h);
 
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------


    It returns a vector d
    containing the inverse diagonal elements. 

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    d           magma_s_vector*
                vector with diagonal elements

    @ingroup magmasparse_s
    ********************************************************************/

magma_int_t
magma_sjacobisetup_diagscal( magma_s_sparse_matrix A, magma_s_vector *d ){

    magma_int_t i;

    magma_s_sparse_matrix A_h1, B;
    magma_s_vector diag;
    magma_s_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_S_ZERO );

    if( A.storage_type != Magma_CSR){
        magma_s_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_s_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_s_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = 1.0/B.val[i];
                if( MAGMA_S_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
    }
    magma_s_vtransfer( diag, d, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_s_mfree( &A_h1 );
    }
    magma_s_mfree( &B );
    magma_s_vfree( &diag);
 
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

    @param
    b           magma_s_vector
                RHS b

    @param
    d           magma_s_vector
                vector with diagonal entries

    @param
    c           magma_s_vector*
                c = D^(-1) * b

    @ingroup magmasparse_s
    ********************************************************************/

magma_int_t
magma_sjacobisetup_vector( magma_s_vector b, magma_s_vector d, 
                           magma_s_vector *c ){

    if( b.memory_location == Magma_CPU ){
        magma_s_vector diag, c_t, b_h;
        magma_s_vinit( &c_t, Magma_CPU, b.num_rows, MAGMA_S_ZERO );

        magma_s_vtransfer( b, &b_h, b.memory_location, Magma_CPU);
        magma_s_vtransfer( d, &diag, b.memory_location, Magma_CPU);

        for( magma_int_t rowindex=0; rowindex<b.num_rows; rowindex++ ){   
            c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

        }  
        magma_s_vtransfer( c_t, c, Magma_CPU, b.memory_location); 

        magma_s_vfree( &diag);
        magma_s_vfree( &c_t);
        magma_s_vfree( &b_h);

        return MAGMA_SUCCESS;
    }
    else if( b.memory_location == Magma_DEV ){
        // fill vector
        magma_sjacobisetup_vector_gpu( b.num_rows, b.val, d.val, c->val );
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

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    b           magma_s_vector
                RHS b

    @param
    m           magma_s_sparse_matrix*
                M = D^(-1) * (L+U)

    @param
    c           magma_s_vector*
                c = D^(-1) * b

    @ingroup magmasparse_s
    ********************************************************************/

magma_int_t
magma_sjacobisetup( magma_s_sparse_matrix A, magma_s_vector b, 
                    magma_s_sparse_matrix *M, magma_s_vector *c ){

    magma_int_t i;

    magma_s_sparse_matrix A_h1, A_h2, B, C;
    magma_s_vector diag, c_t, b_h;
    magma_s_vinit( &c_t, Magma_CPU, A.num_rows, MAGMA_S_ZERO );
    magma_s_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_S_ZERO );
    magma_s_vtransfer( b, &b_h, A.memory_location, Magma_CPU);

    if( A.storage_type != Magma_CSR ){
        magma_s_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_s_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_s_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_S_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                               (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_S_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

    }

    magma_s_csr_compressor(&B.val, &B.row, &B.col, 
                           &C.val, &C.row, &C.col, &B.num_rows );  

    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.row[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if( A.storage_type != Magma_CSR){
        A_h2.alignment = A.alignment;
        A_h2.blocksize = A.blocksize;
        magma_s_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
        magma_s_mtransfer( A_h2, M, Magma_CPU, A.memory_location);
    }
    else{
        magma_s_mtransfer( C, M, Magma_CPU, A.memory_location);
    }     
    magma_s_vtransfer( c_t, c, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_s_mfree( &A_h1 );
        magma_s_mfree( &A_h2 );   
    }   
    magma_s_mfree( &B );
    magma_s_mfree( &C );  
    magma_s_vfree( &diag);
    magma_s_vfree( &c_t);
    magma_s_vfree( &b_h);

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

    @param
    m           magma_s_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param
    c           magma_s_vector
                c = D^(-1) * b

    @param
    x           magma_s_vector*
                iteration vector x

    @param
    solver_par  magma_s_solver_par*
                solver parameters

    @ingroup magmasparse_s
    ********************************************************************/

magma_int_t
magma_sjacobiiter( magma_s_sparse_matrix M, magma_s_vector c, magma_s_vector *x,  
                                 magma_s_solver_par *solver_par ){

    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE, 
                                            c_mone = MAGMA_S_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_s_vector t;
    magma_s_vinit( &t, Magma_DEV, dofs, c_zero );


    for( magma_int_t i=0; i<solver_par->maxiter; i++ ){
        magma_s_spmv( c_mone, M, *x, c_zero, t );                // t = - M * x
        magma_saxpy( dofs, c_one , c.val, 1 , t.val, 1 );        // t = t + c
        magma_scopy( dofs, t.val, 1 , x->val, 1 );               // x = t
    }

    magma_s_vfree(&t);

    return MAGMA_SUCCESS;
}   /* magma_sjacobiiter */

