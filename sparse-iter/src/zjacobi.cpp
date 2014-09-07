/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b

    @param
    x           magma_z_vector*
                solution approximation

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zjacobi( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_z_solver_par *solver_par )
{
    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = 0;

    real_Double_t tempo1, tempo2;
    double residual;
    magma_zresidual( A, b, *x, &residual );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                                c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    double nom0;


    magma_z_sparse_matrix M;
    magma_z_vector c, r;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_spmv( c_one, A, *x, c_zero, r );                  // r = A x
    magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);           // r = r - b
    nom0 = magma_dznrm2(dofs, r.val, 1);                      // den = || r ||

    // Jacobi setup
    magma_zjacobisetup( A, b, &M, &c );
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;

    magma_device_sync(); tempo1=magma_wtime();

    // Jacobi iterator
    magma_zjacobiiter( M, c, x, &jacobiiter_par ); 

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_zresidual( A, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    magma_z_mfree( &M );
    magma_z_vfree( &c );
    magma_z_vfree( &r );

    return MAGMA_SUCCESS;
}   /* magma_zjacobi */






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
    A           magma_z_sparse_matrix
                input matrix A

    @param
    M           magma_z_sparse_matrix*
                M = D^(-1) * (L+U)

    @param
    d           magma_z_vector*
                vector with diagonal elements of A

    @ingroup magmasparse_z
    ********************************************************************/

magma_int_t
magma_zjacobisetup_matrix( magma_z_sparse_matrix A, 
                    magma_z_sparse_matrix *M, magma_z_vector *d ){

    magma_int_t i;

    magma_z_sparse_matrix A_h1, A_h2, B, C;
    magma_z_vector diag;
    magma_z_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );

    if( A.storage_type != Magma_CSR){
        magma_z_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_z_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_z_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
    }
    magma_z_csr_compressor(&B.val, &B.row, &B.col, 
                           &C.val, &C.row, &C.col, &B.num_rows );  
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.row[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if( A.storage_type != Magma_CSR){
        magma_z_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
        magma_z_mtransfer( A_h2, M, Magma_CPU, A.memory_location);
    }
    else{
        magma_z_mtransfer( C, M, Magma_CPU, A.memory_location);
    }    
    magma_z_vtransfer( diag, d, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_z_mfree( &A_h1 );
        magma_z_mfree( &A_h2 );   
    }
    magma_z_mfree( &B );
    magma_z_mfree( &C ); 

    magma_z_vfree( &diag);
 
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
    A           magma_z_sparse_matrix
                input matrix A

    @param
    d           magma_z_vector*
                vector with diagonal elements

    @ingroup magmasparse_z
    ********************************************************************/

magma_int_t
magma_zjacobisetup_diagscal( magma_z_sparse_matrix A, magma_z_vector *d ){

    magma_int_t i;

    magma_z_sparse_matrix A_h1, B;
    magma_z_vector diag;
    magma_z_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );

    if( A.storage_type != Magma_CSR){
        magma_z_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_z_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_z_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = 1.0/B.val[i];
                if( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
    }
    magma_z_vtransfer( diag, d, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_z_mfree( &A_h1 );
    }
    magma_z_mfree( &B );
    magma_z_vfree( &diag);
 
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
    b           magma_z_vector
                RHS b

    @param
    d           magma_z_vector
                vector with diagonal entries

    @param
    c           magma_z_vector*
                c = D^(-1) * b

    @ingroup magmasparse_z
    ********************************************************************/

magma_int_t
magma_zjacobisetup_vector( magma_z_vector b, magma_z_vector d, 
                           magma_z_vector *c ){

    if( b.memory_location == Magma_CPU ){
        magma_z_vector diag, c_t, b_h;
        magma_z_vinit( &c_t, Magma_CPU, b.num_rows, MAGMA_Z_ZERO );

        magma_z_vtransfer( b, &b_h, b.memory_location, Magma_CPU);
        magma_z_vtransfer( d, &diag, b.memory_location, Magma_CPU);

        for( magma_int_t rowindex=0; rowindex<b.num_rows; rowindex++ ){   
            c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

        }  
        magma_z_vtransfer( c_t, c, Magma_CPU, b.memory_location); 

        magma_z_vfree( &diag);
        magma_z_vfree( &c_t);
        magma_z_vfree( &b_h);

        return MAGMA_SUCCESS;
    }
    else if( b.memory_location == Magma_DEV ){
        // fill vector
        magma_z_vector tmp;
        magma_z_vinit( &tmp, Magma_DEV, b.num_rows, MAGMA_Z_ZERO );
        magma_zjacobisetup_vector_gpu( 
                    b.num_rows, b.val, d.val, c->val, tmp.val );
        magma_z_vfree( &tmp );
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
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b

    @param
    M           magma_z_sparse_matrix*
                M = D^(-1) * (L+U)

    @param
    c           magma_z_vector*
                c = D^(-1) * b

    @ingroup magmasparse_z
    ********************************************************************/

magma_int_t
magma_zjacobisetup( magma_z_sparse_matrix A, magma_z_vector b, 
                    magma_z_sparse_matrix *M, magma_z_vector *c ){

    magma_int_t i;

    magma_z_sparse_matrix A_h1, A_h2, B, C;
    magma_z_vector diag, c_t, b_h;
    magma_z_vinit( &c_t, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );
    magma_z_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_Z_ZERO );
    magma_z_vtransfer( b, &b_h, A.memory_location, Magma_CPU);

    if( A.storage_type != Magma_CSR ){
        magma_z_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_z_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_z_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                               (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

    }

    magma_z_csr_compressor(&B.val, &B.row, &B.col, 
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
        magma_z_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
        magma_z_mtransfer( A_h2, M, Magma_CPU, A.memory_location);
    }
    else{
        magma_z_mtransfer( C, M, Magma_CPU, A.memory_location);
    }     
    magma_z_vtransfer( c_t, c, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_z_mfree( &A_h1 );
        magma_z_mfree( &A_h2 );   
    }   
    magma_z_mfree( &B );
    magma_z_mfree( &C );  
    magma_z_vfree( &diag);
    magma_z_vfree( &c_t);
    magma_z_vfree( &b_h);

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
    M           magma_z_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param
    c           magma_z_vector
                c = D^(-1) * b

    @param
    x           magma_z_vector*
                iteration vector x

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @ingroup magmasparse_z
    ********************************************************************/

magma_int_t
magma_zjacobiiter( magma_z_sparse_matrix M, magma_z_vector c, magma_z_vector *x,  
                                 magma_z_solver_par *solver_par ){

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_z_vector t, swap;
    magma_z_vinit( &t, Magma_DEV, dofs, c_zero );


    for( magma_int_t i=0; i<solver_par->maxiter; i++ ){
        magma_z_spmv( c_mone, M, *x, c_zero, t );                // t = - M * x
        magma_zaxpy( dofs, c_one , c.val, 1 , t.val, 1 );        // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = t;
        t = swap;        
        //magma_zcopy( dofs, t.val, 1 , x->val, 1 );               // x = t
    }

    magma_z_vfree( &t );

    return MAGMA_SUCCESS;
}   /* magma_zjacobiiter */



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param
    M           magma_z_sparse_matrix
                input matrix M = D^(-1) * (L+U)

    @param
    c           magma_z_vector
                c = D^(-1) * b

    @param
    x           magma_z_vector*
                iteration vector x

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @param
    solver_par  magma_z_precond_par*
                precond parameters

    @ingroup magmasparse_z
    ********************************************************************/

magma_int_t
magma_zjacobiiter_precond( magma_z_sparse_matrix M, magma_z_vector *x, 
            magma_z_solver_par *solver_par, magma_z_preconditioner *precond ){

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_z_vector swap;

    for( magma_int_t i=0; i<solver_par->maxiter; i++ ){
        magma_z_spmv( c_mone, M, *x, c_zero, precond->work2 );   // t = - M * x
        magma_zaxpy( dofs, c_one , 
                precond->work1.val, 1 , precond->work2.val, 1 ); // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = precond->work2;
        precond->work2 = swap;        
        //magma_zcopy( dofs, t.val, 1 , x->val, 1 );               // x = t
    }

    return MAGMA_SUCCESS;
}   /* magma_zjacobiiter */
