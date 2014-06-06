/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @author Hartwig Anzt 

       @generated from zjacobi.cpp normal z -> d, Fri May 30 10:41:41 2014
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method.

    Arguments
    =========

    magma_d_sparse_matrix A                   input matrix A
    magma_d_vector b                          RHS b
    magma_d_vector *x                         solution approximation
    magma_d_solver_par *solver_par       solver parameters

    ========================================================================  */


magma_int_t
magma_djacobi( magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
           magma_d_solver_par *solver_par )
{
    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = 0;

    real_Double_t tempo1, tempo2;
    double residual;
    magma_dresidual( A, b, *x, &residual );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // some useful variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE, 
                                                c_mone = MAGMA_D_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    double nom0;


    magma_d_sparse_matrix M;
    magma_d_vector c, r;
    magma_d_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_d_spmv( c_one, A, *x, c_zero, r );                  // r = A x
    magma_daxpy(dofs,  c_mone, b.val, 1, r.val, 1);           // r = r - b
    nom0 = magma_dnrm2(dofs, r.val, 1);                      // den = || r ||

    // Jacobi setup
    magma_djacobisetup( A, b, &M, &c );
    magma_d_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;

    magma_device_sync(); tempo1=magma_wtime();

    // Jacobi iterator
    magma_djacobiiter( M, c, x, &jacobiiter_par ); 

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_dresidual( A, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    magma_d_mfree( &M );
    magma_d_vfree( &c );
    magma_d_vfree( &r );

    return MAGMA_SUCCESS;
}   /* magma_djacobi */






/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the Matrix M for the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    It returns the preconditioner Matrix M and a vector d
    containing the diagonal elements. 

    Arguments
    =========

    magma_d_sparse_matrix A                   input matrix A
    magma_d_vector b                          RHS b
    magma_d_sparse_matrix *M                  M = D^(-1) * (L+U)
    magma_d_vector *d                         vector with diagonal elements

    ========================================================================  */

magma_int_t
magma_djacobisetup_matrix( magma_d_sparse_matrix A, magma_d_vector b, 
                    magma_d_sparse_matrix *M, magma_d_vector *d ){

    magma_int_t i;

    magma_d_sparse_matrix A_h1, A_h2, B, C;
    magma_d_vector diag, b_h;
    magma_d_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_D_ZERO );
    magma_d_vtransfer( b, &b_h, A.memory_location, Magma_CPU);

    if( A.storage_type != Magma_CSR){
        magma_d_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_d_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_d_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_D_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_D_MAKE( 0., 0. );
            }
        }
    }
    magma_d_csr_compressor(&B.val, &B.row, &B.col, 
                           &C.val, &C.row, &C.col, &B.num_rows, &B.num_rows);  
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.row[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if( A.storage_type != Magma_CSR){
        magma_d_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
        magma_d_mtransfer( A_h2, M, Magma_CPU, A.memory_location);
    }
    else{
        magma_d_mtransfer( C, M, Magma_CPU, A.memory_location);
    }    
    magma_d_vtransfer( diag, d, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_d_mfree( &A_h1 );
        magma_d_mfree( &A_h2 );   
    }
    magma_d_mfree( &B );
    magma_d_mfree( &C ); 

    magma_d_vfree( &diag);
    magma_d_vfree( &b_h);
 
    return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======


    It returns a vector d
    containing the inverse diagonal elements. 

    Arguments
    =========

    magma_d_sparse_matrix A                   input matrix A
    magma_d_vector *d                         vector with diagonal elements

    ========================================================================  */

magma_int_t
magma_djacobisetup_diagscal( magma_d_sparse_matrix A, magma_d_vector *d ){

    magma_int_t i;

    magma_d_sparse_matrix A_h1, B;
    magma_d_vector diag;
    magma_d_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_D_ZERO );

    if( A.storage_type != Magma_CSR){
        magma_d_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_d_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_d_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = 1.0/B.val[i];
                if( MAGMA_D_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                                (int) rowindex);
            }
        }
    }
    magma_d_vtransfer( diag, d, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_d_mfree( &A_h1 );
    }
    magma_d_mfree( &B );
    magma_d_vfree( &diag);
 
    return MAGMA_SUCCESS;
}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c

    Arguments
    =========

    magma_d_vector b                          RHS b
    magma_d_vector d                          vector with diagonal entries
    magma_d_vector *c                         c = D^(-1) * b

    ========================================================================  */

magma_int_t
magma_djacobisetup_vector( magma_d_vector b, magma_d_vector d, 
                           magma_d_vector *c ){

    if( b.memory_location == Magma_CPU ){
        magma_d_vector diag, c_t, b_h;
        magma_d_vinit( &c_t, Magma_CPU, b.num_rows, MAGMA_D_ZERO );

        magma_d_vtransfer( b, &b_h, b.memory_location, Magma_CPU);
        magma_d_vtransfer( d, &diag, b.memory_location, Magma_CPU);

        for( magma_int_t rowindex=0; rowindex<b.num_rows; rowindex++ ){   
            c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

        }  
        magma_d_vtransfer( c_t, c, Magma_CPU, b.memory_location); 

        magma_d_vfree( &diag);
        magma_d_vfree( &c_t);
        magma_d_vfree( &b_h);

        return MAGMA_SUCCESS;
    }
    else if( b.memory_location == Magma_DEV ){
        // fill vector
        magma_djacobisetup_vector_gpu( b.num_rows, b.val, d.val, c->val );
        return MAGMA_SUCCESS;
    }

    return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    =========

    magma_d_sparse_matrix A                   input matrix A
    magma_d_vector b                          RHS b
    magma_d_sparse_matrix *M                  M = D^(-1) * (L+U)
    magma_d_vector *c                         c = D^(-1) * b

    ========================================================================  */

magma_int_t
magma_djacobisetup( magma_d_sparse_matrix A, magma_d_vector b, 
                    magma_d_sparse_matrix *M, magma_d_vector *c ){

    magma_int_t i;

    magma_d_sparse_matrix A_h1, A_h2, B, C;
    magma_d_vector diag, c_t, b_h;
    magma_d_vinit( &c_t, Magma_CPU, A.num_rows, MAGMA_D_ZERO );
    magma_d_vinit( &diag, Magma_CPU, A.num_rows, MAGMA_D_ZERO );
    magma_d_vtransfer( b, &b_h, A.memory_location, Magma_CPU);

    if( A.storage_type != Magma_CSR ){
        magma_d_mtransfer( A, &A_h1, A.memory_location, Magma_CPU);
        magma_d_mconvert( A_h1, &B, A_h1.storage_type, Magma_CSR);
    }
    else{
        magma_d_mtransfer( A, &B, A.memory_location, Magma_CPU);
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ){
        magma_int_t start = (B.row[rowindex]);
        magma_int_t end = (B.row[rowindex+1]);
        for( i=start; i<end; i++ ){
            if( B.col[i]==rowindex ){
                diag.val[rowindex] = B.val[i];
                if( MAGMA_D_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n", 
                                                               (int) rowindex);
            }
        }
        for( i=start; i<end; i++ ){
            B.val[i] = B.val[i] / diag.val[rowindex];
            if( B.col[i]==rowindex ){
                B.val[i] = MAGMA_D_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];

    }

    magma_d_csr_compressor(&B.val, &B.row, &B.col, 
                           &C.val, &C.row, &C.col, &B.num_rows, &B.num_rows);  

    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.row[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if( A.storage_type != Magma_CSR){
        A_h2.alignment = A.alignment;
        A_h2.blocksize = A.blocksize;
        magma_d_mconvert( C, &A_h2, Magma_CSR, A_h1.storage_type);
        magma_d_mtransfer( A_h2, M, Magma_CPU, A.memory_location);
    }
    else{
        magma_d_mtransfer( C, M, Magma_CPU, A.memory_location);
    }     
    magma_d_vtransfer( c_t, c, Magma_CPU, A.memory_location);

    if( A.storage_type != Magma_CSR){
        magma_d_mfree( &A_h1 );
        magma_d_mfree( &A_h2 );   
    }   
    magma_d_mfree( &B );
    magma_d_mfree( &C );  
    magma_d_vfree( &diag);
    magma_d_vfree( &c_t);
    magma_d_vfree( &b_h);

    return MAGMA_SUCCESS;

}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    =========

    magma_d_sparse_matrix M                   input matrix M = D^(-1) * (L+U)
    magma_d_vector c                          c = D^(-1) * b
    magma_d_vector *x                         iteration vector x
    magma_d_solver_par *solver_par       solver parameters

    ========================================================================  */


magma_int_t
magma_djacobiiter( magma_d_sparse_matrix M, magma_d_vector c, magma_d_vector *x,  
                                 magma_d_solver_par *solver_par ){

    // local variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE, 
                                            c_mone = MAGMA_D_NEG_ONE;
    magma_int_t dofs = M.num_rows;
    magma_d_vector t;
    magma_d_vinit( &t, Magma_DEV, dofs, c_zero );


    for( magma_int_t i=0; i<solver_par->maxiter; i++ ){
        magma_d_spmv( c_mone, M, *x, c_zero, t );                // t = - M * x
        magma_daxpy( dofs, c_one , c.val, 1 , t.val, 1 );        // t = t + c
        magma_dcopy( dofs, t.val, 1 , x->val, 1 );               // x = t
    }

    magma_d_vfree(&t);

    return MAGMA_SUCCESS;
}   /* magma_djacobiiter */

