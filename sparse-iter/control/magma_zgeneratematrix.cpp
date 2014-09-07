/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/
#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>

#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"


using namespace std;

/**
    Purpose
    -------

    Generate a symmetric n x n CSR matrix for a stencil.

    Arguments
    ---------

    @param
    n           magma_int_t
                number of rows

    @param
    offdiags    magma_int_t
                number of offdiagonals 

    @param
    diag_offset magma_int_t*
                array containing the offsets 

                                                (length offsets+1)
    @param
    diag_vals   magmaDoubleComplex*
                array containing the values

                                                (length offsets+1)
    @param
    A           magma_z_sparse_matrix*
                matrix to generate   

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zmgenerator(  magma_int_t n,
                    magma_int_t offdiags,
                    magma_index_t *diag_offset,
                    magmaDoubleComplex *diag_vals,
                    magma_z_sparse_matrix *A ){

    magma_z_sparse_matrix B;

    B.num_rows = n;
    B.num_cols = n;
    B.memory_location = Magma_CPU;
    B.storage_type = Magma_ELLPACK;
    B.max_nnz_row = (2*offdiags+1);

    magma_zmalloc_cpu( &B.val, B.max_nnz_row*n );
    magma_index_malloc_cpu( &B.col, B.max_nnz_row*n );

    for( int i=0; i<n; i++ ){ // stride over rows
        // stride over the number of nonzeros in each row
        // left of diagonal
        for( int j=0; j<offdiags; j++ ){ 
            B.val[ i*B.max_nnz_row + j ] = diag_vals[ offdiags - j ];
            B.col[ i*B.max_nnz_row + j ] = -1 * diag_offset[ offdiags-j ] + i;
        } 
        // elements on the diagonal
        B.val[ i*B.max_nnz_row + offdiags ] = diag_vals[ 0 ];
        B.col[ i*B.max_nnz_row + offdiags ] = i;
        // right of diagonal
        for( int j=0; j<offdiags; j++ ){ 
            B.val[ i*B.max_nnz_row + j + offdiags +1 ] = diag_vals[ j+1 ];
            B.col[ i*B.max_nnz_row + j + offdiags +1 ] = diag_offset[ j+1 ] + i;
        } 
    }

    // set invalid entries to zero
    for( int i=0; i<n; i++ ){ // stride over rows
        for( int j=0; j<B.max_nnz_row; j++ ){ // nonzeros in every row
            if( (B.col[i*B.max_nnz_row + j] < 0) || 
                    (B.col[i*B.max_nnz_row + j] >= n) ){
                B.val[ i*B.max_nnz_row + j ] = MAGMA_Z_MAKE( 0.0, 0.0 );
            }
        } 

    }    

    B.nnz = 0;

    for( int i=0; i<n; i++ ){ // stride over rows
        for( int j=0; j<B.max_nnz_row; j++ ){ // nonzeros in every row
            if( MAGMA_Z_REAL( B.val[i*B.max_nnz_row + j]) != 0.0 ) 
                B.nnz++;
        } 

    }  

    // converting it to CSR will remove the invalit entries completely
    magma_z_mconvert( B, A, Magma_ELLPACK, Magma_CSR );

    return MAGMA_SUCCESS;
}   




/**
    Purpose
    -------

    Generate a 27-point stencil for a 3D FD discretization.

    Arguments
    ---------

    @param
    n           magma_int_t
                number of rows

    @param
    A           magma_z_sparse_matrix*
                matrix to generate   

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zm_27stencil(  magma_int_t n,
                     magma_z_sparse_matrix *A ){

    magma_int_t i,j,k;
    magma_z_sparse_matrix hA;
    
    // generate matrix of desired structure and size (3d 27-point stencil)
    magma_int_t nn = n*n*n;
    magma_int_t offdiags = 13;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n-1;
    diag_offset[3] = n;
    diag_offset[4] = n+1;
    diag_offset[5] = n*n-n-1;
    diag_offset[6] = n*n-n;
    diag_offset[7] = n*n-n+1;
    diag_offset[8] = n*n-1;
    diag_offset[9] = n*n;
    diag_offset[10] = n*n+1;
    diag_offset[11] = n*n+n-1;
    diag_offset[12] = n*n+n;
    diag_offset[13] = n*n+n+1;

    diag_vals[0] = MAGMA_Z_MAKE( 26.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[3] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[4] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[5] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[6] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[7] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[8] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[9] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[10] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[11] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[12] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[13] = MAGMA_Z_MAKE( -1.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );


    // now set some entries to zero (boundary...)
    for( i=0; i<n*n; i++ ){
    for( j=0; j<n; j++ ){
        magma_index_t row = i*n+j;
        magma_index_t l_bound = i*n;
        magma_index_t u_bound = (i+1)*n;
        for( k=hA.row[row]; k<hA.row[row+1]; k++){

            if((hA.col[k] == row-1 ||
                hA.col[k] == row-n-1 ||
                hA.col[k] == row+n-1 ||
                hA.col[k] == row-n*n+n-1 ||
                hA.col[k] == row+n*n-n-1 ||
                hA.col[k] == row-n*n-1 ||
                hA.col[k] == row+n*n-1 ||
                hA.col[k] == row-n*n-n-1 ||
                hA.col[k] == row+n*n+n-1 ) && (row+1)%n == 1 )
                    
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );

            if((hA.col[k] == row+1 ||
                hA.col[k] == row-n+1 ||
                hA.col[k] == row+n+1 ||
                hA.col[k] == row-n*n+n+1 ||
                hA.col[k] == row+n*n-n+1 ||
                hA.col[k] == row-n*n+1 ||
                hA.col[k] == row+n*n+1 ||
                hA.col[k] == row-n*n-n+1 ||
                hA.col[k] == row+n*n+n+1 ) && (row)%n ==n-1 )
                    
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
        }
        
    }
    }
    magma_z_mconvert( hA, A, Magma_CSR, Magma_CSR );
    magma_z_mfree( &hA );

    return MAGMA_SUCCESS;
}   



/**
    Purpose
    -------

    Generate a 5-point stencil for a 2D FD discretization.

    Arguments
    ---------

    @param
    n           magma_int_t
                number of rows

    @param
    A           magma_z_sparse_matrix*
                matrix to generate   

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zm_5stencil(  magma_int_t n,
                    magma_z_sparse_matrix *A ){

    magma_int_t i,j,k;
    magma_z_sparse_matrix hA;
    
    // generate matrix of desired structure and size (2d 5-point stencil)
    magma_int_t nn = n*n;
    magma_int_t offdiags = 2;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;

    diag_vals[0] = MAGMA_Z_MAKE( 4.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &hA );

    // now set some entries to zero (boundary...)
    for( i=0; i<n; i++ ){
    for( j=0; j<n; j++ ){
        magma_index_t row = i*n+j;
        magma_index_t l_bound = i*n;
        magma_index_t u_bound = (i+1)*n;
        for( k=hA.row[row]; k<hA.row[row+1]; k++){

            if((hA.col[k] == row-1 ) && (row+1)%n == 1 )
                    
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );

            if((hA.col[k] == row+1 ) && (row)%n ==n-1 )
                    
                    hA.val[k] = MAGMA_Z_MAKE( 0.0, 0.0 );
        }
        
    }
    }

    magma_z_mconvert( hA, A, Magma_CSR, Magma_CSR );
    magma_z_mfree( &hA );

    return MAGMA_SUCCESS;
}   

