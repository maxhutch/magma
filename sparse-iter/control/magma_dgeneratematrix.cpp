/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_zgeneratematrix.cpp normal z -> d, Fri Jul 18 17:34:30 2014
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

#include "magmasparse_d.h"
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
    diag_vals   double*
                array containing the values

                                                (length offsets+1)
    @param
    A           magma_d_sparse_matrix*
                matrix to generate   

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dmgenerator(  magma_int_t n,
                    magma_int_t offdiags,
                    magma_index_t *diag_offset,
                    double *diag_vals,
                    magma_d_sparse_matrix *A ){

    magma_d_sparse_matrix B;

    B.num_rows = n;
    B.num_cols = n;
    B.memory_location = Magma_CPU;
    B.storage_type = Magma_ELLPACK;
    B.max_nnz_row = (2*offdiags+1);

    magma_dmalloc_cpu( &B.val, B.max_nnz_row*n );
    magma_index_malloc_cpu( &B.col, B.max_nnz_row*n );

    for( int i=0; i<n; i++ ){ // stride over rows
        // stride over the number of nonzeros in each row
        for( int j=0; j<offdiags; j++ ){ // left of diagonal
            B.val[ i*B.max_nnz_row + j ] = diag_vals[ offdiags - j ];
            B.col[ i*B.max_nnz_row + j ] = -1 * diag_offset[ offdiags-j ] + i;
        } 
        // elements on the diagonal
        B.val[ i*B.max_nnz_row + offdiags ] = diag_vals[ 0 ];
        B.col[ i*B.max_nnz_row + offdiags ] = i;
        for( int j=0; j<offdiags; j++ ){ // right of diagonal
            B.val[ i*B.max_nnz_row + j + offdiags +1 ] = diag_vals[ j+1 ];
            B.col[ i*B.max_nnz_row + j + offdiags +1 ] = diag_offset[ j+1 ] + i;
        } 
    }

    // set invalid entries to zero
    for( int i=0; i<n; i++ ){ // stride over rows
        for( int j=0; j<B.max_nnz_row; j++ ){ // nonzeros in every row
            if( (B.col[i*B.max_nnz_row + j] < 0) || 
                    (B.col[i*B.max_nnz_row + j] >= n) ){
                B.val[ i*B.max_nnz_row + j ] = MAGMA_D_MAKE( 0.0, 0.0 );
            }
        } 

    }    
    B.nnz = 0;

    for( int i=0; i<n; i++ ){ // stride over rows
        for( int j=0; j<B.max_nnz_row; j++ ){ // nonzeros in every row
            if( MAGMA_D_REAL( B.val[i*B.max_nnz_row + j]) != 0.0 ) 
                B.nnz++;
        } 

    }  

    // converting it to CSR will remove the invalit entries completely
    magma_d_mconvert( B, A, Magma_ELLPACK, Magma_CSR );

    return MAGMA_SUCCESS;
}   

