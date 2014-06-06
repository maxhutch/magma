/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_zlag2c.cpp mixed zc -> ds, Fri May 30 10:41:34 2014
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
#include "../include/magmasparse_z.h"
#include "../include/magmasparse_ds.h"
#include "../../include/magma.h"
#include "../include/mmio.h"
#include "common_magma.h"


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    convertes magma_d_vector from Z to C

    Arguments
    =========

    magma_d_vector x        input vector descriptor
    magma_s_vector *y       output vector descriptor

    ========================================================================  */

magma_int_t
magma_vector_dlag2s( magma_d_vector x, magma_s_vector *y )
{
    magma_int_t info;
    if( x.memory_location == Magma_DEV){
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_smalloc( &y->val, x.num_rows );
        magmablas_dlag2s_sparse( x.num_rows, 1, x.val, x.num_rows, y->val, 
                    x.num_rows, &info );
        return MAGMA_SUCCESS;
    }
    else if( x.memory_location == Magma_CPU ){
        y->memory_location = x.memory_location;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        magma_smalloc_cpu( &y->val, x.num_rows );

        magma_int_t one= 1;
        magma_int_t info;
        lapackf77_dlag2s( &x.num_rows, &one, 
                       x.val, &x.num_rows, 
                       y->val, &x.num_rows, &info);
        return MAGMA_SUCCESS;

    }
    else
        return MAGMA_ERR_NOT_SUPPORTED;
}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    convertes magma_d_sparse_matrix from Z to C

    Arguments
    =========

    magma_d_sparse_matrix A         input matrix descriptor
    magma_s_sparse_matrix *B        output matrix descriptor

    ========================================================================  */

magma_int_t
magma_sparse_matrix_dlag2s( magma_d_sparse_matrix A, magma_s_sparse_matrix *B )
{
    magma_int_t info;
    if( A.memory_location == Magma_DEV){
        B->storage_type = A.storage_type;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        B->max_nnz_row = A.max_nnz_row;
        if( A.storage_type == Magma_CSR ){
            magma_smalloc( &B->val, A.nnz );
            magmablas_dlag2s_sparse( A.nnz, 1, A.val, A.nnz, B->val, 
                    A.nnz, &info );
            B->row = A.row;
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_ELLPACK ){
            magma_smalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_dlag2s_sparse( A.num_rows*A.max_nnz_row, 1, A.val, 
            A.num_rows*A.max_nnz_row, B->val, A.num_rows*A.max_nnz_row, &info );
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_ELL ){
            magma_smalloc( &B->val, A.num_rows*A.max_nnz_row );
            magmablas_dlag2s_sparse(  A.num_rows*A.max_nnz_row, 1, A.val, 
            A.num_rows*A.max_nnz_row, B->val, A.num_rows*A.max_nnz_row, &info );
            B->col = A.col;
            return MAGMA_SUCCESS;
        }
        if( A.storage_type == Magma_DENSE ){
            magma_smalloc( &B->val, A.num_rows*A.num_cols );
            magmablas_dlag2s_sparse(  A.num_rows, A.num_cols, A.val, A.num_rows, 
                    B->val, A.num_rows, &info );
            return MAGMA_SUCCESS;
        }
        else{
            return MAGMA_ERR_NOT_SUPPORTED;
            printf("error:format not supported\n");
        }
    }
    else{
        return MAGMA_ERR_NOT_SUPPORTED;
        printf("error:matrix not on GPU\n");
        }
}

