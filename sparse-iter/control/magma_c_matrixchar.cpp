/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_z_matrixchar.cpp normal z -> c, Fri May 30 10:41:45 2014
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
#include "../include/magmasparse_c.h"
#include "../../include/magma.h"
#include "../include/mmio.h"


#define THRESHOLD 10e-99

using namespace std;


magma_int_t 
magma_crowentries( magma_c_sparse_matrix *A ){

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Checks the maximal number of nonzeros in a row of matrix A. 
    Inserts the data into max_nnz_row.


    Arguments
    =========

    magma_c_sparse_matrix *A              sparse matrix     

    ========================================================================  */

    // check whether matrix on CPU
    if( A->memory_location == Magma_CPU ){
        // CSR  
        if( A->storage_type == Magma_CSR ){
            magma_index_t i, *length, maxrowlength=0;
            magma_indexmalloc_cpu( &length, A->num_rows);

            for( i=0; i<A->num_rows; i++ ){
                length[i] = A->row[i+1]-A->row[i];
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            A->max_nnz_row = maxrowlength;
            magma_free( length );
            return MAGMA_SUCCESS; 
        }
        // Dense
        else if( A->storage_type == Magma_DENSE ){
            magma_int_t i, j, maxrowlength=0;
            magma_index_t *length;
            magma_indexmalloc_cpu( &length, A->num_rows);

            for( i=0; i<A->num_rows; i++ ){
                length[i] = 0;
                for( j=0; j<A->num_cols; j++ ){
                    if( MAGMA_C_REAL( A->val[i*A->num_cols + j] ) != 0. )
                        length[i]++;
                    } 
                if(length[i] > maxrowlength)
                     maxrowlength = length[i];
            }
            A->max_nnz_row = maxrowlength;
            magma_free( length );
            return MAGMA_SUCCESS; 
        }
    } // end CPU case

    else{
        printf("error: matrix not on CPU.\n");
        return MAGMA_ERR_ALLOCATION;
    }
    return MAGMA_SUCCESS; 
}



magma_int_t 
magma_cdiameter( magma_c_sparse_matrix *A ){

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Computes the diameter of a sparse matrix and stores the value in diameter.


    Arguments
    =========

    magma_c_sparse_matrix *A              sparse matrix     

    ========================================================================  */

    // check whether matrix on CPU
    if( A->memory_location == Magma_CPU ){
        // CSR  
        if( A->storage_type == Magma_CSR ){
            magma_index_t i, j, tmp,  *dim, maxdim=0;
            magma_indexmalloc_cpu( &dim, A->num_rows);
            for( i=0; i<A->num_rows; i++ ){
                dim[i] = 0;
                for( j=A->row[i]; j<A->row[i+1]; j++ ){
                   // if( MAGMA_C_REAL(A->val[j]) > THRESHOLD ){
                        tmp = abs( i - A->col[j] );
                        if( tmp > dim[i] )
                            dim[i] = tmp;
                   // }
                }
                if( dim[i] > maxdim )
                     maxdim = dim[i];
            }
            magma_free( &dim );
            A->diameter = maxdim;
            return MAGMA_SUCCESS; 
        }
        // Dense
        else if( A->storage_type == Magma_DENSE ){
            magma_index_t i, j, tmp,  *dim, maxdim=0;
            magma_indexmalloc_cpu( &dim, A->num_rows);
            for( i=0; i<A->num_rows; i++ ){
                dim[i] = 0;
                for( j=0; j<A->num_cols; j++ ){
                    if( MAGMA_C_REAL( A->val[i*A->num_cols + j] ) !=  0.0 ){
                        tmp = abs( i -j );
                        if( tmp > dim[i] )
                            dim[i] = tmp;
                    }
                }
                if( dim[i] > maxdim )
                     maxdim = dim[i];
            }
            magma_free( &dim );
            A->diameter = maxdim;
            return MAGMA_SUCCESS; 
        }
        // ELLPACK
        else if( A->storage_type == Magma_ELLPACK ){
            magma_index_t i, j, tmp,  *dim, maxdim=0;
            magma_indexmalloc_cpu( &dim, A->num_rows);
            for( i=0; i<A->num_rows; i++ ){
                dim[i] = 0;
                for( j=i*A->max_nnz_row; j<(i+1)*A->max_nnz_row; j++ ){
                    if( MAGMA_C_REAL( A->val[j] ) > THRESHOLD ){
                        tmp = abs( i - A->col[j] );
                        if( tmp > dim[i] )
                            dim[i] = tmp;
                    }
                }
                if( dim[i] > maxdim )
                     maxdim = dim[i];
            }
            magma_free( &dim );
            A->diameter = maxdim;
            return MAGMA_SUCCESS; 
        }
        // ELL
        else if( A->storage_type == Magma_ELL ){
            printf("error:format not supported.\n");
            return MAGMA_ERR_ALLOCATION;
        }
    } // end CPU case

    else{
        printf("error: matrix not on CPU.\n");
        return MAGMA_ERR_ALLOCATION;
    }
    return MAGMA_SUCCESS; 
}
