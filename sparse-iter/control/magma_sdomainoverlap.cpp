/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zdomainoverlap.cpp normal z -> s, Sun May  3 11:23:01 2015
       @author Hartwig Anzt
*/
#include "common_magmasparse.h"


extern "C"
magma_int_t
magma_sindexcopy(
    magma_int_t num_copy,
    magma_int_t offset,
    magma_index_t *tmp_x,
    magma_index_t *x,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    CHECK( magma_sindexsort( tmp_x, 0, num_copy-1, queue ));
    for( magma_int_t j=0; j<num_copy; j++ ){
        x[ j+offset ] = tmp_x[ j ];
        tmp_x[ j ] = -1;
    }
        
cleanup:    
    return info;
}


/**
    Purpose
    -------

    Generates the update list.

    Arguments
    ---------

    @param[in]
    x           magma_index_t*
                array to sort

    @param[in]
    num_rows    magma_int_t
                number of rows in matrix
                
    @param[out]
    num_indices magma_int_t*
                number of indices in array

    @param[in]
    rowptr      magma_index_t*
                rowpointer of matrix
                
    @param[in]
    colidx      magma_index_t*
                colindices of matrix
                
    @param[in]
    x           magma_index_t*
                array containing indices for domain overlap

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_sdomainoverlap(
    magma_index_t num_rows,
    magma_index_t *num_indices,
    magma_index_t *rowptr,
    magma_index_t *colidx,
    magma_index_t *x,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t blocksize=128;

    magma_int_t row=0, col=0, num_ind=0, offset=0;

    magma_index_t *tmp_x;
    CHECK( magma_index_malloc_cpu( &tmp_x, blocksize ));

    for(magma_int_t i=0; i<blocksize; i++ ){
        tmp_x[i] = -1;
    }
    
    for(magma_int_t i=0; i<num_rows; i++){
        row = i;
        for(magma_int_t j=rowptr[row]; j<rowptr[row+1]; j++){
            col = colidx[j];
            int floatitem = 0;
            for(magma_int_t k=0; k<blocksize; k++){
              if( tmp_x[k] == col )
                  floatitem = 1;
            }
            if( floatitem == 0 ){
                tmp_x[num_ind] = col;
                num_ind++;
                (*num_indices)++;
            }
            if( num_ind == blocksize || j == rowptr[num_rows]-1 ){
                magma_sindexcopy( num_ind, offset, tmp_x, x, queue );
                offset=offset+num_ind;
                num_ind = 0;
                break;
            }
        }
    }

cleanup:
    magma_free_cpu( tmp_x );
    return info;

}

