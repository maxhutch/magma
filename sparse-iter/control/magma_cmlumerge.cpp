/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_zmlumerge.cpp normal z -> c, Sat Nov 15 19:54:23 2014
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Takes an strictly lower triangular matrix L and an upper triangular matrix U
    and merges them into a matrix A containing the upper and lower triangular 
    parts.

    Arguments
    ---------

    @param[in]
    L           magma_c_sparse_matrix
                input strictly lower triangular matrix L

    @param[in]
    U           magma_c_sparse_matrix
                input upper triangular matrix U
    
    @param[out]
    A           magma_c_sparse_matrix*
                output matrix 
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmlumerge(    
    magma_c_sparse_matrix L, 
    magma_c_sparse_matrix U,
    magma_c_sparse_matrix *A,
    magma_queue_t queue ){
    if( L.storage_type == Magma_CSR && U.storage_type == Magma_CSR ){
        if( L.memory_location == Magma_CPU && U.memory_location == Magma_CPU ){
            
            magma_c_mtransfer( L, A, Magma_CPU, Magma_CPU, queue );
            magma_free_cpu( A->col );
            magma_free_cpu( A->val );
            // make sure it is strictly lower triangular
            magma_int_t z = 0;
            for(magma_int_t i=0; i<A->num_rows; i++){
                for(magma_int_t j=L.row[i]; j<L.row[i+1]; j++){
                    if( L.col[j] < i ){// make sure it is strictly lower triangular
                        z++;
                    }
                }
                for(magma_int_t j=U.row[i]; j<U.row[i+1]; j++){
                    z++;
                }
            }
            A->nnz = z;
            // fill A with the new structure;
            magma_int_t stat_cpu = 0;
            stat_cpu += magma_index_malloc_cpu( &A->col, A->nnz );
            stat_cpu += magma_cmalloc_cpu( &A->val, A->nnz );
            if( stat_cpu != 0 ){
                magma_c_mfree( A, queue );
                printf("error: memory allocation.\n");
                return MAGMA_ERR_HOST_ALLOC;
            }
            z = 0;
            for(magma_int_t i=0; i<A->num_rows; i++){
                A->row[i] = z;
                for(magma_int_t j=L.row[i]; j<L.row[i+1]; j++){
                    if( L.col[j] < i ){// make sure it is strictly lower triangular
                        A->col[z] = L.col[j];
                        A->val[z] = L.val[j];
                        z++;
                    }
                }
                for(magma_int_t j=U.row[i]; j<U.row[i+1]; j++){
                    A->col[z] = U.col[j];
                    A->val[z] = U.val[j];
                    z++;
                }
            }
            A->row[A->num_rows] = z;
            A->nnz = z;
            return MAGMA_SUCCESS; 
        }
        else{
    
            printf("error: matrix not on CPU.\n"); 
    
            return MAGMA_SUCCESS; 
        }
    }
    else{
        
        printf("error: matrix not on CPU.\n"); 
    
        return MAGMA_SUCCESS; 
    }
}





