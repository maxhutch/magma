/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from magma_zcsrsplit.cpp normal z -> s, Fri May 30 10:41:45 2014
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

#define min(a, b) ((a) < (b) ? (a) : (b))


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Splits a CSR matrix into two matrices, one containing the diagonal blocks
    with the diagonal element stored first, one containing the rest of the 
    original matrix.

    Arguments
    =========

    magma_int_t blocksize               size of the diagonal blocks
    magma_s_sparse_matrix A             CSR input matrix
    magma_s_sparse_matrix *D            CSR matrix containing diagonal blocks
    magma_s_sparse_matrix *R            CSR matrix containing rest

    ========================================================================  */




magma_int_t
magma_scsrsplit(    magma_int_t bsize,
                    magma_s_sparse_matrix A,
                    magma_s_sparse_matrix *D,
                    magma_s_sparse_matrix *R ){

    if(  A.memory_location == Magma_CPU &&
            (   A.storage_type == Magma_CSR ||
                A.storage_type == Magma_CSRCOO ) ){

        magma_int_t i, k, j, nnz_diag, nnz_offd;


        nnz_diag = nnz_offd = 0;
        // Count the new number of nonzeroes in the two matrices
        for( i=0; i<A.num_rows; i+=bsize )
            for( k=i; k<min(A.num_rows,i+bsize); k++ )
                for( j=A.row[k]; j<A.row[k+1]; j++ )
                if ( A.col[j] < i )
                    nnz_offd++;
                else if ( A.col[j] < i+bsize )
                    nnz_diag++;
                else
                    nnz_offd++;

        // Allocate memory for the new matrices
        D->storage_type = Magma_CSRD;
        D->memory_location = A.memory_location;
        D->num_rows = A.num_rows;
        D->num_cols = A.num_cols;
        D->nnz = nnz_diag;

        R->storage_type = Magma_CSR;
        R->memory_location = A.memory_location;
        R->num_rows = A.num_rows;
        R->num_cols = A.num_cols;
        R->nnz = nnz_offd;

        magma_smalloc_cpu( &D->val, nnz_diag );
        magma_indexmalloc_cpu( &D->row, A.num_rows+1 );
        magma_indexmalloc_cpu( &D->col, nnz_diag );

        magma_smalloc_cpu( &R->val, nnz_offd );
        magma_indexmalloc_cpu( &R->row, A.num_rows+1 );
        magma_indexmalloc_cpu( &R->col, nnz_offd );

        // Fill up the new sparse matrices  
        D->row[0] = 0;
        R->row[0] = 0;

        nnz_offd = nnz_diag = 0;
        for( i=0; i<A.num_rows; i+=bsize){
            for( k=i; k<min(A.num_rows,i+bsize); k++ ){
                D->row[k+1] = D->row[k];
                R->row[k+1] = R->row[k];
     
                for( j=A.row[k]; j<A.row[k+1]; j++ ){
                    if ( A.col[j] < i ){
                        R->val[nnz_offd] = A.val[j];
                        R->col[nnz_offd] = A.col[j];
                        R->row[k+1]++;  
                        nnz_offd++;
                    }
                    else if ( A.col[j] < i+bsize ){
                        // larger than diagonal remain as before
                        if ( A.col[j]>k ){
                            D->val[nnz_diag] = A.val[ j ];
                            D->col[nnz_diag] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        // diagonal is written first
                        else if ( A.col[j]==k ) {
                            D->val[D->row[k]] = A.val[ j ];
                            D->col[D->row[k]] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        // smaller than diagonal are shifted one to the right 
                        // to have room for the diagonal
                        else {
                            D->val[nnz_diag+1] = A.val[ j ];
                            D->col[nnz_diag+1] = A.col[ j ];
                            D->row[k+1]++;
                        }
                        nnz_diag++;
                    }
                    else {
                        R->val[nnz_offd] = A.val[j];
                        R->col[nnz_offd] = A.col[j];
                        R->row[k+1]++;  
                        nnz_offd++;
                    }
                }
            }
        }
        return MAGMA_SUCCESS; 
    }
    else{
        magma_s_sparse_matrix Ah, ACSR, DCSR, RCSR, Dh, Rh;
        magma_s_mtransfer( A, &Ah, A.memory_location, Magma_CPU );
        magma_s_mconvert( Ah, &ACSR, A.storage_type, Magma_CSR );

        magma_scsrsplit( bsize, ACSR, &DCSR, &RCSR );

        magma_s_mconvert( DCSR, &Dh, Magma_CSR, A.storage_type );
        magma_s_mconvert( RCSR, &Rh, Magma_CSR, A.storage_type );

        magma_s_mtransfer( Dh, D, Magma_CPU, A.memory_location );
        magma_s_mtransfer( Rh, R, Magma_CPU, A.memory_location );

        magma_s_mfree( &Ah );
        magma_s_mfree( &ACSR );
        magma_s_mfree( &Dh );
        magma_s_mfree( &DCSR );
        magma_s_mfree( &Rh );
        magma_s_mfree( &RCSR );

        return MAGMA_SUCCESS; 
    }
}



