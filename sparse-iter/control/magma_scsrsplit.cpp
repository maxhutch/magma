/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zcsrsplit.cpp normal z -> s, Sun May  3 11:23:01 2015
       @author Hartwig Anzt

*/
#include "common_magmasparse.h"


/**
    Purpose
    -------

    Splits a CSR matrix into two matrices, one containing the diagonal blocks
    with the diagonal element stored first, one containing the rest of the
    original matrix.

    Arguments
    ---------

    @param[in]
    bsize       magma_int_t
                size of the diagonal blocks

    @param[in]
    A           magma_s_matrix
                CSR input matrix

    @param[out]
    D           magma_s_matrix*
                CSR matrix containing diagonal blocks

    @param[out]
    R           magma_s_matrix*
                CSR matrix containing rest
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_scsrsplit(
    magma_int_t bsize,
    magma_s_matrix A,
    magma_s_matrix *D,
    magma_s_matrix *R,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i, k, j, nnz_diag, nnz_offd;
    
    D->val = NULL;
    D->col = NULL;
    D->row = NULL;
    D->rowidx = NULL;
    D->blockinfo = NULL;
    D->diag = NULL;
    D->dval = NULL;
    D->dcol = NULL;
    D->drow = NULL;
    D->drowidx = NULL;
    D->ddiag = NULL;
    R->val = NULL;
    R->col = NULL;
    R->row = NULL;
    R->rowidx = NULL;
    R->blockinfo = NULL;
    R->diag = NULL;
    R->dval = NULL;
    R->dcol = NULL;
    R->drow = NULL;
    R->drowidx = NULL;
    R->ddiag = NULL;
    
    if (  A.memory_location == Magma_CPU &&
            (   A.storage_type == Magma_CSR ||
                A.storage_type == Magma_CSRCOO ) ) {



        nnz_diag = nnz_offd = 0;
        // Count the new number of nonzeroes in the two matrices
        for( i=0; i<A.num_rows; i+=bsize ){
            for( k=i; k<min(A.num_rows,i+bsize); k++ ){
                int check = 0;
                for( j=A.row[k]; j<A.row[k+1]; j++ ){
                    if ( A.col[j] < i )
                        nnz_offd++;
                    else if ( A.col[j] < i+bsize ){
                        if( A.col[j] == k ){
                            check = 1;
                        }
                        nnz_diag++;
                    }
                    else
                        nnz_offd++;
                }
                if( check == 0 ){
                    printf("error: matrix contains zero on diagonal at (%d,%d).\n", i, i);
                    info = -1;
                    goto cleanup;
                }
            }
        }

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

        CHECK( magma_smalloc_cpu( &D->val, nnz_diag ));
        CHECK( magma_index_malloc_cpu( &D->row, A.num_rows+1 ));
        CHECK( magma_index_malloc_cpu( &D->col, nnz_diag ));

        CHECK( magma_smalloc_cpu( &R->val, nnz_offd ));
        CHECK( magma_index_malloc_cpu( &R->row, A.num_rows+1 ));
        CHECK( magma_index_malloc_cpu( &R->col, nnz_offd ));
        
        // Fill up the new sparse matrices
        D->row[0] = 0;
        R->row[0] = 0;

        nnz_offd = nnz_diag = 0;
        for( i=0; i<A.num_rows; i+=bsize) {
            for( k=i; k<min(A.num_rows,i+bsize); k++ ) {
                D->row[k+1] = D->row[k];
                R->row[k+1] = R->row[k];
     
                for( j=A.row[k]; j<A.row[k+1]; j++ ) {
                    if ( A.col[j] < i ) {
                        R->val[nnz_offd] = A.val[j];
                        R->col[nnz_offd] = A.col[j];
                        R->row[k+1]++;
                        nnz_offd++;
                    }
                    else if ( A.col[j] < i+bsize ) {
                        // larger than diagonal remain as before
                        if ( A.col[j]>k ) {
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
    }
    else {
        magma_s_matrix Ah={Magma_CSR}, ACSR={Magma_CSR}, DCSR={Magma_CSR}, RCSR={Magma_CSR}, Dh={Magma_CSR}, Rh={Magma_CSR};
        CHECK( magma_smtransfer( A, &Ah, A.memory_location, Magma_CPU, queue ));
        CHECK( magma_smconvert( Ah, &ACSR, A.storage_type, Magma_CSR, queue ));

        CHECK( magma_scsrsplit( bsize, ACSR, &DCSR, &RCSR, queue ));

        CHECK( magma_smconvert( DCSR, &Dh, Magma_CSR, A.storage_type, queue ));
        CHECK( magma_smconvert( RCSR, &Rh, Magma_CSR, A.storage_type, queue ));

        CHECK( magma_smtransfer( Dh, D, Magma_CPU, A.memory_location, queue ));
        CHECK( magma_smtransfer( Rh, R, Magma_CPU, A.memory_location, queue ));

        magma_smfree( &Ah, queue );
        magma_smfree( &ACSR, queue );
        magma_smfree( &Dh, queue );
        magma_smfree( &DCSR, queue );
        magma_smfree( &Rh, queue );
        magma_smfree( &RCSR, queue );
    }
cleanup:
    if( info != 0 ){
        magma_smfree( D, queue );
        magma_smfree( R, queue );
    }
    return info;
}



