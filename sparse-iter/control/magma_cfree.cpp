/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/control/magma_zfree.cpp normal z -> c, Mon May  2 23:30:50 2016
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Free the memory of a magma_c_matrix.


    Arguments
    ---------

    @param[in,out]
    A           magma_c_matrix*
                matrix to free
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmfree(
    magma_c_matrix *A,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU ) {
       if ( A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT ){
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (A->storage_type == Magma_ELLD ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLRT ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_SELLP ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSRLIST ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            magma_free_cpu( A->list );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC
                                        || A->storage_type == Magma_CSRD
                                        || A->storage_type == Magma_CSRL
                                        || A->storage_type == Magma_CSRU ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (  A->storage_type == Magma_CSRCOO ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            magma_free_cpu( A->rowidx );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_BCSR ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            magma_free_cpu( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
            A->blockinfo = 0;
        }
        if ( A->storage_type == Magma_DENSE ) {
            magma_free_cpu( A->val );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        A->val = NULL;
        A->col = NULL;
        A->row = NULL;
        A->rowidx = NULL;
        A->blockinfo = NULL;
        A->diag = NULL;
        A->dval = NULL;
        A->dcol = NULL;
        A->drow = NULL;
        A->drowidx = NULL;
        A->ddiag = NULL;
        A->dlist = NULL;
        A->list = NULL;
    }

    if ( A->memory_location == Magma_DEV ) {
       if ( A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT ){
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLD ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLRT ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_SELLP ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSRLIST ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dlist ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC
                                        || A->storage_type == Magma_CSRD
                                        || A->storage_type == Magma_CSRL
                                        || A->storage_type == Magma_CSRU ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (  A->storage_type == Magma_CSRCOO ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drowidx ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_BCSR ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            magma_free_cpu( A->blockinfo );
            A->blockinfo = NULL;
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_DENSE ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        A->val = NULL;
        A->col = NULL;
        A->row = NULL;
        A->rowidx = NULL;
        A->blockinfo = NULL;
        A->diag = NULL;
        A->dval = NULL;
        A->dcol = NULL;
        A->drow = NULL;
        A->drowidx = NULL;
        A->ddiag = NULL;
        A->dlist = NULL;
        A->list = NULL;
    }

    else {
        // printf("Memory Free Error.\n");
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}
