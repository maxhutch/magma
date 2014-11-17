/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_z_free.cpp normal z -> c, Sat Nov 15 19:54:23 2014
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
#include "magmasparse_c.h"
#include "magma.h"
#include "mmio.h"



using namespace std;








/**
    Purpose
    -------

    Free the memory of a magma_c_vector.


    Arguments
    ---------

    @param[i,out]
    x           magma_c_vector*
                vector to free    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_vfree(
    magma_c_vector *x,
    magma_queue_t queue )
{
    if ( x->memory_location == Magma_CPU ) {
        magma_free_cpu( x->val );
        x->num_rows = 0;
        x->nnz = 0;
        x->val = NULL;
        return MAGMA_SUCCESS;     
    }
    else if ( x->memory_location == Magma_DEV ) {
        if ( magma_free( x->dval ) != MAGMA_SUCCESS ) {
            printf("Memory Free Error.\n");  
            return MAGMA_ERR_INVALID_PTR;
        }
        
        x->num_rows = 0;
        x->nnz = 0;
        x->dval = NULL;
        return MAGMA_SUCCESS;     
    }
    else {
        printf("Memory Free Error.\n");  
        return MAGMA_ERR_INVALID_PTR;
    }
}


/**
    Purpose
    -------

    Free the memory of a magma_c_sparse_matrix.


    Arguments
    ---------

    @param[in,out]
    A           magma_c_sparse_matrix*
                matrix to free    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_mfree(
    magma_c_sparse_matrix *A,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU ) {
       if ( A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT ){
            free( A->val );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if (A->storage_type == Magma_ELLD ) {
            free( A->val );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                       
        } 
        if ( A->storage_type == Magma_ELLRT ) {
            free( A->val );
            free( A->row );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                        
        } 
        if ( A->storage_type == Magma_SELLP ) {
            free( A->val );
            free( A->row );
            free( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if ( A->storage_type == Magma_CSR || A->storage_type == Magma_CSC 
                                        || A->storage_type == Magma_CSRD
                                        || A->storage_type == Magma_CSRL
                                        || A->storage_type == Magma_CSRU ) {
            free( A->val );
            free( A->col );
            free( A->row );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
        } 
        if (  A->storage_type == Magma_CSRCOO ) {
            free( A->val );
            free( A->col );
            free( A->row );
            free( A->rowidx );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                     
        } 
        if ( A->storage_type == Magma_BCSR ) {
            free( A->val );
            free( A->col );
            free( A->row );
            free( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; 
            A->blockinfo = 0;                    
        } 
        if ( A->storage_type == Magma_DENSE ) {
            free( A->val );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                      
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
        return MAGMA_SUCCESS; 
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
            A->nnz = 0;                    
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
            A->nnz = 0;                   
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
            A->nnz = 0;                         
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
            A->nnz = 0;                  
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
            A->nnz = 0;                    
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
            A->nnz = 0;                      
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
            free( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;                  
        } 
        if ( A->storage_type == Magma_DENSE ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");  
                return MAGMA_ERR_INVALID_PTR;
                
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0;        
                
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
        return MAGMA_SUCCESS; 
    }

    else {
        printf("Memory Free Error.\n");  
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}



   


