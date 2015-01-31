/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zmcsrcompressor_gpu.cu normal z -> c, Fri Jan 30 19:00:29 2015
       @author Hartwig Anzt

*/

#include "common_magma.h"
#include "magmasparse.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE1 256
   #define BLOCK_SIZE2 1
#else
   #define BLOCK_SIZE1 256
   #define BLOCK_SIZE2 1
#endif


// copy nonzeros into new structure
__global__ void
magma_cmcsrgpu_kernel1( int num_rows,  
                 magmaFloatComplex *A_val, 
                 magma_index_t *A_rowptr, 
                 magma_index_t *A_colind,
                 magmaFloatComplex *B_val, 
                 magma_index_t *B_rowptr, 
                 magma_index_t *B_colind ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaFloatComplex zero = MAGMA_C_ZERO;
        int start = A_rowptr[ row ];
        int new_location = start;
        int end = A_rowptr[ row+1 ];
        for( j=start; j<end; j++ ){
            if( A_val[j] != zero ){
       //         B_val[new_location] = A_val[j];
       //         B_colind[new_location] = A_colind[j];
                new_location++;
            } 
        }
        // this is not a correctr rowpointer! this is nn_z in this row!
        B_rowptr[ row ] = new_location-start;
    }
}


// generate a valid rowpointer
__global__ void
magma_cmcsrgpu_kernel2( int num_rows,  
                 magma_index_t *B_rowptr,
                 magma_index_t *A_rowptr ){

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int j, nnz = 0;

    if( idx == 0 ){
    A_rowptr[ 0 ] = nnz;
        for( j=0; j<num_rows; j++ ){
            nnz+=B_rowptr[ j ];
            A_rowptr[ j+1 ] = nnz;
        }
    }
}



// copy new structure into original matrix
__global__ void
magma_cmcsrgpu_kernel3( int num_rows,  
                 magmaFloatComplex *B_val, 
                 magma_index_t *B_rowptr, 
                 magma_index_t *B_colind,
                 magma_index_t *B2_rowptr, 
                 magmaFloatComplex *A_val, 
                 magma_index_t *A_rowptr, 
                 magma_index_t *A_colind
                                            ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j, new_location;
    
    if(row<num_rows){
    new_location = A_rowptr[ row ];
        int start = B2_rowptr[ row ];
        int end = B2_rowptr[ row+1 ];
        magmaFloatComplex zero = MAGMA_C_ZERO;
        for( j=start; j<end; j++ ){
            if( A_val[j] != zero ){
                B_val[new_location] = A_val[j];
                B_colind[new_location] = A_colind[j];
                new_location++;
            } 
               // A_val[ j ] = B_val[ j ];
               // A_colind[ j ] = B_colind[ j ];
        }
    }
}


/**
    Purpose
    -------

    Removes zeros in a CSR matrix. This is a GPU implementation of the 
    CSR compressor.

    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix*
                input/output matrix 
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmcsrcompressor_gpu(
    magma_c_sparse_matrix *A,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_DEV && A->storage_type == Magma_CSR ) {

        magma_int_t stat_cpu = 0, stat_dev = 0;
        magma_c_sparse_matrix B, B2;
        
        B.val = NULL;
        B.col = NULL;
        B.row = NULL;
        B.rowidx = NULL;
        B.blockinfo = NULL;
        B.diag = NULL;
        B.dval = NULL;
        B.dcol = NULL;
        B.drow = NULL;
        B.drowidx = NULL;
        B.ddiag = NULL;
        
        B2.val = NULL;
        B2.col = NULL;
        B2.row = NULL;
        B2.rowidx = NULL;
        B2.blockinfo = NULL;
        B2.diag = NULL;
        B2.dval = NULL;
        B2.dcol = NULL;
        B2.drow = NULL;
        B2.drowidx = NULL;
        B2.ddiag = NULL;

        stat_dev += magma_index_malloc( &B.drow, A->num_rows + 1 );
        stat_dev += magma_index_malloc( &B2.drow, A->num_rows + 1 );
        if( stat_dev != 0 ){         
            magma_c_mfree( &B, queue );
            magma_c_mfree( &B2, queue ); 
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        
        magma_index_copyvector( (A->num_rows+1), A->drow, 1, B2.drow, 1 );

        dim3 grid1( (A->num_rows+BLOCK_SIZE1-1)/BLOCK_SIZE1, 1, 1);  

        // copying the nonzeros into B and write in B.drow how many there are
        magma_cmcsrgpu_kernel1<<< grid1, BLOCK_SIZE1, 0, queue >>>
                ( A->num_rows, A->dval, A->drow, A->dcol, B.dval, B.drow, B.dcol );

        // correct the row pointer
        dim3 grid2( 1, 1, 1);  
        magma_cmcsrgpu_kernel2<<< grid2, BLOCK_SIZE2, 0, queue >>>
                ( A->num_rows, B.drow, A->drow );
        // access the true number of nonzeros
        magma_index_t *cputmp;
        stat_cpu += magma_index_malloc_cpu( &cputmp, 1 );
        if( stat_cpu != 0 ){
            magma_free_cpu( cputmp );
            magma_c_mfree( &B, queue );
            magma_c_mfree( &B2, queue );
            return MAGMA_ERR_HOST_ALLOC;
        }
        magma_index_getvector( 1, A->row+(A->num_rows), 1, cputmp, 1 );
        A->nnz = (magma_int_t) cputmp[0];

        // reallocate with right size
        stat_dev += magma_cmalloc( &B.dval, A->nnz );
        stat_dev += magma_index_malloc( &B.dcol, A->nnz );
        if( stat_dev != 0 ){         
            magma_c_mfree( &B, queue );
            magma_c_mfree( &B2, queue ); 
            return MAGMA_ERR_DEVICE_ALLOC;
        }
        
        // copy correct values back
        magma_cmcsrgpu_kernel3<<< grid1, BLOCK_SIZE1, 0, queue >>>
                ( A->num_rows, B.dval, B.drow, B.dcol, B2.drow, A->dval, A->drow, A->dcol );

        magma_free( A->dcol );
        magma_free( A->dval );                

        A->dcol = B.dcol;
        A->dval = B.dval;

        magma_free( B2.drow );
        magma_free( B.drow );  


        return MAGMA_SUCCESS; 
    }
    else {

        magma_c_sparse_matrix dA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_c_mconvert( *A, &CSRA, A->storage_type, Magma_CSR, queue );
        magma_c_mtransfer( *A, &dA, A->memory_location, Magma_DEV, queue );

        magma_cmcsrcompressor_gpu( &dA, queue );

        magma_c_mfree( &dA, queue );
        magma_c_mfree( A, queue );
        magma_c_mtransfer( dA, &CSRA, Magma_DEV, A_location, queue );
        magma_c_mconvert( CSRA, A, Magma_CSR, A_storage, queue );
        magma_c_mfree( &dA, queue );
        magma_c_mfree( &CSRA, queue );    

        return MAGMA_SUCCESS; 
    }
}


