/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/blas/zgeisai.cu, normal z -> c, Sun Nov 20 20:20:41 2016

*/
#include "magmasparse_internal.h"

#define PRECISION_c
#define COMPLEX
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4


// initialize arrays with zero
__global__ void
magma_cgpumemzero_z(
    magmaFloatComplex * d,
    int n,
    int dim_x,
    int dim_y )
{
    int i = blockIdx.y * gridDim.x + blockIdx.x;
    int idx = threadIdx.x;

    if( i >= n ){
       return;
    }
    if( idx >= dim_x ){
       return;
    }

    for( int j=0; j<dim_y; j++)
        d[ i*dim_x*dim_y + j*dim_y + idx ] = MAGMA_C_MAKE( 0.0, 0.0 );
}

__global__ void
magma_clocations_lower_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE ] = MAGMA_C_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel


__global__ void
magma_clocations_trunc_lower_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_C_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 32 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_C_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j+1]-BLOCKSIZE+i ];
    }
}// kernel



__global__ void
magma_clocations_upper_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE+count-1 ] = MAGMA_C_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_clocations_trunc_upper_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_C_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 32 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_C_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_cfilltrisystems_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x);

    if ( i>=n ){
        return;
    }
    for( int j=0; j<sizes[ i ]; j++ ){// no need for first
        int k = row[ locations[ j+i*WARP_SIZE ] ];
        int l = i*WARP_SIZE;
        int idx = 0;
        while( k < row[ locations[ j+i*WARP_SIZE ]+1 ] && l < (i+1)*WARP_SIZE ){ // stop once this column is done
            if( locations[ l ] == col[k] ){ //match
                // int loc = i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx;
                trisystems[ i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx ]
                                                        = val[ k ];
                k++;
                l++;
                idx++;
            } else if( col[k] < locations[ l ] ){// need to check next element
                k++;
            } else { // element does not exist, i.e. l < LC.col[k]
                // printf("increment l\n");
                l++; // check next elment in the sparsity pattern
                idx++; // leave this element equal zero
            }
        }
    }
}// kernel


/**
    Purpose
    -------

    This routine prepares the batch of small triangular systems that
    need to be solved for computing the ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    uplotype    magma_uplo_t
                input matrix

    @param[in]
    transtype   magma_trans_t
                input matrix

    @param[in]
    diagtype    magma_diag_t
                input matrix

    @param[in]
    L           magma_c_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.

    @param[in]
    LC          magma_c_matrix
                sparsity pattern of the ISAI matrix.
                Col-Major CSR storage.

    @param[in,out]
    sizes       magma_index_t*
                array containing the sizes of the small triangular systems

    @param[in,out]
    locations   magma_index_t*
                array containing the locations in the respective column of L

    @param[in,out]
    trisystems  magmaFloatComplex*
                batch of generated small triangular systems. All systems are
                embedded in uniform memory blocks of size BLOCKSIZE x BLOCKSIZE

    @param[in,out]
    rhs         magmaFloatComplex*
                RHS of the small triangular systems

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_cmprepare_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_c_matrix L,
    magma_c_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs,
    magma_queue_t queue )
{
    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( float( LC.num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( LC.num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( LC.num_rows, dimgrid1*dimgrid2 );
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );


    int blocksize21 = BLOCKSIZE;
    int blocksize22 = 1;

    int dimgrid21 = magma_ceildiv( LC.num_rows, blocksize21 );
    int dimgrid22 = 1;
    int dimgrid23 = 1;
    dim3 grid2( dimgrid21, dimgrid22, dimgrid23 );
    dim3 block2( blocksize21, blocksize22, 1 );

    magma_cgpumemzero_z<<< grid, block, 0, queue->cuda_stream() >>>(
        trisystems, LC.num_rows, WARP_SIZE, WARP_SIZE );

    magma_cgpumemzero_z<<< grid, block, 0, queue->cuda_stream() >>>(
        rhs, LC.num_rows, WARP_SIZE, 1);


   // magma_cprint_gpu( 32, 32, L.dval, 32, queue );

    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    //cudaFuncSetCacheConfig(cudaFuncCachePreferShared);


    if( uplotype == MagmaLower ){
        magma_clocations_lower_kernel<<< grid, block, 0, queue->cuda_stream() >>>(
                        LC.num_rows,
                        LC.drow,
                        LC.dcol,
                        LC.dval,
                        sizes,
                        locations,
                        trisystems,
                        rhs );
    } else {
        magma_clocations_upper_kernel<<< grid, block, 0, queue->cuda_stream() >>>(
                        LC.num_rows,
                        LC.drow,
                        LC.dcol,
                        LC.dval,
                        sizes,
                        locations,
                        trisystems,
                        rhs );
    }

    // magma_cprint_gpu( 32, 32, L.dval, 32, queue );


    magma_cfilltrisystems_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>(
                        L.num_rows,
                        L.drow,
                        L.dcol,
                        L.dval,
                        sizes,
                        locations,
                        trisystems,
                        rhs );
    //magma_cprint_gpu( 32, 32, L.dval, 32, queue );

    return MAGMA_SUCCESS;
}


__global__ void
magma_cbackinsert_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int end = sizes[j];
    if( j >= n ){
        return;
    }

    if ( i>=end ){
        return;
    }

    val[row[j]+i] = rhs[j*WARP_SIZE+i];
}// kernel



/**
    Purpose
    -------
    Inserts the values into the preconditioner matrix

    Arguments
    ---------


    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular

    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix

    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not

    @param[in,out]
    M           magma_c_matrix*
                SPAI preconditioner CSR col-major

    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[out]
    trisystems  magmaFloatComplex*
                trisystems

    @param[out]
    rhs         magmaFloatComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmbackinsert_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_c_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( float( M->num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( M->num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( M->num_rows, dimgrid1*dimgrid2 );

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );

    magma_cbackinsert_kernel<<< grid, block, 0, queue->cuda_stream() >>>(
            M->num_rows,
            M->drow,
            M->dcol,
            M->dval,
            sizes,
            rhs );

    return info;
}
