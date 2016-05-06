/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#ifndef HERK_TEMPLATE_KERNEL_BATCHED_CUH
#define HERK_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "gemm_template_device.cuh"


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void herk_template_batched_nt_kernel(
    magma_uplo_t uplo, int N, int K, 
    T alpha, T const * const * Aarray, int LDA,
    T beta, T**       Carray, int LDC,
    int offsetA, int offsetB)
{
    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ( ( uplo == MagmaLower ) && ( blockIdx.y*BLK_N > (blockIdx.x+1)*BLK_M ) )
        return;
    
    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ( ( uplo == MagmaUpper)  && ( blockIdx.x*BLK_M > (blockIdx.y+1)*BLK_N ) )
        return;

    int batchid = blockIdx.z;
    
    /*
    #ifdef TEXTURE_1D
    int matrixA_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    int matrixB_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    offsetA += batchid*matrixA_size;
    offsetB += batchid*matrixB_size;
    #endif
    */
    
    gemm_template_device_nt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( N, N, K, Aarray[batchid], LDA, Aarray[batchid], LDA, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void herk_template_batched_tn_kernel(
    magma_uplo_t uplo, int N, int K, 
    T alpha, T const * const * Aarray, int LDA,
    T beta, T**       Carray, int LDC,
    int offsetA, int offsetB )
{
    // for lower: each thread-block checks its bottom left corner of its corresponding C block
    if ( ( uplo == MagmaLower ) && ( blockIdx.y*BLK_N > (blockIdx.x+1)*BLK_M ) )
        return;
    
    // for upper: each thread-block checks its top right corner of its corresponding C block
    if ( ( uplo == MagmaUpper)  && ( blockIdx.x*BLK_M > (blockIdx.y+1)*BLK_N ) )
        return;

    int batchid = blockIdx.z;
    
    /*
    #ifdef TEXTURE_1D
    int matrixA_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    int matrixB_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    offsetA += batchid*matrixA_size;
    offsetB += batchid*matrixB_size;
    #endif
    */
    
    gemm_template_device_tn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( N, N, K, Aarray[batchid], LDA, Aarray[batchid], LDA, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// NT, NC 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void herk_template_batched_nt(
    magma_uplo_t uplo, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t offsetA, magma_int_t offsetB, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( n, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    herk_template_batched_nt_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (uplo, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, offsetA, offsetB);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// TN, CN 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void herk_template_batched_tn(
    magma_uplo_t uplo, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t offsetA, magma_int_t offsetB, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( n, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    herk_template_batched_tn_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (uplo, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, offsetA, offsetB);
}

#endif //HERK_TEMPLATE_KERNEL_BATCHED_CUH
