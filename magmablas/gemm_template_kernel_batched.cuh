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

#ifndef GEMM_TEMPLATE_KERNEL_BATCHED_CUH
#define GEMM_TEMPLATE_KERNEL_BATCHED_CUH

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_template_device_defs.cuh"
#include "gemm_template_device.cuh"


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_nn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int offsetA, int offsetB )
{
    //if ( blockIdx.y > blockIdx.x ) return; //for lower blkx > blky do not have to compute
    int batchid = blockIdx.z;
    
    /*
    #ifdef TEXTURE_1D
    int matrixA_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    int matrixB_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    offsetA += batchid*matrixA_size;
    offsetB += batchid*matrixB_size;
    #endif
    */
    
    gemm_template_device_nn
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_nt_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int offsetA, int offsetB )
{
    //if ( blockIdx.y > blockIdx.x ) return; //for lower blkx > blky do not have to compute
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
        ( M, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_tn_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int offsetA, int offsetB )
{
    //if ( blockIdx.y > blockIdx.x ) return; //for lower blkx > blky do not have to compute
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
        ( M, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
static __global__
void gemm_template_batched_tt_kernel(
    int M, int N, int K,
    T const * const * Aarray, int LDA,
    T const * const * Barray, int LDB,
    T**       Carray, int LDC,
    T alpha, T beta,
    int offsetA, int offsetB )
{
    //if ( blockIdx.y > blockIdx.x ) return; //for lower blkx > blky do not have to compute
    int batchid = blockIdx.z;
    
    /*
    #ifdef TEXTURE_1D
    int matrixA_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    int matrixB_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    offsetA += batchid*matrixA_size;
    offsetB += batchid*matrixB_size;
    #endif
    */
    
    gemm_template_device_tt
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, (BLK_M/DIM_X), (BLK_N/DIM_Y), CONJA, CONJB>
        ( M, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////
// NN 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,  
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_nn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t offsetA, magma_int_t offsetB, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    gemm_template_batched_nn_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// NT, NC 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec, 
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_nt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t offsetA, magma_int_t offsetB, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    gemm_template_batched_nt_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// TN, CN 
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_tn(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t offsetA, magma_int_t offsetB, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    gemm_template_batched_tn_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// TT, TC, CT, CC
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, const int DIM_X, const int DIM_Y, const int BLK_M, const int BLK_N, const int BLK_K, const int dim_vec,
         const int DIM_XA, const int DIM_YA, const int DIM_XB, const int DIM_YB, 
         const int CONJA, const int CONJB>
void gemm_template_batched_tt(
    magma_int_t m, magma_int_t n, magma_int_t k,
    T const * const * dA_array, magma_int_t ldda,
    T const * const * dB_array, magma_int_t lddb,
    T**       dC_array, magma_int_t lddc,
    T alpha, T beta,
    magma_int_t offsetA, magma_int_t offsetB, 
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 dimBlock(DIM_X, DIM_Y);
    dim3 dimGrid( magma_ceildiv( m, BLK_M ), magma_ceildiv( n, BLK_N ), batchCount );
    gemm_template_batched_tt_kernel
        <T, DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, DIM_XA, DIM_YA, DIM_XB, DIM_YB, CONJA, CONJB>
        <<< dimGrid, dimBlock, 0, queue->cuda_stream() >>>
        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, offsetA, offsetB);
}

#endif //GEMM_TEMPLATE_KERNEL_BATCHED_CUH
