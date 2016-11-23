/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/ztrtri_lower_batched.cu, normal z -> d, Sun Nov 20 20:20:30 2016

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       This file implements lower case, and is called by dtrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_BATCHED
#include "dtrtri.cuh"
#include "dtrtri_lower_device.cuh"



/******************************************************************************/
__global__ void
dtrtri_diag_lower_kernel_batched(
    magma_diag_t diag, int n, double const * const * dA_array, int lda, double **dinvA_array)
{
    int batchid = blockIdx.z;
    dtrtri_diag_lower_device(diag, n, dA_array[batchid], lda, dinvA_array[batchid]);
}


/******************************************************************************/
__global__ void
triple_dgemm16_part1_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm16_part1_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm16_part2_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm16_part2_lower_device( n,  Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm32_part1_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm32_part1_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm32_part2_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm32_part2_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm64_part1_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm64_part1_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm64_part2_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm64_part2_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part1_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm_above64_part1_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part2_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm_above64_part2_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part3_lower_kernel_batched(
    int n, double const * const * Ain_array, int lda, double **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_dgemm_above64_part3_lower_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


// =============================================================================
// vbatched kernels


/******************************************************************************/
__global__ void
dtrtri_diag_lower_kernel_vbatched(
    magma_diag_t diag, magma_int_t* n, double const * const * dA_array, magma_int_t* lda, double **dinvA_array)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return; 
    
    if(blockIdx.x >= magma_ceildiv(my_n, IB)) return;
    
    dtrtri_diag_lower_device(diag, my_n, dA_array[batchid], (int)lda[batchid], dinvA_array[batchid]);
}


// The kernels below have 3D grids
// grid.x and grid.y are independent from my_n
// only grid.y is dependent on my_n, so terminating thread blocks is based on blockIdx.y


/******************************************************************************/
__global__ void
triple_dgemm16_part1_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm16_part1_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm16_part2_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm16_part2_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm32_part1_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm32_part1_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm32_part2_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm32_part2_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm64_part1_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm64_part1_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm64_part2_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm64_part2_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part1_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm_above64_part1_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part2_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm_above64_part2_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_dgemm_above64_part3_lower_kernel_vbatched(
    magma_int_t* n, double const * const * Ain_array, magma_int_t* lda, double **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_dgemm_above64_part3_lower_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}
