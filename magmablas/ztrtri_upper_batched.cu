/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
       This file implements upper case, and is called by ztrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_BATCHED
#include "ztrtri.cuh"
#include "ztrtri_upper_device.cuh"


/******************************************************************************/
__global__ void
ztrtri_diag_upper_kernel_batched(
    magma_diag_t diag, int n, magmaDoubleComplex const * const * dA_array, int lda, magmaDoubleComplex **dinvA_array)
{
    int batchid = blockIdx.z;
    ztrtri_diag_upper_device(diag, n, dA_array[batchid], lda, dinvA_array[batchid]);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm16_part1_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm16_part2_upper_device( n,  Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm32_part1_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm32_part2_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm64_part1_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm64_part2_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm_above64_part1_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm_above64_part2_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part3_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    int batchid = blockIdx.z;
    triple_zgemm_above64_part3_upper_device( n, Ain_array[batchid], lda, dinvA_array[batchid], jb, npages);
}


// =============================================================================
// vbatched kernels


/******************************************************************************/
__global__ void
ztrtri_diag_upper_kernel_vbatched(
    magma_diag_t diag, magma_int_t* n, magmaDoubleComplex const * const * dA_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    if(blockIdx.x >= magma_ceildiv(my_n, IB)) return;
    
    ztrtri_diag_upper_device(diag, my_n, dA_array[batchid], (int)lda[batchid], dinvA_array[batchid]);
}


// The kernels below have 3D grids
// grid.x and grid.y are independent from my_n
// only grid.y is dependent on my_n, so terminating thread blocks is based on blockIdx.y


/******************************************************************************/
__global__ void
triple_zgemm16_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm16_part1_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm16_part2_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm32_part1_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm32_part2_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm64_part1_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm64_part2_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm_above64_part1_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm_above64_part2_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part3_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages)
{
    const int batchid = blockIdx.z;
    const int my_n = (int)n[batchid];
    if(my_n <= 0) return;
    
    const int my_npages = magma_ceildiv(my_n, jb*2);
    if(blockIdx.y >= my_npages*(jb/16) ) return;
    triple_zgemm_above64_part3_upper_device( my_n, Ain_array[batchid], (int)lda[batchid], dinvA_array[batchid], jb, my_npages);
}
