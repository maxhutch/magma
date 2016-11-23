/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/ztrtri_upper.cu, normal z -> s, Sun Nov 20 20:20:31 2016

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
       This file implements upper case, and is called by strtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "strtri.cuh"
#include "strtri_upper_device.cuh"


/******************************************************************************/
__global__ void
strtri_diag_upper_kernel(
    magma_diag_t diag, int n, const float *A, int lda, float *d_dinvA)
{
    strtri_diag_upper_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_sgemm16_part1_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm16_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm16_part2_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm16_part2_upper_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm32_part1_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm32_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm32_part2_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm32_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm64_part1_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm64_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm64_part2_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm64_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm_above64_part1_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm_above64_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm_above64_part2_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm_above64_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_sgemm_above64_part3_upper_kernel(
    int n, const float *Ain, int lda, float *d_dinvA, int jb, int npages)
{
    triple_sgemm_above64_part3_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}
