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

#define TRTRI_NONBATCHED
#include "ztrtri.cuh"
#include "ztrtri_upper_device.cuh"


/******************************************************************************/
__global__ void
ztrtri_diag_upper_kernel(
    magma_diag_t diag, int n, const magmaDoubleComplex *A, int lda, magmaDoubleComplex *d_dinvA)
{
    ztrtri_diag_upper_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm16_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm16_part2_upper_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm32_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm32_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm64_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm64_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm_above64_part1_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm_above64_part2_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part3_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm_above64_part3_upper_device( n, Ain, lda, d_dinvA, jb, npages);
}
