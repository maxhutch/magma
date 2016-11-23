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
       
       This file implements lower case, and is called by ztrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "ztrtri.cuh"
#include "ztrtri_lower_device.cuh"


/******************************************************************************/
__global__ void
ztrtri_diag_lower_kernel(
    magma_diag_t diag, int n, const magmaDoubleComplex *A, int lda, magmaDoubleComplex *d_dinvA)
{
    ztrtri_diag_lower_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm16_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm16_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm16_part2_lower_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm32_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm32_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm32_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm64_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm_above64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm_above64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_zgemm_above64_part3_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    triple_zgemm_above64_part3_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}
