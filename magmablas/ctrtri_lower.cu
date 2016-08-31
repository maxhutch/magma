/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from magmablas/ztrtri_lower.cu, normal z -> c, Tue Aug 30 09:38:35 2016

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       This file implements lower case, and is called by ctrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

#define TRTRI_NONBATCHED
#include "ctrtri.cuh"
#include "ctrtri_lower_device.cuh"


/******************************************************************************/
__global__ void
ctrtri_diag_lower_kernel(
    magma_diag_t diag, int n, const magmaFloatComplex *A, int lda, magmaFloatComplex *d_dinvA)
{
    ctrtri_diag_lower_device(diag, n, A, lda, d_dinvA);
}


/******************************************************************************/
__global__ void
triple_cgemm16_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm16_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm16_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm16_part2_lower_device( n,  Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm32_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm32_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm32_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm32_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm64_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm64_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm_above64_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm_above64_part1_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm_above64_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm_above64_part2_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}


/******************************************************************************/
__global__ void
triple_cgemm_above64_part3_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_dinvA, int jb, int npages)
{
    triple_cgemm_above64_part3_lower_device( n, Ain, lda, d_dinvA, jb, npages);
}
