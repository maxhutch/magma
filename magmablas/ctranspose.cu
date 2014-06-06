/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:45 2013

       @author Stan Tomov
*/
#include "common_magma.h"
#define PRECISION_c
#include "commonblas.h"

__global__ void ctranspose_32( magmaFloatComplex *B, int ldb, const magmaFloatComplex *A, int lda )
{        
        __shared__ magmaFloatComplex a[32][CSIZE_1SHARED+1];
        
        int inx = threadIdx.x;
        int iny = threadIdx.y;
        int ibx = blockIdx.x*32;
        int iby = blockIdx.y*32;
        
        A += ibx + inx + __mul24( iby + iny, lda );
        B += iby + inx + __mul24( ibx + iny, ldb );
        
        a[iny+0][inx] = A[0*lda];
        a[iny+8][inx] = A[8*lda];
        a[iny+16][inx] = A[16*lda];
        a[iny+24][inx] = A[24*lda];
        
        __syncthreads();
        
#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c)
        B[0*ldb] = a[inx][iny+0];
        B[8*ldb] = a[inx][iny+8];
        B[16*ldb] = a[inx][iny+16];
        B[24*ldb] = a[inx][iny+24];
#else /* defined(PRECISION_z) */
        B[0*ldb]    = a[inx][iny+0];
        B[8*ldb]    = a[inx][iny+8];
        B[0*ldb+16] = a[inx+16][iny+0];
        B[8*ldb+16] = a[inx+16][iny+8];

        __syncthreads();
        A += CSIZE_1SHARED;
        B += __mul24( 16, ldb);

        a[iny+0][inx] = A[0*lda];
        a[iny+8][inx] = A[8*lda];
        a[iny+16][inx] = A[16*lda];
        a[iny+24][inx] = A[24*lda];

        __syncthreads();

        B[0*ldb] = a[inx][iny+0];
        B[8*ldb] = a[inx][iny+8];
        B[0*ldb+16] = a[inx+16][iny+0];
        B[8*ldb+16] = a[inx+16][iny+8];
#endif
} 

//
//        m, n - dimensions in the source matrix
//             This version works when m and n are divisible by 32.
//
extern "C" void 
magmablas_ctranspose(magmaFloatComplex       *odata, magma_int_t ldo, 
                     const magmaFloatComplex *idata, magma_int_t ldi, 
                     magma_int_t m, magma_int_t n )
{
        //assert( (m%32) == 0 && (n%32) == 0, "misaligned transpose" );
        dim3 threads( CSIZE_1SHARED, 8, 1 );
        dim3 grid( m/32, n/32, 1 );
        ctranspose_32<<< grid, threads, 0, magma_stream >>>( odata, ldo, idata, ldi );
}
