/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from ztrtri.h normal z -> c, Fri Jan 30 19:00:10 2015

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       Definitions used in ctrtri_diag.cu ctrtri_lower.cu ctrtri_upper.cu
*/

#ifndef CTRTRI_H
#define CTRTRI_H

// TODO this IB is a half warp -- that seems really ineffficient!
#define IB 16   // inner blocking size, <= 32
#define NB 128  // outer blocking size, > IB


/*
 * caxpy16 computes c += alpha*b, where b and c are 16-element vectors.
 */
static __device__ void
caxpy16(
    magmaFloatComplex alpha,
    const magmaFloatComplex * __restrict__ b,
    magmaFloatComplex       * __restrict__ c )
{
    c[0]  += alpha * b[0];
    c[1]  += alpha * b[1];
    c[2]  += alpha * b[2];
    c[3]  += alpha * b[3];
    c[4]  += alpha * b[4];
    c[5]  += alpha * b[5];
    c[6]  += alpha * b[6];
    c[7]  += alpha * b[7];
    c[8]  += alpha * b[8];
    c[9]  += alpha * b[9];
    c[10] += alpha * b[10];
    c[11] += alpha * b[11];
    c[12] += alpha * b[12];
    c[13] += alpha * b[13];
    c[14] += alpha * b[14];
    c[15] += alpha * b[15];
}


// unused -- but nearly identical code throughout ctrtri_lower & upper.cu
static __device__ void
cgemm_kernel_16(
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *B, int ldb,
    magmaFloatComplex *C, int ldc,
    magmaFloatComplex alpha, int jb, int tx, int ty)
{
    const magmaFloatComplex *Blast = B + jb;
    __shared__ magmaFloatComplex sB[16][17];
    
    // compute NT x 16 block of C
    // each thread computes one 1x16 row, C(id,0:15)
    magmaFloatComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    magmaFloatComplex rA[4];

    do {
        // load 16 x 16 block of B using NX x NY threads
        #pragma unroll
        for( int i=0; i < 16; i += blockDim.x ) {
            #pragma unroll
            for( int j=0; j < 16; j += blockDim.y ) {
                sB[tx + i][ty + j] = B[i + j*ldb];
            }
        }
        __syncthreads();
        
        // load NT x 16 block of A; each thread initially loads 1x4 row,
        // then continues loading more elements as axpys are done.
        rA[0] = A[0*lda];
        rA[1] = A[1*lda];
        rA[2] = A[2*lda];
        rA[3] = A[3*lda];

        // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
        caxpy16( rA[0], &sB[ 0][0], rC );  rA[0] = A[ 4*lda];
        caxpy16( rA[1], &sB[ 1][0], rC );  rA[1] = A[ 5*lda];
        caxpy16( rA[2], &sB[ 2][0], rC );  rA[2] = A[ 6*lda];
        caxpy16( rA[3], &sB[ 3][0], rC );  rA[3] = A[ 7*lda];
                                                     
        caxpy16( rA[0], &sB[ 4][0], rC );  rA[0] = A[ 8*lda];
        caxpy16( rA[1], &sB[ 5][0], rC );  rA[1] = A[ 9*lda];
        caxpy16( rA[2], &sB[ 6][0], rC );  rA[2] = A[10*lda];
        caxpy16( rA[3], &sB[ 7][0], rC );  rA[3] = A[11*lda];

        caxpy16( rA[0], &sB[ 8][0], rC );  rA[0] = A[12*lda];
        caxpy16( rA[1], &sB[ 9][0], rC );  rA[1] = A[13*lda];
        caxpy16( rA[2], &sB[10][0], rC );  rA[2] = A[14*lda];
        caxpy16( rA[3], &sB[11][0], rC );  rA[3] = A[15*lda];

        caxpy16( rA[0], &sB[12][0], rC );
        caxpy16( rA[1], &sB[13][0], rC );
        caxpy16( rA[2], &sB[14][0], rC );
        caxpy16( rA[3], &sB[15][0], rC );

        // move to next block of A and B
        A += 16*lda;
        B += 16;
        __syncthreads();
    } while( B < Blast );

    // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
    for( int i = 0; i < 16; i++ ) {
        C[0] = alpha*rC[i];
        C += ldc;
    }
}


__global__ void
ctrtri_diag_lower_kernel(
    magma_diag_t diag, int n, const magmaFloatComplex *A, int lda, magmaFloatComplex *d_invA);

__global__ void
triple_cgemm16_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm16_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm32_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm32_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm64_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm64_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm_above64_part1_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm_above64_part2_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm_above64_part3_lower_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);


    
__global__ void
ctrtri_diag_upper_kernel(
    magma_diag_t diag, int n, const magmaFloatComplex *A, int lda, magmaFloatComplex *d_invA);

__global__ void
triple_cgemm16_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm16_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm32_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm32_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm64_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm64_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm_above64_part1_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm_above64_part2_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);

__global__ void
triple_cgemm_above64_part3_upper_kernel(
    int n, const magmaFloatComplex *Ain, int lda, magmaFloatComplex *d_invA, int jb, int npages);







__global__ void
ctrtri_diag_lower_kernel_batched(
    magma_diag_t diag, int n, magmaFloatComplex const * const * dA_array, int lda, magmaFloatComplex **dinvA_array);

__global__ void
triple_cgemm16_part1_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm16_part2_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm32_part1_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm32_part2_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm64_part1_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm64_part2_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm_above64_part1_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm_above64_part2_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm_above64_part3_lower_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);


    
__global__ void
ctrtri_diag_upper_kernel_batched(
    magma_diag_t diag, int n, magmaFloatComplex const * const * dA_array, int lda, magmaFloatComplex **dinvA_array);

__global__ void
triple_cgemm16_part1_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm16_part2_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm32_part1_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm32_part2_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm64_part1_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm64_part2_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm_above64_part1_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm_above64_part2_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);

__global__ void
triple_cgemm_above64_part3_upper_kernel_batched(
    int n, magmaFloatComplex const * const * Ain_array, int lda, magmaFloatComplex **dinvA_array, int jb, int npages);














#endif        //  #ifndef CTRTRI_H
