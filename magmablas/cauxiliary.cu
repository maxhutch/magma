/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:44 2013

*/
#include "common_magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from cgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

__global__ void cset_to_zero(magmaFloatComplex *A, int lda)
{
    int ind = blockIdx.x*lda + threadIdx.x;

    A += ind;
    A[0] = MAGMA_C_ZERO;
//   A[16*lda] = 0.;
}

__global__ void cset_nbxnb_to_zero(int nb, magmaFloatComplex *A, int lda)
{
   int ind = blockIdx.x*lda + threadIdx.x, i, j;

   A += ind;
   for(i=0; i<nb; i+=32) {
     for(j=0; j<nb; j+=32)
         A[j] = MAGMA_C_ZERO;
     A += 32*lda;
   }
}

extern "C"
void czero_32x32_block(magmaFloatComplex *A, magma_int_t lda)
{
  // cset_to_zero<<< 16, 32, 0, magma_stream >>>(A, lda);
  cset_to_zero<<< 32, 32, 0, magma_stream >>>(A, lda);
}

extern "C"
void czero_nbxnb_block(magma_int_t nb, magmaFloatComplex *A, magma_int_t lda)
{
  cset_nbxnb_to_zero<<< 32, 32, 0, magma_stream >>>(nb, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- GPU kernel for initializing a matrix by 0
*/
#define claset_threads 64

__global__ void claset(int m, int n, magmaFloatComplex *A, int lda)
{
   int ibx = blockIdx.x * claset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m)
        A[i*lda] = MAGMA_C_ZERO;
}

__global__ void claset_identity(int m, int n, magmaFloatComplex *A, int lda)
{
   int ibx = blockIdx.x * claset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m) {
        if (ind != i+iby)
           A[i*lda] = MAGMA_C_ZERO;
        else
           A[i*lda] = MAGMA_C_ONE;
     }
}

__global__ void claset_identityonly(int m, int n, magmaFloatComplex *A, int lda)
{
   int ibx = blockIdx.x * claset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m) {
        if (ind == i+iby)
           A[i*lda] = MAGMA_C_ONE;
     }
}


__global__ void clasetlower(int m, int n, magmaFloatComplex *A, int lda)
{
   int ibx = blockIdx.x * claset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m && ind > i+iby)
        A[i*lda] = MAGMA_C_ZERO;
}

__global__ void clasetupper(int m, int n, magmaFloatComplex *A, int lda)
{
   int ibx = blockIdx.x * claset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m && ind < i+iby)
        A[i*lda] = MAGMA_C_ZERO;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to 0 on the GPU.
*/
extern "C" void
magmablas_claset(char uplo, magma_int_t m, magma_int_t n,
                 magmaFloatComplex *A, magma_int_t lda)
{
   dim3 threads(claset_threads, 1, 1);
   dim3 grid(m/claset_threads+(m % claset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
     if (uplo == MagmaLower)
        clasetlower<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
     else if (uplo == MagmaUpper)
        clasetupper<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
     else
        claset<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the GPU.
*/
extern "C" void
magmablas_claset_identity(magma_int_t m, magma_int_t n,
                          magmaFloatComplex *A, magma_int_t lda)
{
   dim3 threads(claset_threads, 1, 1);
   dim3 grid(m/claset_threads+(m % claset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
      claset_identity<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the diag without touching the offdiag GPU.
*/
extern "C" void
magmablas_claset_identityonly(magma_int_t m, magma_int_t n,
                          magmaFloatComplex *A, magma_int_t lda)
{
   dim3 threads(claset_threads, 1, 1);
   dim3 grid(m/claset_threads+(m % claset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
      claset_identityonly<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Given two matrices, 'a' on the CPU and 'da' on the GPU, this function
      returns the Frobenious norm of the difference of the two matrices.
      The function is used for debugging.
*/
extern "C"
float cpu_gpu_cdiff(
    magma_int_t M, magma_int_t N,
    const magmaFloatComplex *a,  magma_int_t lda,
    const magmaFloatComplex *da, magma_int_t ldda )
{
  magma_int_t d_one = 1;
  magma_int_t j;
  magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
  float  work[1];
  magmaFloatComplex *ha = (magmaFloatComplex*)malloc( M * N * sizeof(magmaFloatComplex));
  float res;

  cublasGetMatrix(M, N, sizeof(magmaFloatComplex), da, ldda, ha, M);
  for(j=0; j<N; j++)
    blasf77_caxpy(&M, &c_neg_one, a+j*lda, &d_one, ha+j*M, &d_one);
  res = lapackf77_clange("f", &M, &N, ha, &M, work);

  free(ha);
  return res;
}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting 0 in the nb-1 upper subdiagonals and 1 in the diagonal
    @author Raffaele Solca
 */
__global__ void csetdiag1subdiag0_L(int k, magmaFloatComplex *A, int lda)
{

  int nb = blockDim.x;
  int ibx = blockIdx.x * nb;

  int ind = ibx + threadIdx.x + 1;

  A += ind - nb + __mul24((ibx), lda);

  magmaFloatComplex tmp = MAGMA_C_ZERO;
  if(threadIdx.x == nb-1)
    tmp = MAGMA_C_ONE;

#pragma unroll
  for(int i=0; i<nb; i++)
    if (ibx+i < k && ind + i  >= nb) {
      A[i*(lda+1)] = tmp;
    }

}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting 0 in the nb-1 lower subdiagonals and 1 in the diagonal
    @author Raffaele Solca
 */

__global__ void csetdiag1subdiag0_U(int k, magmaFloatComplex *A, int lda)
{

  int nb = blockDim.x;
  int ibx = blockIdx.x * nb;

  int ind = ibx + threadIdx.x;

  A += ind + __mul24((ibx), lda);

  magmaFloatComplex tmp = MAGMA_C_ZERO;
  if(threadIdx.x == 0)
    tmp = MAGMA_C_ONE;

#pragma unroll
  for(int i=0; i<nb; i++)
    if (ibx+i < k && ind + i < k) {
      A[i*(lda+1)] = tmp;
    }

}

/* ////////////////////////////////////////////////////////////////////////////
 -- Set 1s in the diagonal and 0s in the nb-1 lower (UPLO='U') or
    upper (UPLO='L') subdiagonals.
    stream and no stream interfaces
    @author Raffaele Solca
 */
extern "C" void
magmablas_csetdiag1subdiag0_stream(char uplo, magma_int_t k, magma_int_t nb,
                 magmaFloatComplex *A, magma_int_t lda, magma_queue_t stream)
{
  dim3 threads(nb, 1, 1);
  dim3 grid((k-1)/nb+1);
  if(k>lda)
    fprintf(stderr,"wrong second argument of csetdiag1subdiag0");
  if(uplo == MagmaLower)
    csetdiag1subdiag0_L<<< grid, threads, 0, stream >>> (k, A, lda);
  else if(uplo == MagmaUpper) {
    csetdiag1subdiag0_U<<< grid, threads, 0, stream >>> (k, A, lda);
  }
  else
    fprintf(stderr,"wrong first argument of csetdiag1subdiag0");

  return;
}

extern "C" void
magmablas_csetdiag1subdiag0(char uplo, magma_int_t k, magma_int_t nb,
                 magmaFloatComplex *A, magma_int_t lda)
{
  magmablas_csetdiag1subdiag0_stream(uplo, k, nb, A, lda, magma_stream);
}

