/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from zgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

__global__ void zset_to_zero(magmaDoubleComplex *A, int lda)
{
    int ind = blockIdx.x*lda + threadIdx.x;

    A += ind;
    A[0] = MAGMA_Z_ZERO;
//   A[16*lda] = 0.;
}

__global__ void zset_nbxnb_to_zero(int nb, magmaDoubleComplex *A, int lda)
{
   int ind = blockIdx.x*lda + threadIdx.x, i, j;

   A += ind;
   for(i=0; i<nb; i+=32) {
     for(j=0; j<nb; j+=32)
         A[j] = MAGMA_Z_ZERO;
     A += 32*lda;
   }
}

extern "C"
void zzero_32x32_block(magmaDoubleComplex *A, magma_int_t lda)
{
  // zset_to_zero<<< 16, 32, 0, magma_stream >>>(A, lda);
  zset_to_zero<<< 32, 32, 0, magma_stream >>>(A, lda);
}

extern "C"
void zzero_nbxnb_block(magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda)
{
  zset_nbxnb_to_zero<<< 32, 32, 0, magma_stream >>>(nb, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- GPU kernel for initializing a matrix by 0
*/
#define zlaset_threads 64

__global__ void zlaset(int m, int n, magmaDoubleComplex *A, int lda)
{
   int ibx = blockIdx.x * zlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m)
        A[i*lda] = MAGMA_Z_ZERO;
}

__global__ void zlaset_identity(int m, int n, magmaDoubleComplex *A, int lda)
{
   int ibx = blockIdx.x * zlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m) {
        if (ind != i+iby)
           A[i*lda] = MAGMA_Z_ZERO;
        else
           A[i*lda] = MAGMA_Z_ONE;
     }
}

__global__ void zlaset_identityonly(int m, int n, magmaDoubleComplex *A, int lda)
{
   int ibx = blockIdx.x * zlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m) {
        if (ind == i+iby)
           A[i*lda] = MAGMA_Z_ONE;
     }
}


__global__ void zlasetlower(int m, int n, magmaDoubleComplex *A, int lda)
{
   int ibx = blockIdx.x * zlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m && ind > i+iby)
        A[i*lda] = MAGMA_Z_ZERO;
}

__global__ void zlasetupper(int m, int n, magmaDoubleComplex *A, int lda)
{
   int ibx = blockIdx.x * zlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m && ind < i+iby)
        A[i*lda] = MAGMA_Z_ZERO;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to 0 on the GPU.
*/
extern "C" void
magmablas_zlaset(char uplo, magma_int_t m, magma_int_t n,
                 magmaDoubleComplex *A, magma_int_t lda)
{
   dim3 threads(zlaset_threads, 1, 1);
   dim3 grid(m/zlaset_threads+(m % zlaset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
     if (uplo == MagmaLower)
        zlasetlower<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
     else if (uplo == MagmaUpper)
        zlasetupper<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
     else
        zlaset<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the GPU.
*/
extern "C" void
magmablas_zlaset_identity(magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda)
{
   dim3 threads(zlaset_threads, 1, 1);
   dim3 grid(m/zlaset_threads+(m % zlaset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
      zlaset_identity<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the diag without touching the offdiag GPU.
*/
extern "C" void
magmablas_zlaset_identityonly(magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda)
{
   dim3 threads(zlaset_threads, 1, 1);
   dim3 grid(m/zlaset_threads+(m % zlaset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
      zlaset_identityonly<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Given two matrices, 'a' on the CPU and 'da' on the GPU, this function
      returns the Frobenious norm of the difference of the two matrices.
      The function is used for debugging.
*/
extern "C"
double cpu_gpu_zdiff(
    magma_int_t M, magma_int_t N,
    const magmaDoubleComplex *a,  magma_int_t lda,
    const magmaDoubleComplex *da, magma_int_t ldda )
{
  magma_int_t d_one = 1;
  magma_int_t j;
  magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
  double  work[1];
  magmaDoubleComplex *ha = (magmaDoubleComplex*)malloc( M * N * sizeof(magmaDoubleComplex));
  double res;

  cublasGetMatrix(M, N, sizeof(magmaDoubleComplex), da, ldda, ha, M);
  for(j=0; j<N; j++)
    blasf77_zaxpy(&M, &c_neg_one, a+j*lda, &d_one, ha+j*M, &d_one);
  res = lapackf77_zlange("f", &M, &N, ha, &M, work);

  free(ha);
  return res;
}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting 0 in the nb-1 upper subdiagonals and 1 in the diagonal
    @author Raffaele Solca
 */
__global__ void zsetdiag1subdiag0_L(int k, magmaDoubleComplex *A, int lda)
{

  int nb = blockDim.x;
  int ibx = blockIdx.x * nb;

  int ind = ibx + threadIdx.x + 1;

  A += ind - nb + __mul24((ibx), lda);

  magmaDoubleComplex tmp = MAGMA_Z_ZERO;
  if(threadIdx.x == nb-1)
    tmp = MAGMA_Z_ONE;

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

__global__ void zsetdiag1subdiag0_U(int k, magmaDoubleComplex *A, int lda)
{

  int nb = blockDim.x;
  int ibx = blockIdx.x * nb;

  int ind = ibx + threadIdx.x;

  A += ind + __mul24((ibx), lda);

  magmaDoubleComplex tmp = MAGMA_Z_ZERO;
  if(threadIdx.x == 0)
    tmp = MAGMA_Z_ONE;

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
magmablas_zsetdiag1subdiag0_stream(char uplo, magma_int_t k, magma_int_t nb,
                 magmaDoubleComplex *A, magma_int_t lda, magma_queue_t stream)
{
  dim3 threads(nb, 1, 1);
  dim3 grid((k-1)/nb+1);
  if(k>lda)
    fprintf(stderr,"wrong second argument of zsetdiag1subdiag0");
  if(uplo == MagmaLower)
    zsetdiag1subdiag0_L<<< grid, threads, 0, stream >>> (k, A, lda);
  else if(uplo == MagmaUpper) {
    zsetdiag1subdiag0_U<<< grid, threads, 0, stream >>> (k, A, lda);
  }
  else
    fprintf(stderr,"wrong first argument of zsetdiag1subdiag0");

  return;
}

extern "C" void
magmablas_zsetdiag1subdiag0(char uplo, magma_int_t k, magma_int_t nb,
                 magmaDoubleComplex *A, magma_int_t lda)
{
  magmablas_zsetdiag1subdiag0_stream(uplo, k, nb, A, lda, magma_stream);
}

