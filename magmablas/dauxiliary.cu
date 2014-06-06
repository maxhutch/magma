/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:44 2013

*/
#include "common_magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- This is an auxiliary routine called from dgehrd.  The routine is called
      in 16 blocks, 32 thread per block and initializes to zero the 1st
      32x32 block of A.
*/

__global__ void dset_to_zero(double *A, int lda)
{
    int ind = blockIdx.x*lda + threadIdx.x;

    A += ind;
    A[0] = MAGMA_D_ZERO;
//   A[16*lda] = 0.;
}

__global__ void dset_nbxnb_to_zero(int nb, double *A, int lda)
{
   int ind = blockIdx.x*lda + threadIdx.x, i, j;

   A += ind;
   for(i=0; i<nb; i+=32) {
     for(j=0; j<nb; j+=32)
         A[j] = MAGMA_D_ZERO;
     A += 32*lda;
   }
}

extern "C"
void dzero_32x32_block(double *A, magma_int_t lda)
{
  // dset_to_zero<<< 16, 32, 0, magma_stream >>>(A, lda);
  dset_to_zero<<< 32, 32, 0, magma_stream >>>(A, lda);
}

extern "C"
void dzero_nbxnb_block(magma_int_t nb, double *A, magma_int_t lda)
{
  dset_nbxnb_to_zero<<< 32, 32, 0, magma_stream >>>(nb, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- GPU kernel for initializing a matrix by 0
*/
#define dlaset_threads 64

__global__ void dlaset(int m, int n, double *A, int lda)
{
   int ibx = blockIdx.x * dlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m)
        A[i*lda] = MAGMA_D_ZERO;
}

__global__ void dlaset_identity(int m, int n, double *A, int lda)
{
   int ibx = blockIdx.x * dlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m) {
        if (ind != i+iby)
           A[i*lda] = MAGMA_D_ZERO;
        else
           A[i*lda] = MAGMA_D_ONE;
     }
}

__global__ void dlaset_identityonly(int m, int n, double *A, int lda)
{
   int ibx = blockIdx.x * dlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m) {
        if (ind == i+iby)
           A[i*lda] = MAGMA_D_ONE;
     }
}


__global__ void dlasetlower(int m, int n, double *A, int lda)
{
   int ibx = blockIdx.x * dlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m && ind > i+iby)
        A[i*lda] = MAGMA_D_ZERO;
}

__global__ void dlasetupper(int m, int n, double *A, int lda)
{
   int ibx = blockIdx.x * dlaset_threads;
   int iby = blockIdx.y * 32;

   int ind = ibx + threadIdx.x;

   A += ind + __mul24(iby, lda);

   #pragma unroll
   for(int i=0; i<32; i++)
     if (iby+i < n && ind < m && ind < i+iby)
        A[i*lda] = MAGMA_D_ZERO;
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to 0 on the GPU.
*/
extern "C" void
magmablas_dlaset(char uplo, magma_int_t m, magma_int_t n,
                 double *A, magma_int_t lda)
{
   dim3 threads(dlaset_threads, 1, 1);
   dim3 grid(m/dlaset_threads+(m % dlaset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
     if (uplo == MagmaLower)
        dlasetlower<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
     else if (uplo == MagmaUpper)
        dlasetupper<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
     else
        dlaset<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the GPU.
*/
extern "C" void
magmablas_dlaset_identity(magma_int_t m, magma_int_t n,
                          double *A, magma_int_t lda)
{
   dim3 threads(dlaset_threads, 1, 1);
   dim3 grid(m/dlaset_threads+(m % dlaset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
      dlaset_identity<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Set the m x n matrix pointed by A to I on the diag without touching the offdiag GPU.
*/
extern "C" void
magmablas_dlaset_identityonly(magma_int_t m, magma_int_t n,
                          double *A, magma_int_t lda)
{
   dim3 threads(dlaset_threads, 1, 1);
   dim3 grid(m/dlaset_threads+(m % dlaset_threads != 0), n/32+(n%32!=0));

   if (m!=0 && n !=0)
      dlaset_identityonly<<< grid, threads, 0, magma_stream >>> (m, n, A, lda);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Given two matrices, 'a' on the CPU and 'da' on the GPU, this function
      returns the Frobenious norm of the difference of the two matrices.
      The function is used for debugging.
*/
extern "C"
double cpu_gpu_ddiff(
    magma_int_t M, magma_int_t N,
    const double *a,  magma_int_t lda,
    const double *da, magma_int_t ldda )
{
  magma_int_t d_one = 1;
  magma_int_t j;
  double c_neg_one = MAGMA_D_NEG_ONE;
  double  work[1];
  double *ha = (double*)malloc( M * N * sizeof(double));
  double res;

  cublasGetMatrix(M, N, sizeof(double), da, ldda, ha, M);
  for(j=0; j<N; j++)
    blasf77_daxpy(&M, &c_neg_one, a+j*lda, &d_one, ha+j*M, &d_one);
  res = lapackf77_dlange("f", &M, &N, ha, &M, work);

  free(ha);
  return res;
}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting 0 in the nb-1 upper subdiagonals and 1 in the diagonal
    @author Raffaele Solca
 */
__global__ void dsetdiag1subdiag0_L(int k, double *A, int lda)
{

  int nb = blockDim.x;
  int ibx = blockIdx.x * nb;

  int ind = ibx + threadIdx.x + 1;

  A += ind - nb + __mul24((ibx), lda);

  double tmp = MAGMA_D_ZERO;
  if(threadIdx.x == nb-1)
    tmp = MAGMA_D_ONE;

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

__global__ void dsetdiag1subdiag0_U(int k, double *A, int lda)
{

  int nb = blockDim.x;
  int ibx = blockIdx.x * nb;

  int ind = ibx + threadIdx.x;

  A += ind + __mul24((ibx), lda);

  double tmp = MAGMA_D_ZERO;
  if(threadIdx.x == 0)
    tmp = MAGMA_D_ONE;

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
magmablas_dsetdiag1subdiag0_stream(char uplo, magma_int_t k, magma_int_t nb,
                 double *A, magma_int_t lda, magma_queue_t stream)
{
  dim3 threads(nb, 1, 1);
  dim3 grid((k-1)/nb+1);
  if(k>lda)
    fprintf(stderr,"wrong second argument of dsetdiag1subdiag0");
  if(uplo == MagmaLower)
    dsetdiag1subdiag0_L<<< grid, threads, 0, stream >>> (k, A, lda);
  else if(uplo == MagmaUpper) {
    dsetdiag1subdiag0_U<<< grid, threads, 0, stream >>> (k, A, lda);
  }
  else
    fprintf(stderr,"wrong first argument of dsetdiag1subdiag0");

  return;
}

extern "C" void
magmablas_dsetdiag1subdiag0(char uplo, magma_int_t k, magma_int_t nb,
                 double *A, magma_int_t lda)
{
  magmablas_dsetdiag1subdiag0_stream(uplo, k, nb, A, lda, magma_stream);
}

