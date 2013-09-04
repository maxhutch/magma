/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013
*/
#include "common_magma.h"

#define num_threads 128
#define zgemv_bs     32
#define threadSize  128

#define magmablas_zgemvt_fermi magmablas_zgemvt

__global__ void 
zgemvtn_kernel2_fermi(int n, int m, int n1, magmaDoubleComplex alpha,  magmaDoubleComplex* A, int lda, magmaDoubleComplex *x, int incx, magmaDoubleComplex *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x*incx;

  magmaDoubleComplex res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  __shared__ magmaDoubleComplex buff[num_threads];
  for(int i=0; i<n1; i += num_threads ){
    __syncthreads();
    //buff[threadIdx.x]  = x[i];
    buff[threadIdx.x]  = MAGMA_Z_MAKE(cuCreal(x[i*incx]), -cuCimag(x[i*incx]));

    __syncthreads();
    #pragma unroll
    for(int j=0; j < num_threads ; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }
  }
  __syncthreads();

  if (m>n1){
     //buff[threadIdx.x]  = x[n1];
     buff[threadIdx.x]  = MAGMA_Z_MAKE(cuCreal(x[n1*incx]), -cuCimag(x[n1*incx]));

     __syncthreads();
     for(int j=0; j<(m-n1); j++){
         res += A[0]*buff[j];
         A+=lda;
     }
  }

  if (ind<n)
     y[ind] += alpha * res;
}

extern "C" void
magmablas_zgemvtn_fermi(int n, int m, magmaDoubleComplex alpha, magmaDoubleComplex *A, int lda, magmaDoubleComplex *x, int incx, magmaDoubleComplex *y)
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    Purpose
    =======

    This routine computes Y = alpha A x on the GPU.

    N      - (input) INTEGER.
             On entry, N specifies the number of rows of the matrix A.

    M      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, m ) on the GPU.
   
    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.
     
    Y      - (output) SINGLE PRECISION array of        dimension m. 
             On exit Y = alpha A X.

    ===================================================================== */

    int blocks;
    if (n % num_threads==0)
        blocks = n/num_threads;
    else
        blocks = n/num_threads + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
/*    if(n<=8500) 
                zgemvn_kernel1_fermi<<< grid, threads, 0, magma_stream >>>(n, m, (m / zgemv_bs)*zgemv_bs, 
                                                   alpha, A, lda, x, y);
    else */
                zgemvtn_kernel2_fermi<<< grid, threads, 0, magma_stream >>>(n, m, (m / num_threads)*num_threads, 
                                                   alpha, A, lda, x, incx, y);
}



extern "C" void
magmablas_zgemvt_fermi(char flag, int m, int n, magmaDoubleComplex alpha, 
                       magmaDoubleComplex *A, int lda, magmaDoubleComplex *x, int incx, magmaDoubleComplex beta, magmaDoubleComplex *y, int incy) 
{

    //if(beta.x==0.0 && beta.y==0.0 && incx ==1 && incy==1)
        {
                //if (flag == 'N' || flag == 'n')
                {
                        //if(m<7000)
                        //{
                        //        cublasZgemv(flag, m, n, alpha, A, lda, x, incx, beta, y, incy);
                        //}
                        //else
                        {
                                magmablas_zgemvtn_fermi(m,  n, alpha, A, lda, x, incx, y);
                        }
                }
                //else if(flag == 'T' || flag == 't')
                //{
                //        magmablas_zgemvt_fermi(m,  n, alpha, A, lda, x, y);
                //}
                //else if(flag == 'C' || flag == 'c') 
                //{
                //        magmablas_zgemvc_fermi(m,  n, alpha, A, lda, x, y);
                //}
                //else 
                //{
                //        cublasZgemv(flag, m, n, alpha, A, lda, x, incx, beta, y, incy);
                //}
        }
        //else
        //{
        //        cublasZgemv(flag, m, n, alpha, A, lda, x, incx, beta, y, incy);
        //}
}

#undef num_threads
#undef zgemv_bs
#undef threadSize 
