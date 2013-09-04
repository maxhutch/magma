/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions normal d

*/
#include "common_magma.h"

#define num_threads 128
#define cgemv_bs 32
#define threadSize 128

#define magmablas_cgemv_fermi magmablas_cgemv 

__global__ void 
cgemvn_kernel1_fermi(
    int n, int m, int n1, magmaFloatComplex alpha,
    const magmaFloatComplex *A, int lda,
    const magmaFloatComplex *x, magmaFloatComplex beta, 
    magmaFloatComplex *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;

  magmaFloatComplex res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  for(int i=0; i<n1; i += cgemv_bs ){

    #pragma unroll
    for(int j=0; j < cgemv_bs ; j++){
       res += A[0] * x[j];
       A   += lda;
    }
        x += cgemv_bs;
  }

  if (m>n1){

     for(int j=0; j<(m-n1); j++){
         res += A[0] * x[j];
         A   += lda;
     }
  }

  if (ind<n)
     y[ind] = alpha * res + beta * y[ind];

}

__global__ void 
cgemvn_kernel2_fermi(
    int n, int m, int n1, magmaFloatComplex alpha,
    const magmaFloatComplex *A, int lda,
    const magmaFloatComplex *x, magmaFloatComplex beta, 
    magmaFloatComplex *y)
{
  int ind = blockIdx.x*num_threads + threadIdx.x;

  A += ind;
  x += threadIdx.x;

  magmaFloatComplex res;
  MAGMA_Z_SET2REAL(res, 0.0f);

  __shared__ magmaFloatComplex buff[num_threads];
  for(int i=0; i<n1; i += num_threads ){
    __syncthreads();
    buff[threadIdx.x]  = x[i];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < num_threads ; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }
  }
  __syncthreads();

  if (m>n1){
     buff[threadIdx.x]  = x[n1];

     __syncthreads();
     for(int j=0; j<(m-n1); j++){
         res += A[0]*buff[j];
         A+=lda;
     }
  }

  if (ind<n)
     y[ind] = alpha * res + beta * y[ind];
}

extern "C" void
magmablas_cgemvn_fermi(
    magma_int_t n, magma_int_t m, magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *x, magmaFloatComplex beta,
    magmaFloatComplex *y)
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
  /*  if(n<=8500) 
                cgemvn_kernel1_fermi<<< grid, threads, 0, magma_stream >>>(n, m, (m / cgemv_bs)*cgemv_bs, 
                                                   alpha, A, lda, x, y);
        else 
   */
                cgemvn_kernel2_fermi<<< grid, threads, 0, magma_stream >>>(n, m, (m / num_threads)*num_threads, 
                                                   alpha, A, lda, x, beta, y);

}



__global__ void 
cgemvt_kernel_fermi(
    int m, int n, magmaFloatComplex alpha, int n1,
    const magmaFloatComplex *A, int lda,
    const magmaFloatComplex *x, magmaFloatComplex beta,
    magmaFloatComplex *y)
{
        unsigned int tx = threadIdx.x;

        __shared__ magmaFloatComplex sdata[threadSize];
        

        magmaFloatComplex res;
    MAGMA_Z_SET2REAL(res, 0.0f);
        magmaFloatComplex zero;
    MAGMA_Z_SET2REAL(zero, 0.0f);
     
        for(int i=0; i<n1; i+= threadSize)
        {
                res += A[tx + i + lda * blockIdx.y] * x[tx + i];
        }

        
        if(m > n1)
        {
                if( tx + n1 <  m )
                {
                        res  += A[tx + n1 + lda *blockIdx.y] * x[tx + n1];
                }
                else 
                {
                        res  += zero;
                }
        }        

    sdata[tx] = res;
        __syncthreads();
    

        for(int s=blockDim.x/2; s>32;s>>=1)
        {
                        if(tx<s)
                        {
                                        sdata[tx] += sdata[tx+s];
                        } 
                        __syncthreads();
        }


        if(tx < 32) 
        {
                sdata[tx] += sdata[tx + 32];
        }

    if(tx == 0)
        {
                for(int i=1;i<32;i++)
                {
                        sdata[tx] += sdata[tx + i];
                }
        }

    if( tx == 0 ) 
        {

                if (blockIdx.y < n)
                {
                        y[blockIdx.y] = sdata[0] * alpha + beta * y[blockIdx.y];
                }
        }
}




extern "C" void
magmablas_cgemvt_fermi(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda, 
    const magmaFloatComplex *x, magmaFloatComplex beta,
    magmaFloatComplex *y)
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    Purpose
    =======

    This routine computes y = alpha *  A^t *  x on the GPU.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Y      - (output) SINGLE PRECISION array of dimension n.
             On exit Y = alpha A^t X.

    ===================================================================== */

    dim3 grid    ( 1,  n,  1);
    dim3 threads ( threadSize,   1,  1);

    cgemvt_kernel_fermi<<< grid, threads, 0, magma_stream >>>( m, n, alpha, ( m / threadSize) * threadSize,
                                       A, lda, x, beta,  y);
    

}




__global__ void 
cgemvc_kernel_fermi(
    int m, int n, magmaFloatComplex alpha, int n1,
    const magmaFloatComplex *A, int lda,
    const magmaFloatComplex *x, magmaFloatComplex beta,
    magmaFloatComplex *y)
{
        unsigned int tx = threadIdx.x;

        __shared__ magmaFloatComplex sdata[threadSize];
        

        magmaFloatComplex res;
    MAGMA_Z_SET2REAL(res, 0.0f);
        magmaFloatComplex zero;
    MAGMA_Z_SET2REAL(zero, 0.0f);
     
        for(int i=0; i<n1; i+= threadSize)
        {
                res += cuConjf(A[tx + i + lda * blockIdx.y]) * x[tx + i];
        }

        
        if(m > n1)
        {
                if( tx + n1 <  m )
                {
                        res  += cuConjf(A[tx + n1 + lda *blockIdx.y]) * x[tx + n1];
                }
                else 
                {
                        res  += zero;
                }
        }        

    sdata[tx] = res;
        __syncthreads();
    
    /*
        if(tx < 128) 
        {
                sdata[tx] += sdata[tx + 128];
        }
    __syncthreads();
        */

        if(tx < 64) 
        {
                sdata[tx] += sdata[tx + 64];
        }
    __syncthreads();

        if(tx < 32) 
        {
                sdata[tx] += sdata[tx + 32];
        }

    if(tx == 0)
        {
                for(int i=1;i<32;i++)
                {
                        sdata[tx] += sdata[tx + i];
                }
        }

    if( tx == 0 ) 
        {

                if (blockIdx.y < n)
                {
                        y[blockIdx.y] = sdata[0] * alpha + beta * y[blockIdx.y];
                }
        }
}




extern "C" void
magmablas_cgemvc_fermi(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda, 
    const magmaFloatComplex *x, magmaFloatComplex beta,
    magmaFloatComplex *y)
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    Purpose
    =======

    This routine computes y = alpha *  conjg(A^t) *  x on the GPU.

    M      - (input) INTEGER.
             On entry, M specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A

    A      - (input) SINGLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) SINGLE PRECISION array of dimension m.

    Y      - (output) SINGLE PRECISION array of dimension n.
             On exit Y = alpha conjg(A^t) X.

    ===================================================================== */

    dim3 grid    ( 1,  n,  1);
    dim3 threads ( threadSize,   1,  1);

    cgemvc_kernel_fermi<<< grid, threads, 0, magma_stream >>>( m, n, alpha, ( m / threadSize) * threadSize,
                                       A, lda, x, beta,  y);
    

}




extern "C" void
magmablas_cgemv_fermi(
    char flag, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *x, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, magma_int_t incy ) 
{

    if(incx==1 && incy ==1)
    {
                if (flag == 'N' || flag == 'n')
                {
                        if(m<8000)
                        {
                                cublasCgemv(flag, m, n, alpha, A, lda, x, incx, beta, y, incy);
                           }
                        else 
                        {
                                magmablas_cgemvn_fermi(m,  n, alpha, A, lda, x,  beta, y);
                        }
                }
                else if(flag == 'T' || flag == 't')
                {
                        magmablas_cgemvt_fermi(m,  n, alpha, A, lda, x, beta, y);
                }
                else if(flag == 'C' || flag == 'c')
                {
                        magmablas_cgemvc_fermi(m,  n, alpha, A, lda, x, beta,  y);
                }
                else 
                {
                        cublasCgemv(flag, m, n, alpha, A, lda, x, incx, beta, y, incy);
                }
     }
     else

            cublasCgemv(flag, m, n, alpha, A, lda, x, incx, beta, y, incy);

}


#undef num_threads
#undef cgemv_bs
#undef threadSize 
