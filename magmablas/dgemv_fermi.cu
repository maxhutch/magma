/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions normal d

*/
#include "common_magma.h"

#define magmablas_dgemv_fermi magmablas_dgemv

#define dgemv_bs 64
#define threadSize 256



__global__ void 
dgemvn_kernel_fermi(
    int m, int n, int n1, double alpha,
    const double *A, int lda,
    const double *x, int incx, double beta, 
    double *y, int incy)
{
  int ind = blockIdx.x*dgemv_bs + threadIdx.x;


  if(ind < m)
  {
    A += ind;
  }

  double res = 0.0;

  __shared__ double buff[dgemv_bs];

  for(int i=0; i<n1; i += dgemv_bs ){
    __syncthreads();
    buff[threadIdx.x]  = x[(threadIdx.x + i) * incx];

    __syncthreads();
    #pragma unroll
    for(int j=0; j < dgemv_bs ; j++){
       res+=A[0]*buff[j];
       A+=lda;
    }
  }
  __syncthreads();

  if(ind < m)
  {
   if (n>n1)
   {
      for(int j=0; j<(n-n1); j++){
            res += A[0] * x[(n1+j) * incx];
            A+=lda;
      }
   }
  }
  if (ind<m)
     y[ind*incy] = alpha * res + beta * y[ind*incy];
}


extern "C" void
magmablas_dgemvn_fermi(
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    const double *x, magma_int_t incx, double beta, 
    double *y, magma_int_t incy)
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    Purpose
    =======

    This routine computes y = alpha A x on the GPU.

    M      - (input) INTEGER.
             On entry, N specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, M specifies the number of columns of the matrix A

    A      - (input) DOUBLE PRECISION array of dimension ( LDA, m ) on the GPU.
   
    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) DOUBLE PRECISION array of dimension m.
     
    Y      - (output) DOUBLE PRECISION array of        dimension m. 
             On exit Y = alpha A X.

    ===================================================================== */

    magma_int_t blocks;
    
    if (m % dgemv_bs==0)
        blocks = m/dgemv_bs;
    else
        blocks = m/dgemv_bs + 1;

    dim3 grid(blocks, 1, 1);
    dim3 threads(dgemv_bs, 1, 1);
 
    dgemvn_kernel_fermi<<< grid, threads, 0, magma_stream >>>(m, n, (n/ dgemv_bs)*dgemv_bs, 
                                    alpha, A, lda, x, incx, beta, y, incy);
}


__global__ void
dgemvt_kernel_fermi(
    int m, int n, double alpha, int n1,
    const double *A, int lda,
    const double *x, int incx, double beta, 
    double *y, int incy)
{
        int tx = threadIdx.x;

        __shared__ double sdata[threadSize];


        double res;
        res = 0.0;

        for(int i=0; i<n1; i+= threadSize)
        {
                res += A[tx + i + lda * blockIdx.y] * x[(tx + i)*incx];
        }

        if(m > n1)
        {
                if( tx + n1 <  m )
                {
                        res  += A[tx + n1 + lda *blockIdx.y] * x[(tx + n1)*incx];
                }
                else
                {
                        res  = res;
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

        if(tx<32)
        {
                sdata[tx] += sdata[tx+32];
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
                        y[blockIdx.y*incy] = sdata[0] * alpha + beta * y[blockIdx.y*incy];
                }
        }
}




extern "C" void
magmablas_dgemvt_fermi(
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    const double *x, magma_int_t incx, double beta,
    double *y, magma_int_t incy)
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    Purpose
    =======

    This routine computes y = alpha A^t x on the GPU.

    M      - (input) INTEGER.
             On entry, m specifies the number of rows of the matrix A.

    N      - (input) INTEGER.
             On entry, n specifies the number of columns of the matrix A

    A      - (input) DOUBLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    X      - (input) DOUBLE PRECISION array of dimension m.

    Y      - (output) DOUBLE PRECISION array of dimension n.
             On exit y = alpha A^t X.

    ===================================================================== */

        dim3 grid    ( 1,  n,  1);
        dim3 threads ( threadSize,   1,  1);

        dgemvt_kernel_fermi<<< grid, threads, 0, magma_stream >>>( m, n, alpha, ( m / threadSize) * threadSize,
                                                                       A, lda, x, incx, beta,  y, incy);
}



extern "C" 
void magmablas_dgemv_fermi(char trans,
                           magma_int_t m, magma_int_t n,
                           double alpha, 
                           const double *A, magma_int_t lda, 
                           const double *x, magma_int_t incx,
                           double beta,
                           double *z, magma_int_t incz)
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

    Purpose
    =======
    This routine computes:
    1) z =       A   x    if trans == 'N' or 'n', alpha == 1, beta == 0, 
                          and incx == incz == 1 (using magmablas code)
    2) z = alpha A^t x    if trans == 'T' or 't', beta == 0,
                          and incx == incz == 1 (using magmablas code)
    3) z = alpha A^trans x + beta z
                          otherwise, using CUBLAS.

   Arguments
   ==========
    TRANS  - CHARACTER*1
             On entry, TRANS specifies the operation to be performed as
             follows:
               TRANS = 'N' or 'n'   z := alpha*A *x + beta*z
               TRANS = 'T' or 't'   z := alpha*A'*x + beta*z

    M      - (input) INTEGER
             On entry, m specifies the number of rows of the matrix A.

    N      - (input) INTEGER
             On entry, n specifies the number of columns of the matrix A
 
    ALPHA  - DOUBLE REAL
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - (input) DOUBLE PRECISION array of dimension ( LDA, n ) on the GPU.
   
    LDA    - (input) INTEGER
             LDA specifies the leading dimension of A.

    X      - (input) DOUBLE PRECISION array of dimension 
             n if trans == 'n'
             m if trans == 't'
     
    INCX   - (input) Specifies the increment for the elements of X.
             INCX must not be zero. Unchanged on exit.
  
    BETA   - DOUBLE REAL
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Z      - (output) DOUBLE PRECISION array of        dimension 
             m if trans == 'n'
             n if trans == 't' 

    INCZ  - (input) Specifies the increment for the elements of Z.
            INCZ must not be zero. Unchanged on exit.
    ===================================================================== */

    //if (incx == 1 && incz == 1) 
    {
       if (trans == 'n' || trans == 'N')
           {
               if ( m >= 7000 && m <= 8000 )
                     cublasDgemv(trans, m, n, alpha, A, lda, x, incx, beta, z, incz);
                else 
                     magmablas_dgemvn_fermi(m,  n, alpha, A, lda, x, incx, beta,  z, incz);
           }
       else if (trans == 't' || trans == 'T')
          magmablas_dgemvt_fermi(m,  n, alpha, A, lda, x, incx, beta, z, incz);
       else
          printf("trans = %c in sgemv_fermi is not available\n", trans);               
    }
//    else
//       cublasDgemv(trans, m, n, alpha, A, lda, x, incx, beta, z, incz);
}

#undef dgemv_bs
#undef threadSize
