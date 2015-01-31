/*
   -- MAGMA (version 1.6.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date January 2015

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
 */
#include "common_magma.h"


#define zgemv_bs 32

extern __shared__ magmaDoubleComplex shared_data[];


__global__ void
kernel_zgemvn_batched(
    int m, int n, magmaDoubleComplex alpha,
    magmaDoubleComplex **dA_array, int lda,
    magmaDoubleComplex **x_array, int incx,
    magmaDoubleComplex beta, magmaDoubleComplex  **y_array, int incy)
{

    magmaDoubleComplex *A = dA_array[blockIdx.x];
    magmaDoubleComplex *x = x_array[blockIdx.x];
    magmaDoubleComplex *y = y_array[blockIdx.x];

    int tx = threadIdx.x;

    magmaDoubleComplex res = MAGMA_Z_ZERO;

    magmaDoubleComplex *buff = (magmaDoubleComplex*)shared_data;

    if(tx < n)
    {
        buff[tx] = x[tx*incx];
    }
    __syncthreads();
   
    
    if(tx < m )
    {
        for(int j=0; j < n ; j++)
        {
            res += A[tx]*buff[j];
            A += lda;
        }
  
        y[tx*incy] = alpha * res + y[tx*incy] * beta;
    }

}

/*
    Matrix Non-transpose Vector Multiplication
    y := alpha*A*x + beta*y,
*/
extern "C"
void magmablas_zgemvn_batched(
    int m, int n, 
    magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, int lda, 
    magmaDoubleComplex **x_array,  int incx,
    magmaDoubleComplex beta, magmaDoubleComplex **y_array,  int incy, 
    int batchCount, magma_queue_t queue)
{

    if( m > 512 || n > 512)
    {
        fprintf( stderr, "m=%d, n=%d, zgemv_batched nontranspose assume row && column lower than %d. Plz call magmablas_zgemv instead", m, n, 512);
        return ;
    }

    dim3 grid(batchCount, 1, 1);
    dim3 threads(max(m,n), 1, 1);
   
    kernel_zgemvn_batched<<< grid, threads, n * sizeof(magmaDoubleComplex), queue >>>( m, n, alpha,  dA_array, lda, x_array, incx,  
                                                                         beta, y_array, incy);
}



__global__ void
kernel_zgemvt_batched(
    int m, int n, int m1, magmaDoubleComplex alpha,
    magmaDoubleComplex **dA_array, int lda,
    magmaDoubleComplex **x_array, int incx,
    magmaDoubleComplex beta, magmaDoubleComplex  **y_array, int incy)
{
  

    magmaDoubleComplex *A_ptr = dA_array[blockIdx.x];
    magmaDoubleComplex *x_ptr = x_array[blockIdx.x];
    magmaDoubleComplex *y_ptr = y_array[blockIdx.x];

    int tx = threadIdx.x;
    
    magmaDoubleComplex res = MAGMA_Z_ZERO;

    if(tx<m)
    {  
        A_ptr += lda * blockIdx.y + tx;
        x_ptr += tx * incx;
    }
        
    __shared__ magmaDoubleComplex sdata[zgemv_bs];

    for(int i=0; i<m1; i+= zgemv_bs)
    {
        res += A_ptr[i] * x_ptr[i*incx];
    }

    if(m > m1)
    {
        if( tx + m1 <  m )
        {
            res  += A_ptr[m1] * x_ptr[m1*incx];
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

    if(zgemv_bs > 32)
    {  
        if(tx<32)
        {
            sdata[tx] += sdata[tx+32];
        }
    }

    if(tx == 0)
    {
        for(int i=1;i<32;i++)
        {
            sdata[tx] += sdata[tx + i];
        }
        
        y_ptr[blockIdx.y * incy] = sdata[0] * alpha + beta * y_ptr[blockIdx.y*incy];
               
    }
}

/*
    Matrix Transpose Vector Multiplication
    y := alpha* A**T *x + beta*y,
*/

extern "C"
void magmablas_zgemvt_batched(
    int m, int n, 
    magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, int lda, 
    magmaDoubleComplex **x_array,  int incx,
    magmaDoubleComplex beta, magmaDoubleComplex **y_array,  int incy, 
    int batchCount, magma_queue_t queue)
{

    dim3 grid(batchCount, n, 1);
    dim3 threads(zgemv_bs, 1, 1);

    int m1 = (m / zgemv_bs) * zgemv_bs;

    kernel_zgemvt_batched <<< grid, threads,0, queue  >>>(m, n, m1, alpha,  dA_array, lda, x_array, incx, beta, y_array, incy);

}
   

#if defined(PRECISION_z) || defined (PRECISION_c)


__global__ void
kernel_zgemvc_batched(
    int m, int n, int m1, magmaDoubleComplex alpha,
    magmaDoubleComplex **dA_array, int lda,
    magmaDoubleComplex **x_array, int incx,
    magmaDoubleComplex beta, magmaDoubleComplex  **y_array, int incy)
{
  

    magmaDoubleComplex *A_ptr = dA_array[blockIdx.x];
    magmaDoubleComplex *x_ptr = x_array[blockIdx.x];
    magmaDoubleComplex *y_ptr = y_array[blockIdx.x];

    int tx = threadIdx.x;
    
    magmaDoubleComplex res = MAGMA_Z_ZERO;

    if(tx<m)
    {
        A_ptr += lda * blockIdx.y + tx;
        x_ptr += tx * incx;
    }
        
    __shared__ magmaDoubleComplex sdata[zgemv_bs];

    for(int i=0; i<m1; i+= zgemv_bs)
    {
        res += MAGMA_Z_CNJG (A_ptr[i]) * x_ptr[i*incx];
    }

    if(m > m1)
    {
        if( tx + m1 <  m )
        {
            res  += MAGMA_Z_CNJG(A_ptr[m1]) * x_ptr[m1*incx];
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

    if(zgemv_bs > 32)
    {  
        if(tx<32)
        {
            sdata[tx] += sdata[tx+32];
        }
    }

    if(tx == 0)
    {
        for(int i=1;i<32;i++)
        {
            sdata[tx] += sdata[tx + i];
        }
        
        y_ptr[blockIdx.y * incy] = sdata[0] * alpha + beta * y_ptr[blockIdx.y*incy];
               
    }
}

/*
    Matrix Conjugate Transpose Vector Multiplication
    y := alpha* A**H *x + beta*y,
*/

extern "C"
void magmablas_zgemvc_batched(
    int m, int n, 
    magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, int lda, 
    magmaDoubleComplex **x_array,  int incx,
    magmaDoubleComplex beta, magmaDoubleComplex **y_array,  int incy, 
    int batchCount, magma_queue_t queue)
{

    dim3 grid(batchCount, n, 1);
    dim3 threads(zgemv_bs, 1, 1);

    int m1 = (m / zgemv_bs) * zgemv_bs;

    kernel_zgemvc_batched <<< grid, threads, 0, queue >>>(m, n, m1, alpha,  dA_array, lda, x_array, incx, beta, y_array, incy);
}
   
#endif // defined(PRECISION_z) || defined (PRECISION_c)


/**
    Purpose
    -------

    This routine computes Y = alpha opt(A) x + beta y, on the GPU, where
    A = dA_array[i],x = x_array[i] and y = y_array[i], i=[0,batchCount-1].
    This is a batched version.

    @param[in]
    trans  CHARACTER*1.
           On entry, TRANS specifies the form of op( A ) to be used in
           the matrix multiplication as follows:
           = 'N':  op( A ) = A.
           = 'T':  op( A ) = A**T.
           = 'C':  op( A ) = A**H.

    @param[in]
    m       INTEGER.
            On entry, M specifies the number of rows of the matrix opt(A).

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix opt(A)

    @param[in]
    alpha   COMPLEX*16.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA_array A = dA_array[i] 
            A: COMPLEX*16 array of dimension ( LDA, n ) on the GPU.
   
    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.

    @param[in]
    x_array x = x_array[i]
            x: COMPLEX*16 array of dimension.
            n if trans == MagmaNoTrans.
            m if trans == MagmaTrans or MagmaConjTrans.

    @param[in]
    incx    INTEGER.
            incx specifies the increment for the elments of x.
            incx must not be zero.
    
    @param[in]
    beta    DOUBLE PRECISION.
            On entry, BETA specifies the scalar beta.

    @param[out]
    y_array y = y_array[i]:       
            On exit y = alpha opt(A) x + beta y.
            y: COMPLEX*16 array of dimension.
            m if trans == MagmaNoTrans.
            n if trans == MagmaTrans or MagmaConjTrans.

    @param[in]
    incy    INTEGER.
            incy specifies the increment for the elments of y.
            incy must not be zero.
    
    @param[in]
    batchCount INTEGER
            number of pointers contained in dA_array, x_array and y_array.

    @ingroup magma_zblas2
    *******************************************************************   */

extern "C"
void magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t ldda, 
    magmaDoubleComplex_ptr dx_array[], magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t incy, 
    magma_int_t batchCount, magma_queue_t queue)
{       
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if(m==0 || n ==0 ) return;

    if ( trans == MagmaNoTrans ) {

        magmablas_zgemvn_batched(m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
            
    }
    else if ( trans == MagmaTrans ) {
        magmablas_zgemvt_batched(m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
    }
    else if ( trans == MagmaConjTrans ) {
#if defined(PRECISION_z) || defined (PRECISION_c)
        magmablas_zgemvc_batched(m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
#else
        magmablas_zgemvt_batched(m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue);
#endif
    }
    else {
        fprintf( stderr, "trans = %c is invalid\n", lapacke_trans_const(trans) );
    }
}

#undef zgemv_bs 
