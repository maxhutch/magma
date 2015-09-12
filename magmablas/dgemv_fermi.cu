/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015
       
       @author Mark Gates
       @author Tingxing Dong
       @author Azzam Haidar

       @generated from zgemv_fermi.cu normal z -> d, Fri Sep 11 18:29:22 2015
*/
#include "common_magma.h"
#include "commonblas_d.h"
#include "magma_templates.h"

#define PRECISION_d

#include "gemv_template_device.cuh"

#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v


//////////////////////////////////////////////////////////////////////////////////////////
// NoTrans kernel
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE>
__global__ void
dgemvn_template_kernel_fermi(
    int m, int n, double alpha,
    const double * __restrict__ A, int lda,
    const double * __restrict__ x, int incx, double beta,
    double       *y, int incy)
{
#if (__CUDA_ARCH__ >= 200)

    gemvn_template_device<double, DIM_X, DIM_Y, TILE_SIZE>(m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif /* (__CUDA_ARCH__ >= 200) */
}
//////////////////////////////////////////////////////////////////////////////////////////
// Trans/ConjTans kernel
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
__global__ void
dgemvc_template_kernel_fermi(
    int m, int n, double alpha,
    const double * __restrict__ A, int lda,
    const double * __restrict__ x, int incx, double beta,
    double       *y, int incy)
{
#if (__CUDA_ARCH__ >= 200)
    gemvc_template_device< double, DIM_X, DIM_Y, TILE_SIZE, trans >(m, n, alpha, A, lda, x, incx, beta, y, incy);
#endif /* (__CUDA_ARCH__ >= 200) */
}
//////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////
// NoTrans CPU driver
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
dgemvn_template_fermi(
    magma_int_t m, magma_int_t n, double alpha,
    const double * __restrict__ A, magma_int_t lda,
    const double * __restrict__ x, magma_int_t incx, double beta,
    double       *y, magma_int_t incy, magma_queue_t queue)
{
    dim3 grid( magma_ceildiv(m, TILE_SIZE) );
    dim3 threads( DIM_X, DIM_Y, 1 );

    dgemvn_template_kernel_fermi<DIM_X, DIM_Y, TILE_SIZE><<< grid, threads, 0, queue >>>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}
//////////////////////////////////////////////////////////////////////////////////////////
// Trans/ConjTans CPU driver
template<const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void
dgemvc_template_fermi(
    magma_trans_t trans, magma_int_t m, magma_int_t n, double alpha,
    const double * __restrict__ A, magma_int_t lda,
    const double * __restrict__ x, magma_int_t incx, double beta,
    double       *y, magma_int_t incy, magma_queue_t queue)
{
    dim3 grid    ( 1,  magma_ceildiv(n, TILE_SIZE),  1 );
    dim3 threads ( DIM_X, DIM_Y, 1 );

    if (trans == MagmaConjTrans)
    {
        dgemvc_template_kernel_fermi< DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans ><<< grid, threads, 0, queue >>>(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
    else
    {
        dgemvc_template_kernel_fermi< DIM_X, DIM_Y, TILE_SIZE, MagmaTrans ><<< grid, threads, 0, queue >>>(m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}



//////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------
    DGEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A
 
    @param[in]
    alpha   DOUBLE_PRECISION
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      DOUBLE_PRECISION array of dimension ( LDDA, n ) on the GPU.
   
    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx      DOUBLE_PRECISION array of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.
  
    @param[in]
    beta    DOUBLE_PRECISION
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      DOUBLE_PRECISION array of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void
magmablas_dgemv_q(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy, 
    magma_queue_t queue)
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

    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // call CUDA ARCH 1.x version
        // magmablas for [sd] precisions, cublas for [zc] precisions.
        #if defined(PRECISION_z) || defined(PRECISION_c)
        magma_dgemv( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
        #else
        magmablas_dgemv_tesla( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy );
        #endif
        return;
    }
    
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( trans == MagmaNoTrans ) {
        if (m <= 256) {
            dgemvn_template_fermi<version(N, 137)>
                ( m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue );
        }
        else {
            dgemvn_template_fermi<version(N, 140)>
                ( m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue );
        }
    }
    else {
        dgemvc_template_fermi<version(T, 189)>
            ( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue );
    }
}


extern "C" void
magmablas_dgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dgemv_q( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magma_stream);
}
