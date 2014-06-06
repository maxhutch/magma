/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
 
       @author Mark Gates
       @generated c Tue Dec 17 13:18:37 2013
*/

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// ========================================
// copying vectors
extern "C"
void magma_csetvector_internal(
    magma_int_t n,
    magmaFloatComplex const* hx_src, magma_int_t incx,
    magmaFloatComplex*       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVector(
        n, sizeof(magmaFloatComplex),
        hx_src, incx,
        dy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_cgetvector_internal(
    magma_int_t n,
    magmaFloatComplex const* dx_src, magma_int_t incx,
    magmaFloatComplex*       hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVector(
        n, sizeof(magmaFloatComplex),
        dx_src, incx,
        hy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_csetvector_async_internal(
    magma_int_t n,
    magmaFloatComplex const* hx_src, magma_int_t incx,
    magmaFloatComplex*       dy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        n, sizeof(magmaFloatComplex),
        hx_src, incx,
        dy_dst, incy, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_cgetvector_async_internal(
    magma_int_t n,
    magmaFloatComplex const* dx_src, magma_int_t incx,
    magmaFloatComplex*       hy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        n, sizeof(magmaFloatComplex),
        dx_src, incx,
        hy_dst, incy, stream );
    check_xerror( status, func, file, line );
}


// ========================================
// copying sub-matrices (contiguous columns)
extern "C"
void magma_csetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* hA_src, magma_int_t lda,
    magmaFloatComplex*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        m, n, sizeof(magmaFloatComplex),
        hA_src, lda,
        dB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_cgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* dA_src, magma_int_t lda,
    magmaFloatComplex*       hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        m, n, sizeof(magmaFloatComplex),
        dA_src, lda,
        hB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_csetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* hA_src, magma_int_t lda,
    magmaFloatComplex*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        m, n, sizeof(magmaFloatComplex),
        hA_src, lda,
        dB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_cgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* dA_src, magma_int_t lda,
    magmaFloatComplex*       hB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        m, n, sizeof(magmaFloatComplex),
        dA_src, lda,
        hB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_ccopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* dA_src, magma_int_t lda,
    magmaFloatComplex*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2D(
        dB_dst, ldb*sizeof(magmaFloatComplex),
        dA_src, lda*sizeof(magmaFloatComplex),
        m*sizeof(magmaFloatComplex), n, cudaMemcpyDeviceToDevice );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_ccopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const* dA_src, magma_int_t lda,
    magmaFloatComplex*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, ldb*sizeof(magmaFloatComplex),
        dA_src, lda*sizeof(magmaFloatComplex),
        m*sizeof(magmaFloatComplex), n, cudaMemcpyDeviceToDevice, stream );
    check_xerror( status, func, file, line );
}

#endif // HAVE_CUBLAS
