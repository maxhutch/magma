/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
 
       @author Mark Gates
       @generated s Tue Dec 17 13:18:37 2013
*/

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// ========================================
// copying vectors
extern "C"
void magma_ssetvector_internal(
    magma_int_t n,
    float const* hx_src, magma_int_t incx,
    float*       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVector(
        n, sizeof(float),
        hx_src, incx,
        dy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_sgetvector_internal(
    magma_int_t n,
    float const* dx_src, magma_int_t incx,
    float*       hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVector(
        n, sizeof(float),
        dx_src, incx,
        hy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_ssetvector_async_internal(
    magma_int_t n,
    float const* hx_src, magma_int_t incx,
    float*       dy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        n, sizeof(float),
        hx_src, incx,
        dy_dst, incy, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_sgetvector_async_internal(
    magma_int_t n,
    float const* dx_src, magma_int_t incx,
    float*       hy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        n, sizeof(float),
        dx_src, incx,
        hy_dst, incy, stream );
    check_xerror( status, func, file, line );
}


// ========================================
// copying sub-matrices (contiguous columns)
extern "C"
void magma_ssetmatrix_internal(
    magma_int_t m, magma_int_t n,
    float const* hA_src, magma_int_t lda,
    float*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        m, n, sizeof(float),
        hA_src, lda,
        dB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_sgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    float const* dA_src, magma_int_t lda,
    float*       hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        m, n, sizeof(float),
        dA_src, lda,
        hB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_ssetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    float const* hA_src, magma_int_t lda,
    float*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        m, n, sizeof(float),
        hA_src, lda,
        dB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_sgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    float const* dA_src, magma_int_t lda,
    float*       hB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        m, n, sizeof(float),
        dA_src, lda,
        hB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_scopymatrix_internal(
    magma_int_t m, magma_int_t n,
    float const* dA_src, magma_int_t lda,
    float*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2D(
        dB_dst, ldb*sizeof(float),
        dA_src, lda*sizeof(float),
        m*sizeof(float), n, cudaMemcpyDeviceToDevice );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_scopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    float const* dA_src, magma_int_t lda,
    float*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, ldb*sizeof(float),
        dA_src, lda*sizeof(float),
        m*sizeof(float), n, cudaMemcpyDeviceToDevice, stream );
    check_xerror( status, func, file, line );
}

#endif // HAVE_CUBLAS
