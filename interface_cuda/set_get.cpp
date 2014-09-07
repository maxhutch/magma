/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014
 
       @author Mark Gates
       @precisions normal z -> s d c
*/

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

// generic, type-independent routines to copy data.
// type-safe versions which avoid the user needing sizeof(...) are in copy_[sdcz].cpp

// ========================================
// copying vectors
extern "C"
void magma_setvector_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* hx_src, magma_int_t incx,
    void*       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVector(
        n, elemSize,
        hx_src, incx,
        dy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_getvector_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* dx_src, magma_int_t incx,
    void*       hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVector(
        n, elemSize,
        dx_src, incx,
        hy_dst, incy );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_setvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* hx_src, magma_int_t incx,
    void*       dy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetVectorAsync(
        n, elemSize,
        hx_src, incx,
        dy_dst, incy, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_getvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* dx_src, magma_int_t incx,
    void*       hy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetVectorAsync(
        n, elemSize,
        dx_src, incx,
        hy_dst, incy, stream );
    check_xerror( status, func, file, line );
}

// --------------------
// TODO compare performance with cublasZcopy BLAS function.
// But this implementation can handle any element size, not just [sdcz] precisions.
extern "C"
void magma_copyvector_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* dx_src, magma_int_t incx,
    void*       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    if ( incx == 1 && incy == 1 ) {
        cudaError_t status;
        status = cudaMemcpy(
            dy_dst,
            dx_src,
            n*elemSize, cudaMemcpyDeviceToDevice );
        check_xerror( status, func, file, line );
    }
    else {
        magma_copymatrix_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, func, file, line );
    }
}

// --------------------
extern "C"
void magma_copyvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    void const* dx_src, magma_int_t incx,
    void*       dy_dst, magma_int_t incy,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    if ( incx == 1 && incy == 1 ) {
        cudaError_t status;
        status = cudaMemcpyAsync(
            dy_dst,
            dx_src,
            n*elemSize, cudaMemcpyDeviceToDevice, stream );
        check_xerror( status, func, file, line );
    }
    else {
        magma_copymatrix_async_internal(
            1, n, elemSize, dx_src, incx, dy_dst, incy, stream, func, file, line );
    }
}


// ========================================
// copying sub-matrices (contiguous columns)
extern "C"
void magma_setmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* hA_src, magma_int_t lda,
    void*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrix(
        m, n, elemSize,
        hA_src, lda,
        dB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_getmatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* dA_src, magma_int_t lda,
    void*       hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrix(
        m, n, elemSize,
        dA_src, lda,
        hB_dst, ldb );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* hA_src, magma_int_t lda,
    void*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasSetMatrixAsync(
        m, n, elemSize,
        hA_src, lda,
        dB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* dA_src, magma_int_t lda,
    void*       hB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cublasStatus_t status;
    status = cublasGetMatrixAsync(
        m, n, elemSize,
        dA_src, lda,
        hB_dst, ldb, stream );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_copymatrix_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* dA_src, magma_int_t lda,
    void*       dB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2D(
        dB_dst, ldb*elemSize,
        dA_src, lda*elemSize,
        m*elemSize, n, cudaMemcpyDeviceToDevice );
    check_xerror( status, func, file, line );
}

// --------------------
extern "C"
void magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    void const* dA_src, magma_int_t lda,
    void*       dB_dst, magma_int_t ldb,
    cudaStream_t stream,
    const char* func, const char* file, int line )
{
    cudaError_t status;
    status = cudaMemcpy2DAsync(
        dB_dst, ldb*elemSize,
        dA_src, lda*elemSize,
        m*elemSize, n, cudaMemcpyDeviceToDevice, stream );
    check_xerror( status, func, file, line );
}

#endif // HAVE_CUBLAS
