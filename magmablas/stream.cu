/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014
*/

#include "common_magma.h"

magma_queue_t magma_stream = 0;


/**
    Purpose
    -------
    magmablasSetKernelStream sets the CUDA stream that all MAGMA BLAS and
    CUBLAS routines use.

    Arguments
    ---------
    @param[in]
    stream  magma_queue_t
            The CUDA stream.

    @ingroup magma_s
    ********************************************************************/
extern "C"
cublasStatus_t magmablasSetKernelStream( magma_queue_t stream )
{
    magma_stream = stream;
    return cublasSetKernelStream( stream );
}


/**
    Purpose
    -------
    magmablasSetKernelStream gets the CUDA stream that all MAGMA BLAS
    routines use.

    Arguments
    ---------
    @param[out]
    stream  magma_queue_t
            The CUDA stream.

    @ingroup magma_s
    ********************************************************************/
extern "C"
cublasStatus_t magmablasGetKernelStream( magma_queue_t *stream )
{
    *stream = magma_stream;
    return CUBLAS_STATUS_SUCCESS;
}
