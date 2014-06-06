/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
*/

#include "common_magma.h"

magma_queue_t magma_stream = 0;

cublasStatus_t magmablasSetKernelStream( magma_queue_t stream )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    magmablasSetKernelStream sets the CUDA stream that all MAGMA BLAS and
    CUBLAS routines use.

    Arguments
    =========
    stream  (input) magma_queue_t
            The CUDA stream.

    =====================================================================   */
    magma_stream = stream;
    return cublasSetKernelStream( stream );
}


cublasStatus_t magmablasGetKernelStream( magma_queue_t *stream )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    magmablasSetKernelStream gets the CUDA stream that all MAGMA BLAS
    routines use.

    Arguments
    =========
    stream  (output) magma_queue_t
            The CUDA stream.

    =====================================================================   */
    *stream = magma_stream;
    return CUBLAS_STATUS_SUCCESS;
}
