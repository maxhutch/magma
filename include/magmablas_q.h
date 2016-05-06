/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
*/

#ifndef MAGMABLAS_Q_H
#define MAGMABLAS_Q_H

#include "magmablas_z_q.h"
#include "magmablas_c_q.h"
#include "magmablas_d_q.h"
#include "magmablas_s_q.h"
#include "magmablas_zc_q.h"
#include "magmablas_ds_q.h"

#ifdef __cplusplus
extern "C" {
#endif


// ========================================
// queue support
// new magma_queue_create adds device
#define magma_queue_create_v2( device, queue_ptr ) \
        magma_queue_create_v2_internal( device, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_create_from_cuda( device, stream, cublas, cusparse, queue_ptr ) \
        magma_queue_create_from_cuda_internal( device, stream, cublas, cusparse, queue_ptr, __func__, __FILE__, __LINE__ )

#define magma_queue_destroy( queue ) \
        magma_queue_destroy_internal( queue, __func__, __FILE__, __LINE__ )

#define magma_queue_sync( queue ) \
        magma_queue_sync_internal( queue, __func__, __FILE__, __LINE__ )

size_t
magma_queue_mem_size( magma_queue_t queue );

void magma_queue_create_v2_internal(
    magma_device_t device,
    magma_queue_t* queue_ptr,
    const char* func, const char* file, int line );

void magma_queue_create_from_cuda_internal(
    magma_device_t   device,
    cudaStream_t     stream,
    cublasHandle_t   cublas,
    cusparseHandle_t cusparse,
    magma_queue_t*   queue_ptr,
    const char* func, const char* file, int line );

void magma_queue_destroy_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_queue_sync_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line );


#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_Q_H */
