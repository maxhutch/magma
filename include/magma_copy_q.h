#ifndef MAGMA_COPY_Q_H
#define MAGMA_COPY_Q_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


// ========================================
// copying vectors
// set copies host to device
// get copies device to host
// Add the function, file, and line for error-reporting purposes.

#define magma_setvector_q(               n, elemSize, hx_src, incx, dy_dst, incy, queue ) \
        magma_setvector_q_internal(      n, elemSize, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_getvector_q(               n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_q_internal(      n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_copyvector_q(              n, elemSize, dx_src, incx, dy_dst, incy, queue ) \
        magma_copyvector_q_internal(     n, elemSize, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_setvector_async(           n, elemSize, hx_src, incx, dy_dst, incy, queue ) \
        magma_setvector_async_internal(  n, elemSize, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_getvector_async(           n, elemSize, dx_src, incx, hy_dst, incy, queue ) \
        magma_getvector_async_internal(  n, elemSize, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_copyvector_async(          n, elemSize, dx_src, incx, dy_dst, incy, queue ) \
        magma_copyvector_async_internal( n, elemSize, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void magma_setvector_q_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_getvector_q_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_copyvector_q_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_setvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *hx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_getvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_copyvector_async_internal(
    magma_int_t n, magma_int_t elemSize,
    const void *dx_src, magma_int_t incx,
    void       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns )
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_setmatrix_q(               m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_setmatrix_q_internal(      m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_getmatrix_q(               m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_getmatrix_q_internal(      m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_copymatrix_q(              m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_copymatrix_q_internal(     m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_setmatrix_async(           m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_setmatrix_async_internal(  m, n, elemSize, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_getmatrix_async(           m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_getmatrix_async_internal(  m, n, elemSize, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_copymatrix_async(          m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_copymatrix_async_internal( m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void magma_setmatrix_q_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    void       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_getmatrix_q_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_copymatrix_q_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_setmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *hA_src, magma_int_t lda,
    void       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_getmatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_copymatrix_async_internal(
    magma_int_t m, magma_int_t n, magma_int_t elemSize,
    const void *dA_src, magma_int_t ldda,
    void       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// copying vectors - version for magma_int_t
// TODO to make these truly type-safe, would need intermediate inline
//      magma_i* functions that call the generic magma_* functions.
//      Could do the same with magma_[sdcz]* set/get functions.

#define magma_isetvector_q(               n, hx_src, incx, dy_dst, incy, queue ) \
        magma_isetvector_q_internal(      n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_igetvector_q(               n, dx_src, incx, hy_dst, incy, queue ) \
        magma_igetvector_q_internal(      n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_icopyvector_q(              n, dx_src, incx, dy_dst, incy, queue ) \
        magma_icopyvector_q_internal(     n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_isetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_isetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_igetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_igetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_icopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_icopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void magma_isetvector_q_internal(
    magma_int_t n,
    const magma_int_t *hx_src, magma_int_t incx,
    magma_int_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_q_internal( n, sizeof(magma_int_t), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void magma_igetvector_q_internal(
    magma_int_t n,
    const magma_int_t *dx_src, magma_int_t incx,
    magma_int_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_q_internal( n, sizeof(magma_int_t), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void magma_icopyvector_q_internal(
    magma_int_t n,
    const magma_int_t *dx_src, magma_int_t incx,
    magma_int_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_q_internal( n, sizeof(magma_int_t), dx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void magma_isetvector_async_internal(
    magma_int_t n,
    const magma_int_t *hx_src, magma_int_t incx,
    magma_int_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_async_internal( n, sizeof(magma_int_t), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void magma_igetvector_async_internal(
    magma_int_t n,
    const magma_int_t *dx_src, magma_int_t incx,
    magma_int_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_async_internal( n, sizeof(magma_int_t), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void magma_icopyvector_async_internal(
    magma_int_t n,
    const magma_int_t *dx_src, magma_int_t incx,
    magma_int_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_async_internal( n, sizeof(magma_int_t), dx_src, incx, dy_dst, incy, queue, func, file, line ); }


// ========================================
// copying sub-matrices - version for magma_int_t

#define magma_isetmatrix_q(               m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_isetmatrix_q_internal(      m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_igetmatrix_q(               m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_igetmatrix_q_internal(      m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_icopymatrix_q(              m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_icopymatrix_q_internal(     m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_isetmatrix_async(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_isetmatrix_async_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_igetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_igetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_icopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_icopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void magma_isetmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *hA_src, magma_int_t lda,
    magma_int_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_q_internal( m, n, sizeof(magma_int_t), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void magma_igetmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *dA_src, magma_int_t ldda,
    magma_int_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_q_internal( m, n, sizeof(magma_int_t), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void magma_icopymatrix_q_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *dA_src, magma_int_t ldda,
    magma_int_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_q_internal( m, n, sizeof(magma_int_t), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }

static inline void magma_isetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *hA_src, magma_int_t lda,
    magma_int_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_async_internal( m, n, sizeof(magma_int_t), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void magma_igetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *dA_src, magma_int_t ldda,
    magma_int_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_async_internal( m, n, sizeof(magma_int_t), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void magma_icopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_int_t *dA_src, magma_int_t ldda,
    magma_int_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_async_internal( m, n, sizeof(magma_int_t), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }


// ========================================
// copying vectors - version for magma_index_t

#define magma_index_setvector_q(               n, hx_src, incx, dy_dst, incy, queue ) \
        magma_index_setvector_q_internal(      n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_getvector_q(               n, dx_src, incx, hy_dst, incy, queue ) \
        magma_index_getvector_q_internal(      n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_copyvector_q(              n, dx_src, incx, dy_dst, incy, queue ) \
        magma_index_copyvector_q_internal(     n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_setvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_index_setvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_getvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_index_getvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_index_copyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_index_copyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void magma_index_setvector_q_internal(
    magma_int_t n,
    const magma_index_t *hx_src, magma_int_t incx,
    magma_index_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_q_internal( n, sizeof(magma_index_t), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void magma_index_getvector_q_internal(
    magma_int_t n,
    const magma_index_t *dx_src, magma_int_t incx,
    magma_index_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_q_internal( n, sizeof(magma_index_t), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void magma_index_copyvector_q_internal(
    magma_int_t n,
    const magma_index_t *dx_src, magma_int_t incx,
    magma_index_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_q_internal( n, sizeof(magma_index_t), dx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void magma_index_setvector_async_internal(
    magma_int_t n,
    const magma_index_t *hx_src, magma_int_t incx,
    magma_index_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_async_internal( n, sizeof(magma_index_t), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void magma_index_getvector_async_internal(
    magma_int_t n,
    const magma_index_t *dx_src, magma_int_t incx,
    magma_index_t       *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_async_internal( n, sizeof(magma_index_t), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void magma_index_copyvector_async_internal(
    magma_int_t n,
    const magma_index_t *dx_src, magma_int_t incx,
    magma_index_t       *dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_async_internal( n, sizeof(magma_index_t), dx_src, incx, dy_dst, incy, queue, func, file, line ); }


// ========================================
// copying sub-matrices - version for magma_index_t

#define magma_index_setmatrix_q(               m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_index_setmatrix_q_internal(      m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_index_getmatrix_q(               m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_index_getmatrix_q_internal(      m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_index_copymatrix_q(              m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_index_copymatrix_q_internal(     m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_index_setmatrix_async(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_index_setmatrix_async_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_index_getmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_index_getmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_index_copymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_index_copymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void magma_index_setmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *hA_src, magma_int_t lda,
    magma_index_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_q_internal( m, n, sizeof(magma_index_t), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void magma_index_getmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *dA_src, magma_int_t ldda,
    magma_index_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_q_internal( m, n, sizeof(magma_index_t), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void magma_index_copymatrix_q_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *dA_src, magma_int_t ldda,
    magma_index_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_q_internal( m, n, sizeof(magma_index_t), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }

static inline void magma_index_setmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *hA_src, magma_int_t lda,
    magma_index_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_async_internal( m, n, sizeof(magma_index_t), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void magma_index_getmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *dA_src, magma_int_t ldda,
    magma_index_t       *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_async_internal( m, n, sizeof(magma_index_t), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void magma_index_copymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    const magma_index_t *dA_src, magma_int_t ldda,
    magma_index_t       *dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_async_internal( m, n, sizeof(magma_index_t), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }


#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_COPY_Q_H
