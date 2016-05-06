/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c

       @author Mark Gates
*/
#include "magma_internal.h"
#include "trace.h"

/**
    Purpose
    -------
    ZUNGQR generates an M-by-N COMPLEX_16 matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

        Q  =  H(1) H(2) . . . H(k)

    as returned by ZGEQRF.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix Q. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    @param[in,out]
    A       COMPLEX_16 array A, dimension (LDDA,N).
            On entry, the i-th column must contain the vector
            which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by ZGEQRF_GPU in the
            first k columns of its array argument A.
            On exit, the M-by-N matrix Q.

    @param[in]
    lda     INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF_GPU.

    @param[in]
    T       COMPLEX_16 array, dimension (NB, min(M,N)).
            T contains the T matrices used in blocking the elementary
            reflectors H(i), e.g., this can be the 6th argument of
            magma_zgeqrf_gpu (except stored on the CPU, not the GPU).

    @param[in]
    nb      INTEGER
            This is the block size used in ZGEQRF_GPU, and correspondingly
            the size of the T matrices, used in the factorization, and
            stored in T.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zungqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t nb,
    magma_int_t *info)
{
#define  A(i,j)   ( A    + (i) + (j)*lda )
#define dA(d,i,j) (dA[d] + (i) + (j)*ldda)
#define dT(d,i,j) (dT[d] + (i) + (j)*nb)

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    magma_int_t m_kk, n_kk, k_kk, mi;
    magma_int_t lwork, ldwork;
    magma_int_t d, i, ib, j, jb, ki, kk;
    magmaDoubleComplex *work=NULL;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if ((n < 0) || (n > m)) {
        *info = -2;
    } else if ((k < 0) || (k > n)) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (n <= 0) {
        return *info;
    }
    
    magma_int_t di, dn;
    magma_int_t dpanel;

    magma_int_t ngpu = magma_num_gpus();
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    // Allocate memory on GPUs for A and workspaces
    magma_int_t ldda    = magma_roundup( m, 32 );
    magma_int_t lddwork = magma_roundup( n, 32 );
    magma_int_t min_lblocks = (n / nb) / ngpu;  // min. blocks per gpu
    magma_int_t last_dev    = (n / nb) % ngpu;  // device with last block
    
    magma_int_t  nlocal[ MagmaMaxGPUs ] = { 0 };
    magmaDoubleComplex *dA[ MagmaMaxGPUs ] = { NULL };
    magmaDoubleComplex *dT[ MagmaMaxGPUs ] = { NULL };
    magmaDoubleComplex *dV[ MagmaMaxGPUs ] = { NULL };
    magmaDoubleComplex *dW[ MagmaMaxGPUs ] = { NULL };
    magma_queue_t queues[ MagmaMaxGPUs ] = { NULL };
    
    for( d = 0; d < ngpu; ++d ) {
        // example with n = 75, nb = 10, ngpu = 3
        // min_lblocks = 2
        // last_dev    = 1
        // gpu 0: 2  blocks, cols:  0- 9, 30-39, 60-69
        // gpu 1: 1+ blocks, cols: 10-19, 40-49, 70-74 (partial)
        // gpu 2: 1  block,  cols: 20-29, 50-59
        magma_setdevice( d );
        nlocal[d] = min_lblocks*nb;
        if ( d < last_dev ) {
            nlocal[d] += nb;
        }
        else if ( d == last_dev ) {
            nlocal[d] += (n % nb);
        }
        
        ldwork = nlocal[d]*ldda  // dA
               + nb*m            // dT
               + nb*ldda         // dV
               + nb*lddwork;     // dW
        if ( MAGMA_SUCCESS != magma_zmalloc( &dA[d], ldwork )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
        dT[d] = dA[d] + nlocal[d]*ldda;
        dV[d] = dT[d] + nb*m;
        dW[d] = dV[d] + nb*ldda;
        
        magma_queue_create( d, &queues[d] );
    }
    
    trace_init( 1, ngpu, 1, queues );
    
    // first kk columns are handled by blocked method.
    // ki is start of 2nd-to-last block
    if ((nb > 1) && (nb < k)) {
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);
    } else {
        ki = 0;
        kk = 0;
    }

    // Allocate CPU work space
    // n*nb  for larfb work
    // m*nb  for V
    // nb*nb for T
    lwork = (n + m + nb) * nb;
    magma_zmalloc_cpu( &work, lwork );
    if (work == NULL) {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto cleanup;
    }
    magmaDoubleComplex *work_T, *work_V;
    work_T = work + n*nb;
    work_V = work + n*nb + nb*nb;

    // Use unblocked code for the last or only block.
    if (kk < n) {
        trace_cpu_start( 0, "ungqr", "ungqr last block" );
        m_kk = m - kk;
        n_kk = n - kk;
        k_kk = k - kk;
        
        // zungqr requires less workspace (n*nb), but is slow if k < zungqr's block size.
        // replacing it with the 4 routines below is much faster (e.g., 60x).
        //magma_int_t iinfo;
        //lapackf77_zungqr( &m_kk, &n_kk, &k_kk,
        //                  A(kk, kk), &lda,
        //                  &tau[kk], work, &lwork, &iinfo );
        
        lapackf77_zlacpy( MagmaFullStr, &m_kk, &k_kk, A(kk,kk), &lda, work_V, &m_kk);
        lapackf77_zlaset( MagmaFullStr, &m_kk, &n_kk, &c_zero, &c_one, A(kk, kk), &lda );
        
        lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                          &m_kk, &k_kk,
                          work_V, &m_kk, &tau[kk], work_T, &k_kk);
        lapackf77_zlarfb( MagmaLeftStr, MagmaNoTransStr, MagmaForwardStr, MagmaColumnwiseStr,
                          &m_kk, &n_kk, &k_kk,
                          work_V, &m_kk, work_T, &k_kk, A(kk, kk), &lda, work, &n_kk );
        
        if (kk > 0) {
            for( j=kk; j < n; j += nb ) {
                jb = min( n-j, nb );
                d  =  (j / nb) % ngpu;
                di = ((j / nb) / ngpu) * nb;
                magma_setdevice( d );
                magma_zsetmatrix( m_kk, jb,
                                  A(kk, j),  lda,
                                  dA(d, kk, di), ldda, queues[d] );
                
                // Set A(1:kk,kk+1:n) to zero.
                magmablas_zlaset( MagmaFull, kk, jb, c_zero, c_zero, dA(d, 0, di), ldda, queues[d] );
            }
        }
        trace_cpu_end( 0 );
    }

    if (kk > 0) {
        // Use blocked code
        // send T to all GPUs
        for( d = 0; d < ngpu; ++d ) {
            magma_setdevice( d );
            trace_gpu_start( d, 0, "set", "set T" );
            magma_zsetmatrix_async( nb, min(m,n), T, nb, dT[d], nb, queues[d] );
            trace_gpu_end( d, 0 );
        }
        
        // queue: set Aii (V) --> laset --> laset --> larfb --> [next]
        // CPU has no computation
        for( i = ki; i >= 0; i -= nb ) {
            ib = min(nb, k - i);
            mi = m - i;
            dpanel =  (i / nb) % ngpu;
            di     = ((i / nb) / ngpu) * nb;

            // Send current panel to dV on the GPUs
            lapackf77_zlaset( "Upper", &ib, &ib, &c_zero, &c_one, A(i, i), &lda );
            for( d = 0; d < ngpu; ++d ) {
                magma_setdevice( d );
                trace_gpu_start( d, 0, "set", "set V" );
                magma_zsetmatrix_async( mi, ib,
                                        A(i, i), lda,
                                        dV[d],   ldda, queues[d] );
                trace_gpu_end( d, 0 );
            }
            
            // set panel to identity
            magma_setdevice( dpanel );
            trace_gpu_start( dpanel, 0, "laset", "laset" );
            magmablas_zlaset( MagmaFull, i,  ib, c_zero, c_zero, dA(dpanel, 0, di), ldda, queues[dpanel] );
            magmablas_zlaset( MagmaFull, mi, ib, c_zero, c_one,  dA(dpanel, i, di), ldda, queues[dpanel] );
            trace_gpu_end( dpanel, 0 );
            
            if (i < n) {
                // Apply H to A(i:m,i:n) from the left
                for( d = 0; d < ngpu; ++d ) {
                    magma_setdevice( d );
                    magma_indices_1D_bcyclic( nb, ngpu, d, i, n, &di, &dn );
                    trace_gpu_start( d, 0, "larfb", "larfb" );
                    magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                      mi, dn-di, ib,
                                      dV[d],        ldda, dT(d,0,i), nb,
                                      dA(d, i, di), ldda, dW[d], lddwork, queues[d] );
                    trace_gpu_end( d, 0 );
                }
            }
        }
        
        // copy result back to CPU
        trace_cpu_start( 0, "get", "get A" );
        magma_zgetmatrix_1D_col_bcyclic( m, n, dA, ldda, A, lda, ngpu, nb, queues );
        trace_cpu_end( 0 );
    }
    
    #ifdef TRACING
    char name[80];
    snprintf( name, sizeof(name), "zungqr-n%d-ngpu%d.svg", m, ngpu );
    trace_finalize( name, "trace.css" );
    #endif
    
cleanup:
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magma_free( dA[d] );
        magma_queue_destroy( queues[d] );
    }
    magma_free_cpu( work );
    magma_setdevice( orig_dev );
    
    return *info;
} /* magma_zungqr */
