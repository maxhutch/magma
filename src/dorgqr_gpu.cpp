/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zungqr_gpu.cpp normal z -> d, Mon May  2 23:30:07 2016

       @author Stan Tomov
       @author Mark Gates
*/
#include "magma_internal.h"

/**
    Purpose
    -------
    DORGQR generates an M-by-N DOUBLE PRECISION matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by DGEQRF_GPU.

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
    dA      DOUBLE PRECISION array A on the GPU, dimension (LDDA,N).
            On entry, the i-th column must contain the vector
            which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by DGEQRF_GPU in the
            first k columns of its array argument A.
            On exit, the M-by-N matrix Q.

    @param[in]
    ldda    INTEGER
            The first dimension of the array A. LDDA >= max(1,M).

    @param[in]
    tau     DOUBLE PRECISION array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by DGEQRF_GPU.

    @param[in]
    dT      (workspace) DOUBLE PRECISION work space array on the GPU,
            dimension (2*MIN(M, N) + ceil(N/32)*32 )*NB.
            This must be the 6th argument of magma_dgeqrf_gpu
            [ note that if N here is bigger than N in magma_dgeqrf_gpu,
              the workspace requirement DT in magma_dgeqrf_gpu must be
              as specified in this routine ].

    @param[in]
    nb      INTEGER
            This is the block size used in DGEQRF_GPU, and correspondingly
            the size of the T matrices, used in the factorization, and
            stored in DT.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_dgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dorgqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT, magma_int_t nb,
    magma_int_t *info)
{
#define dA(i,j) (dA + (i) + (j)*ldda)
#define dT(j)   (dT + (j)*nb)

    double c_zero = MAGMA_D_ZERO;
    double c_one  = MAGMA_D_ONE;
    
    magma_int_t m_kk, n_kk, k_kk, mi;
    magma_int_t lwork, lpanel;
    magma_int_t i, ib, ki, kk, iinfo;
    magma_int_t lddwork;
    magmaDouble_ptr dV, dW;
    double *work, *panel;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if ((n < 0) || (n > m)) {
        *info = -2;
    } else if ((k < 0) || (k > n)) {
        *info = -3;
    } else if (ldda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (n <= 0) {
        return *info;
    }

    // first kk columns are handled by blocked method.
    // ki is start of 2nd-to-last block
    if ((nb > 1) && (nb < k)) {
        ki = (k - nb - 1) / nb * nb;
        kk = min( k, ki+nb );
    } else {
        ki = 0;
        kk = 0;
    }

    // Allocate CPU work space
    // n*nb for dorgqr workspace
    // (m - kk)*(n - kk) for last block's panel
    lwork = n*nb;
    lpanel = (m - kk)*(n - kk);
    magma_dmalloc_cpu( &work, lwork + lpanel );
    if ( work == NULL ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    panel = work + lwork;
    
    // Allocate work space on GPU
    if (MAGMA_SUCCESS != magma_dmalloc( &dV, ldda*nb )) {
        magma_free_cpu( work );
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    // dT workspace has:
    // 2*min(m,n)*nb      for T and R^{-1} matrices from geqrf
    // roundup(n,32) * nb for dW larfb workspace.
    lddwork = min(m,n);
    dW = dT + 2*lddwork*nb;

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // Use unblocked code for the last or only block.
    if (kk < n) {
        m_kk = m - kk;
        n_kk = n - kk;
        k_kk = k - kk;
        magma_dgetmatrix( m_kk, k_kk,
                          dA(kk, kk), ldda, panel, m_kk, queue );
        
        lapackf77_dorgqr( &m_kk, &n_kk, &k_kk,
                          panel, &m_kk,
                          &tau[kk], work, &lwork, &iinfo );
        
        magma_dsetmatrix( m_kk, n_kk,
                          panel, m_kk, dA(kk, kk), ldda, queue );
        
        // Set A(1:kk,kk+1:n) to zero.
        magmablas_dlaset( MagmaFull, kk, n - kk, c_zero, c_zero, dA(0, kk), ldda, queue );
    }

    if (kk > 0) {
        // Use blocked code
        // queue:  copy Aii to V --> laset --> laset --> larfb --> [next]
        // CPU has no computation
        
        for (i = ki; i >= 0; i -= nb) {
            ib = min( nb, k-i );
            mi = m - i;
            
            // Copy current panel on the GPU from dA to dV
            magma_dcopymatrix_async( mi, ib,
                                     dA(i,i), ldda,
                                     dV,      ldda, queue );

            // set panel to identity
            magmablas_dlaset( MagmaFull, i,  ib, c_zero, c_zero, dA(0, i), ldda, queue );
            magmablas_dlaset( MagmaFull, mi, ib, c_zero, c_one,  dA(i, i), ldda, queue );
            
            if (i < n) {
                // Apply H to A(i:m,i:n) from the left
                magma_dlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  mi, n-i, ib,
                                  dV,       ldda, dT(i), nb,
                                  dA(i, i), ldda, dW, lddwork, queue );
            }
        }
    }
    magma_queue_sync( queue );

    magma_free( dV );
    magma_free_cpu( work );
    magma_queue_destroy( queue );

    return *info;
} /* magma_dorgqr_gpu */
