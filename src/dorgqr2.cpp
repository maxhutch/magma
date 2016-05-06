/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zungqr2.cpp normal z -> d, Mon May  2 23:30:10 2016

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

    as returned by DGEQRF.

    This version recomputes the T matrices on the CPU and sends them to the GPU.

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
    A       DOUBLE PRECISION array A, dimension (LDDA,N).
            On entry, the i-th column must contain the vector
            which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by DGEQRF_GPU in the
            first k columns of its array argument A.
            On exit, the M-by-N matrix Q.

    @param[in]
    lda     INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    @param[in]
    tau     DOUBLE PRECISION array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by DGEQRF_GPU.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument has an illegal value

    @ingroup magma_dgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dorgqr2(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    magma_int_t *info)
{
#define  A(i,j) ( A + (i) + (j)*lda )
#define dA(i,j) (dA + (i) + (j)*ldda)

    double c_zero = MAGMA_D_ZERO;
    double c_one  = MAGMA_D_ONE;

    magma_int_t nb = magma_get_dgeqrf_nb( m, n );

    magma_int_t  m_kk, n_kk, k_kk, mi;
    magma_int_t lwork, ldda;
    magma_int_t i, ib, ki, kk;  //, iinfo;
    magma_int_t lddwork;
    double *dA, *dV, *dW, *dT, *T;
    double *work;

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

    // first kk columns are handled by blocked method.
    // ki is start of 2nd-to-last block
    if ((nb > 1) && (nb < k)) {
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);
    } else {
        ki = 0;
        kk = 0;
    }

    // Allocate GPU work space
    // ldda*n     for matrix dA
    // ldda*nb    for dV
    // lddwork*nb for dW larfb workspace
    ldda    = magma_roundup( m, 32 );
    lddwork = magma_roundup( n, 32 );
    if (MAGMA_SUCCESS != magma_dmalloc( &dA, ldda*n + ldda*nb + lddwork*nb + nb*nb)) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dV = dA + ldda*n;
    dW = dA + ldda*n + ldda*nb;
    dT = dA + ldda*n + ldda*nb + lddwork*nb;

    // Allocate CPU work space
    lwork = (n+m+nb) * nb;
    magma_dmalloc_cpu( &work, lwork );

    T = work;

    if (work == NULL) {
        magma_free( dA );
        magma_free_cpu( work );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    double *V = work + (n+nb)*nb;

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    // Use unblocked code for the last or only block.
    if (kk < n) {
        m_kk = m - kk;
        n_kk = n - kk;
        k_kk = k - kk;
        /*
            lapackf77_dorgqr( &m_kk, &n_kk, &k_kk,
                              A(kk, kk), &lda,
                              &tau[kk], work, &lwork, &iinfo );
        */
        lapackf77_dlacpy( MagmaFullStr, &m_kk, &k_kk, A(kk,kk), &lda, V, &m_kk);
        lapackf77_dlaset( MagmaFullStr, &m_kk, &n_kk, &c_zero, &c_one, A(kk, kk), &lda );

        lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                          &m_kk, &k_kk,
                          V, &m_kk, &tau[kk], work, &k_kk);
        lapackf77_dlarfb( MagmaLeftStr, MagmaNoTransStr, MagmaForwardStr, MagmaColumnwiseStr,
                          &m_kk, &n_kk, &k_kk,
                          V, &m_kk, work, &k_kk, A(kk, kk), &lda, work+k_kk*k_kk, &n_kk );
        
        if (kk > 0) {
            magma_dsetmatrix( m_kk, n_kk,
                              A(kk, kk),  lda,
                              dA(kk, kk), ldda, queue );
        
            // Set A(1:kk,kk+1:n) to zero.
            magmablas_dlaset( MagmaFull, kk, n - kk, c_zero, c_zero, dA(0, kk), ldda, queue );
        }
    }

    if (kk > 0) {
        // Use blocked code
        // queue: set Aii (V) --> laset --> laset --> larfb --> [next]
        // CPU has no computation
        
        for (i = ki; i >= 0; i -= nb) {
            ib = min(nb, k - i);

            // Send current panel to the GPU
            mi = m - i;
            lapackf77_dlaset( "Upper", &ib, &ib, &c_zero, &c_one, A(i, i), &lda );
            magma_dsetmatrix_async( mi, ib,
                                    A(i, i), lda,
                                    dV,      ldda, queue );
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &mi, &ib,
                              A(i,i), &lda, &tau[i], T, &nb);
            magma_dsetmatrix_async( ib, ib,
                                    T, nb,
                                    dT, nb, queue );

            // set panel to identity
            magmablas_dlaset( MagmaFull, i,  ib, c_zero, c_zero, dA(0, i), ldda, queue );
            magmablas_dlaset( MagmaFull, mi, ib, c_zero, c_one,  dA(i, i), ldda, queue );
            
            magma_queue_sync( queue );
            if (i < n) {
                // Apply H to A(i:m,i:n) from the left
                magma_dlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  mi, n-i, ib,
                                  dV,       ldda, dT, nb,
                                  dA(i, i), ldda, dW, lddwork, queue );
            }
        }
    
        // copy result back to CPU
        magma_dgetmatrix( m, n,
                          dA(0, 0), ldda, A(0, 0), lda, queue );
    }

    magma_queue_destroy( queue );
    magma_free( dA );
    magma_free_cpu( work );

    return *info;
} /* magma_dorgqr */
