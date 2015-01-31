/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zungqr_2stage_gpu.cpp normal z -> s, Fri Jan 30 19:00:18 2015

*/
#include "common_magma.h"

/**
    Purpose
    -------
    SORGQR generates an M-by-N REAL matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by SGEQRF_GPU.

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
    dA      REAL array A on the GPU device,
            dimension (LDDA,N). On entry, the i-th column must contain
            the vector which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by SGEQRF_GPU in the first k
            columns of its array argument A.
            On exit, the M-by-N matrix Q.

    @param[in]
    ldda    INTEGER
            The first dimension of the array A. LDDA >= max(1,M).

    @param[in]
    tau     REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQRF_GPU.

    @param[in]
    dT      REAL work space array on the GPU device,
            dimension (MIN(M, N) )*NB.
            This must be the 6th argument of magma_sgeqrf_gpu
            [ note that if N here is bigger than N in magma_sgeqrf_gpu,
              the workspace requirement DT in magma_sgeqrf_gpu must be
              as specified in this routine ].

    @param[in]
    nb      INTEGER
            This is the block size used in SGEQRF_GPU, and correspondingly
            the size of the T matrices, used in the factorization, and
            stored in DT.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument has an illegal value

    @ingroup magma_ssyev_2stage
    ********************************************************************/
extern "C" magma_int_t
magma_sorgqr_2stage_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info)
{
    #define dA(a_1,a_2) (dA + (a_2)*(ldda) + (a_1))
    #define dT(a_1)     (dT + (a_1)*nb)

    float c_zero = MAGMA_S_ZERO;
    float c_one  = MAGMA_S_ONE;
    
    magma_int_t  i__1, i__2, i__3;
    //magma_int_t lwork;
    magma_int_t i, ib, ki, kk;  //, iinfo;
    //magma_int_t lddwork = min(m, n);
    //float *work, *panel;
    magmaFloat_ptr dwork;
    //magma_queue_t stream[2];
    magma_int_t ldt=nb; // need to be an input parameter

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

    if (n <= 0)
        return *info;

    if (MAGMA_SUCCESS != magma_smalloc( &dwork, n*nb )) {
        printf ("!!!! sorgqr_2stage magma_alloc failed for: dwork\n" );
        exit(-1);
    }

    if ( (nb > 1) && (nb < k) ) {
        /*  Use blocked code after the last block.
            The first kk columns are handled by the block method.
            ki is start of 2nd-to-last block. */
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);

        /* Set A(1:kk,kk+1:n) to zero. */
        /* and A(kk+1:m, kk+1:n) = I */
        magmablas_slaset( MagmaFull, kk,   n-kk, c_zero, c_zero, dA(0, kk), ldda );
        magmablas_slaset( MagmaFull, m-kk, n-kk, c_zero, c_one,  dA(kk,kk), ldda );
    }
    else {
        ki = 0;
        kk = 0;
    }
    
    /* Allocate work space on CPU in pinned memory */
    //lwork = (n+m) * nb;
    //if (kk < n)
    //  lwork = max(lwork, n * nb + (m-kk)*(n-kk));

    //if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, (lwork) )) {
    //    *info = MAGMA_ERR_HOST_ALLOC;
    //    return *info;
    //}
    //panel = work + n * nb;

    //magma_queue_create( &stream[0] );
    //magma_queue_create( &stream[1] );
    /* Use unblocked code for the last or only block. */
    if (kk < n) {
        i__1 = m - kk;
        i__2 = n - kk;
        i__3 = k - kk;
        //magma_sgetmatrix(i__1, i__2, dA(kk, kk), ldda, panel, i__1);
        //lapackf77_sorgqr(&i__1, &i__2, &i__3, panel, &i__1, &tau[kk],
        //                 work, &lwork, &iinfo);
        //
        //magma_ssetmatrix(i__1, i__2, panel, i__1, dA(kk, kk), ldda);
        
        magma_slarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                          i__1, i__2, i__3,
                          dA(kk, kk-nb), ldda, dT(kk-nb), ldt,
                          dA(kk, kk), ldda, dwork, i__2);
        
        //magmablas_slaset(MagmaFull, kk-nb,     nb, c_zero, c_zero, dA(0,kk-nb),     ldda);
        //magmablas_slaset(MagmaFull, m-(kk-nb), nb, c_zero, c_one,  dA(kk-nb,kk-nb), ldda);
    }

    if (kk > 0) {
        /* Use blocked code */
        for (i = ki; i >= nb; i -= nb) {
            ib = min(nb, k - i);
            /* Send current panel to the CPU for update */
            i__2 = m - i;
            //magma_sgetmatrix_async( i__2, ib, dA(i,i), ldda, panel, i__2, stream[0] );  // verify
            if (i + ib < n) {
                /* Apply H to A(i:m,i+ib:n) from the left */
                i__3 = n - i;

                magmablas_slaset( MagmaFull, i,   ib, c_zero, c_zero, dA(0,i), ldda );
                magmablas_slaset( MagmaFull, m-i, ib, c_zero, c_one,  dA(i,i), ldda );

                magma_slarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  i__2, i__3, ib,
                                  dA(i, i-nb), ldda, dT(i-nb),             ldt,
                                  dA(i, i), ldda, dwork, i__3);
            }

            /* Apply H to rows i:m of current block on the CPU */
            //magma_queue_sync( stream[0] );
            //lapackf77_sorgqr(&i__2, &ib, &ib, panel, &i__2, &tau[i],
            //                 work, &lwork, &iinfo);
            //magma_ssetmatrix_async( i__2, ib, panel, i__2, dA(i,i), ldda, stream[1] );  // verify

            /* Set rows 1:i-1 of current block to zero */
            i__2 = i + ib;
            //magmablas_slaset(MagmaFull, i-ib,     ib, c_zero, c_zero, dA(0,i-ib),    ldda);
            //magmablas_slaset(MagmaFull, m-(i-ib), ib, c_zero, c_one,  dA(i-ib,i-ib), ldda);
        }
    }

    magmablas_slaset( MagmaFull, m, nb, c_zero, c_one, dA(0,0), ldda );

    magma_free( dwork );
    //magma_free_pinned( work );
    //magma_queue_destroy( stream[0] );
    //magma_queue_destroy( stream[1] );

    return *info;
} /* magma_sorgqr_gpu */

#undef dA
#undef dT
