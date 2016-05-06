/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Raffaele Solca
       @author Azzam Haidar

       @precisions normal z -> s d c
*/
#include "magma_internal.h"


static void magma_zhegst_m_1_U_row_update(
    magma_int_t nk, magma_int_t nb,
    magmaDoubleComplex* dA_row, magma_int_t ldda,
    magmaDoubleComplex* dC1, magma_int_t lddc1,
    magmaDoubleComplex* dC2, magma_int_t lddc2,
    magma_queue_t queue )
{
    // update 1 rowblock (rowwise zher2k) for itype=1 Upper case
    const double             d_one      = 1.0;
    const magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;
    
    magma_int_t kb = min( nk, nb );
    
    magma_zher2k( MagmaUpper, MagmaConjTrans, kb, nb,
                  c_neg_one, dC1,    lddc1,
                             dC2,    lddc2,
                  d_one,     dA_row, ldda, queue );
    
    magma_zgemm( MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb,
                 c_neg_one, dC1,              lddc1,
                            dC2 + kb*lddc2,   lddc2,
                 c_one,     dA_row + kb*ldda, ldda, queue );
    
    magma_zgemm( MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb,
                 c_neg_one, dC2,              lddc2,
                            dC1 + kb*lddc1,   lddc1,
                 c_one,     dA_row + kb*ldda, ldda, queue );
}


static void magma_zhegst_m_1_L_col_update(
    magma_int_t nk, magma_int_t nb,
    magmaDoubleComplex* dA_col, magma_int_t ldda,
    magmaDoubleComplex* dC1, magma_int_t lddc1,
    magmaDoubleComplex* dC2, magma_int_t lddc2,
    magma_queue_t queue )
{
    // update 1 columnblock (columnwise zher2k) for itype=1 Lower case
    const double             d_one      = 1.0;
    const magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;

    magma_int_t kb = min( nk, nb );

    magma_zher2k( MagmaLower, MagmaNoTrans, kb, nb,
                  c_neg_one, dC1,    lddc1,
                             dC2,    lddc2,
                  d_one,     dA_col, ldda, queue );

    magma_zgemm( MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb,
                 c_neg_one, dC1 + kb,    lddc1,
                            dC2,         lddc2,
                 c_one,     dA_col + kb, ldda, queue );

    magma_zgemm( MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb,
                 c_neg_one, dC2 + kb,    lddc2,
                            dC1,         lddc1,
                 c_one,     dA_col + kb, ldda, queue );
}


/**
    Purpose
    -------
    ZHEGST_M reduces a complex Hermitian-definite generalized
    eigenproblem to standard form.
    
    If ITYPE = 1, the problem is A*x = lambda*B*x,
    and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
    
    If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
    B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.
    
    B must have been previously factorized as U**H*U or L*L**H by ZPOTRF.
    
    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    itype   INTEGER
            = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H);
            = 2 or 3: compute U*A*U**H or L**H*A*L.
    
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored and B is factored as U**H*U;
      -     = MagmaLower:  Lower triangle of A is stored and B is factored as L*L**H.
    
    @param[in]
    n       INTEGER
            The order of the matrices A and B.  N >= 0.
    
    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the transformed matrix, stored in the
            same format as A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
    
    @param[in,out]
    B       COMPLEX_16 array, dimension (LDB,N)
            The triangular factor from the Cholesky factorization of B,
            as returned by ZPOTRF.
            
            B is modified by the routine but restored on exit (in lapack zhegst/zhegs2).
    
    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    

    @ingroup magma_zheev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zhegst_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info)
{
    #define A(i_, j_) (A + (i_)*nb + (j_)*nb*lda)
    #define B(i_, j_) (B + (i_)*nb + (j_)*nb*ldb)
    
    #define dA(   igpu_, i_, j_) (dw[igpu_] + (i_)*nb + (j_)*nb*ldda)
    #define dB_c( igpu_, i_, j_) (dw[igpu_] + (i_)*nb + (j_)*nb*lddbc + dima*ldda)
    #define dB_r( igpu_, i_, j_) (dw[igpu_] + (i_)*nb + (j_)*nb*lddbr + dima*ldda)
    #define dwork(igpu_, i_, j_) (dw[igpu_] + (i_)*nb + (j_)*nb*nb    + dima*ldda + lddbc*lddbr)

    /* Constants */
    const magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_half     = MAGMA_Z_HALF;
    const magmaDoubleComplex c_neg_half = MAGMA_Z_NEG_HALF;
    const double             d_one      = 1.0;
    
    const char* uplo_ = lapack_uplo_const( uplo );
    magmaDoubleComplex_ptr dw[MagmaMaxGPUs];

    magma_queue_t queues[MagmaMaxGPUs][3];
    magma_event_t events[MagmaMaxGPUs][2];

    bool upper = (uplo == MagmaUpper);

    magma_int_t nb = magma_get_zhegst_nb_m( n );

    /* Test the input parameters. */
    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! upper && uplo != MagmaLower) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if (ldb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    magma_int_t nblock = magma_ceildiv( n, nb ); // number of blocks

    magma_int_t ldda = 0;
    magma_int_t dima = 0;

    if ( (itype == 1 && upper) || (itype != 1 && !upper) ) {
        ldda = magma_ceildiv( nblock, ngpu )*nb;
        dima = n;
    } else {
        ldda = n;
        dima = magma_ceildiv( nblock, ngpu )*nb;
    }
    magma_int_t lddbr = 2 * nb;
    magma_int_t lddbc = n;

    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
        magma_setdevice( igpu );

        if (MAGMA_SUCCESS != magma_zmalloc( &dw[igpu], (dima*ldda + lddbc*lddbr + n*nb) )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        magma_queue_create( igpu, &queues[igpu][0] );
        magma_queue_create( igpu, &queues[igpu][1] );
        magma_queue_create( igpu, &queues[igpu][2] );
        
        magma_event_create( &events[igpu][0] );
        magma_event_create( &events[igpu][1] );
    }

    /* Use hybrid blocked code */
    if (itype == 1) {
        if (upper) {
            /* Compute inv(U')*A*inv(U) */

            // copy A to mgpu
            for (magma_int_t k = 0; k < nblock; ++k) {
                magma_int_t igpu = k%ngpu;
                magma_setdevice( igpu );
                magma_int_t kb = min( nb, n-k*nb );
                magma_zsetmatrix_async( kb, n-k*nb,
                                        A(k, k),             lda,
                                        dA(igpu, k/ngpu, k), ldda, queues[igpu][0] );
            }

            for (magma_int_t k = 0; k < nblock; ++k) {
                magma_int_t ind_k  =   k   % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb = min( n-k*nb, nb );

                // Copy B panel
                for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                    magma_setdevice( igpu );
                    magma_queue_sync( queues[igpu][0] ); // sync previous B panel copy

                    // sync dwork copy and update (uses B panel of the next copy)
                    magma_queue_wait_event( queues[igpu][0], events[igpu][1] );
                    magma_queue_wait_event( queues[igpu][2], events[igpu][1] );
                    magma_zsetmatrix_async( kb, n-k*nb,
                                            B(k, k),              ldb,
                                            dB_r(igpu, ind_k, k), lddbr, queues[igpu][0] );
                }

                magma_int_t igpu_p = k%ngpu;

                if (k > 0) {
                    // Update the next panel
                    magma_setdevice( igpu_p );

                    magma_int_t nk = n - k*nb;

                    magma_zher2k( MagmaUpper, MagmaConjTrans, kb, nb,
                                  c_neg_one, dwork(igpu_p, 0, k),     nb,
                                             dB_r(igpu_p, ind_k1, k), lddbr,
                                  d_one,     dA(igpu_p, k/ngpu, k),   ldda, queues[igpu_p][2] );

                    // copy Akk block on the CPU
                    magma_zgetmatrix_async( kb, kb,
                                            dA(igpu_p, k/ngpu, k), ldda,
                                            A(k, k),               lda, queues[igpu_p][2] );

                    magma_event_record( events[igpu_p][0], queues[igpu_p][2]);

                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb,
                                 c_neg_one, dwork(igpu_p, 0, k),       nb,
                                            dB_r(igpu_p, ind_k1, k+1), lddbr,
                                 c_one,     dA(igpu_p, k/ngpu, k+1),   ldda, queues[igpu_p][2] );

                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb,
                                 c_neg_one, dB_r(igpu_p, ind_k1, k), lddbr,
                                            dwork(igpu_p, 0, k+1),   nb,
                                 c_one,     dA(igpu_p, k/ngpu, k+1), ldda, queues[igpu_p][2] );

                    // Update the panels of the other GPUs
                    for (magma_int_t j=k+1; j < nblock; ++j) {
                        magma_int_t igpu = j%ngpu;
                        if (igpu != igpu_p) {
                            magma_setdevice( igpu );
                            // zher2k on j-th row
                            magma_zhegst_m_1_U_row_update(
                                n-j*nb, nb,
                                dA(igpu, j/ngpu, j),   ldda,
                                dwork(igpu, 0, j),     nb,
                                dB_r(igpu, ind_k1, j), lddbr, queues[igpu][1] );
                        }
                    }
                }
                // compute next panel
                magma_setdevice( igpu_p );

                if (k+1 < nblock) {
                    magma_queue_sync( queues[igpu_p][0] ); // sync B panel copy
                    magma_ztrsm( MagmaLeft, uplo, MagmaConjTrans, MagmaNonUnit,
                                 kb, n-(k+1)*nb,
                                 c_one, dB_r(igpu_p, ind_k, k), lddbr,
                                        dA(igpu_p, k/ngpu, k+1), ldda, queues[igpu_p][2] );
                }

                magma_event_sync( events[igpu_p][0] ); // sync Akk copy
                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                if (k+1 < nblock) {
                    magma_zsetmatrix_async( kb, kb,
                                            A(k, k),               lda,
                                            dA(igpu_p, k/ngpu, k), ldda, queues[igpu_p][2] );

                    magma_zhemm( MagmaLeft, uplo, kb, n-(k+1)*nb,
                                 c_neg_half, dA(igpu_p, k/ngpu, k),    ldda,
                                             dB_r(igpu_p, ind_k, k+1), lddbr,
                                 c_one,      dA(igpu_p, k/ngpu, k+1),  ldda, queues[igpu_p][2] );

                    magma_zgetmatrix_async( kb, n-(k+1)*nb,
                                            dA(igpu_p, k/ngpu, k+1), ldda,
                                            A(k, k+1),               lda, queues[igpu_p][2] );
                }

                if (k > 0) {
                    // Update the remaining panels of GPU igpu_p
                    for (magma_int_t j=k+ngpu; j < nblock; j += ngpu) {
                        magma_setdevice( igpu_p );
                        // zher2k on j-th row
                        magma_zhegst_m_1_U_row_update(
                            n-j*nb, nb,
                            dA(igpu_p, j/ngpu, j),   ldda,
                            dwork(igpu_p, 0, j),     nb,
                            dB_r(igpu_p, ind_k1, j), lddbr, queues[igpu_p][1] );
                    }
                }

                if (k+1 < nblock) {
                    // send the partially updated panel of dA to each gpu in dwork block
                    magma_setdevice( igpu_p );
                    magma_queue_sync( queues[igpu_p][2] );

                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );
                        magma_zsetmatrix_async( kb, n-(k+1)*nb,
                                                A(k, k+1),           lda,
                                                dwork(igpu, 0, k+1), nb, queues[igpu][1] );

                        magma_event_record( events[igpu][1], queues[igpu][1]);
                    }

                    magma_setdevice( igpu_p );

                    magma_zhemm( MagmaLeft, uplo, kb, n-(k+1)*nb,
                                 c_neg_half, dA(igpu_p, k/ngpu, k),    ldda,
                                             dB_r(igpu_p, ind_k, k+1), lddbr,
                                 c_one,      dA(igpu_p, k/ngpu, k+1),  ldda, queues[igpu_p][1] );
                }
            }

            for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                magma_queue_sync( queues[igpu][1] );
            }

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 1; j < nblock; ++j) {
                    nloc[(j-1)%ngpu] += nb;

                    magma_int_t jb = min( nb, n-j*nb );

                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );
                        if (nloc[igpu] > 0) {
                            magma_zsetmatrix_async( jb, n-j*nb,
                                                    B(j, j),            ldb,
                                                    dB_r(igpu, j%2, j), lddbr, queues[igpu][j%2] );

                            magma_queue_wait_event( queues[igpu][j%2], events[igpu][0] );

                            magma_ztrsm( MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit, nloc[igpu], jb, c_one, dB_r(igpu, j%2, j), lddbr,
                                        dA(igpu, 0, j), ldda, queues[igpu][j%2] );

                            if ( j < nblock-1 ) {
                                magma_zgemm( MagmaNoTrans, MagmaNoTrans, nloc[igpu], n-(j+1)*nb, nb,
                                             c_neg_one, dA(igpu, 0, j),       ldda,
                                                        dB_r(igpu, j%2, j+1), lddbr,
                                             c_one,     dA(igpu, 0, j+1),     ldda, queues[igpu][j%2] );
                            }
                            magma_event_record( events[igpu][0], queues[igpu][j%2]);
                        }
                    }

                    for (magma_int_t k = 0; k < j; ++k) {
                        magma_int_t igpu = k%ngpu;
                        magma_setdevice( igpu );
                        magma_int_t kb = min( nb, n-k*nb );
                        magma_zgetmatrix_async( kb, jb,
                                                dA(igpu, k/ngpu, j), ldda,
                                                A(k, j),             lda, queues[igpu][j%2] );
                    }
                }
            }
        } else {
            /* Compute inv(L)*A*inv(L') */

            // Copy A to mgpu
            for (magma_int_t k = 0; k < nblock; ++k) {
                magma_int_t igpu = k%ngpu;
                magma_setdevice( igpu );
                magma_int_t kb = min( nb, n-k*nb );
                magma_zsetmatrix_async( (n-k*nb), kb,
                                        A(k, k),             lda,
                                        dA(igpu, k, k/ngpu), ldda, queues[igpu][0] );
            }

            for (magma_int_t k = 0; k < nblock; ++k) {
                magma_int_t ind_k  =   k   % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min( n-k*nb, nb );

                // Copy B panel
                for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                    magma_setdevice( igpu );
                    magma_queue_sync( queues[igpu][0] ); // sync previous B panel copy

                    // sync dwork copy and update (uses B panel of the next copy)
                    magma_queue_wait_event( queues[igpu][0], events[igpu][1] );
                    magma_queue_wait_event( queues[igpu][2], events[igpu][1] );
                    magma_zsetmatrix_async( (n-k*nb), kb,
                                            B(k, k),              ldb,
                                            dB_c(igpu, k, ind_k), lddbc, queues[igpu][0] );
                }

                magma_int_t igpu_p = k%ngpu;

                if (k > 0) {
                    // Update the next panel
                    magma_setdevice( igpu_p );

                    magma_int_t nk = n-k*nb;

                    magma_zher2k( MagmaLower, MagmaNoTrans, kb, nb,
                                  c_neg_one, dwork(igpu_p, k, 0),     n,
                                             dB_c(igpu_p, k, ind_k1), lddbc,
                                  d_one,     dA(igpu_p, k, k/ngpu),   ldda, queues[igpu_p][2] );

                    // copy Akk block on the CPU
                    magma_zgetmatrix_async( kb, kb,
                                            dA(igpu_p, k, k/ngpu), ldda,
                                            A(k, k),               lda, queues[igpu_p][2] );

                    magma_event_record( events[igpu_p][0], queues[igpu_p][2]);

                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb,
                                 c_neg_one, dwork(igpu_p, k+1, 0),   n,
                                            dB_c(igpu_p, k, ind_k1), lddbc,
                                 c_one,     dA(igpu_p, k+1, k/ngpu), ldda, queues[igpu_p][2] );

                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb,
                                 c_neg_one, dB_c(igpu_p, k+1, ind_k1), lddbc,
                                            dwork(igpu_p, k, 0),       n,
                                 c_one,     dA(igpu_p, k+1, k/ngpu),   ldda, queues[igpu_p][2] );

                    // Update the panels of the other GPUs
                    for (magma_int_t j=k+1; j < nblock; ++j) {
                        magma_int_t igpu = j%ngpu;
                        if (igpu != igpu_p) {
                            magma_setdevice( igpu );
                            // zher2k on j-th column
                            magma_zhegst_m_1_L_col_update(
                                n-j*nb, nb,
                                dA(igpu, j, j/ngpu),   ldda,
                                dwork(igpu, j, 0),     n,
                                dB_c(igpu, j, ind_k1), lddbc, queues[igpu][1] );
                        }
                    }
                }
                // compute next panel
                magma_setdevice( igpu_p );

                if (k+1 < nblock) {
                    magma_queue_sync( queues[igpu_p][0] ); // sync B panel copy

                    magma_ztrsm( MagmaRight, uplo, MagmaConjTrans, MagmaNonUnit,
                                 n-(k+1)*nb, kb,
                                 c_one, dB_c(igpu_p, k, ind_k), lddbc,
                                 dA(igpu_p, k+1, k/ngpu), ldda, queues[igpu_p][2] );
                }

                magma_event_sync( events[igpu_p][0] ); // sync Akk copy
                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                if (k+1 < nblock) {
                    magma_zsetmatrix_async( kb, kb,
                                            A(k, k),               lda,
                                            dA(igpu_p, k, k/ngpu), ldda, queues[igpu_p][2] );

                    magma_zhemm( MagmaRight, uplo, n-(k+1)*nb, kb,
                                 c_neg_half, dA(igpu_p, k, k/ngpu),    ldda,
                                             dB_c(igpu_p, k+1, ind_k), lddbc,
                                 c_one,      dA(igpu_p, k+1, k/ngpu),  ldda, queues[igpu_p][2] );

                    magma_zgetmatrix_async( n-(k+1)*nb, kb,
                                            dA(igpu_p, k+1, k/ngpu), ldda,
                                            A(k+1, k),               lda, queues[igpu_p][2] );
                }

                if (k > 0) {
                    // Update the remaining panels of GPU igpu_p
                    for (magma_int_t j=k+ngpu; j < nblock; j += ngpu) {
                        magma_setdevice( igpu_p );
                        // zher2k on j-th column
                        magma_zhegst_m_1_L_col_update(
                            n-j*nb, nb,
                            dA(igpu_p, j, j/ngpu),   ldda,
                            dwork(igpu_p, j, 0),     n,
                            dB_c(igpu_p, j, ind_k1), lddbc, queues[igpu_p][1] );
                    }
                }

                if (k+1 < nblock) {
                    // send the partially updated panel of dA to each gpu in dwork block
                    magma_setdevice( igpu_p );
                    magma_queue_sync( queues[igpu_p][2] );

                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );
                        magma_zsetmatrix_async( (n-(k+1)*nb), kb,
                                                A(k+1, k),           lda,
                                                dwork(igpu, k+1, 0), n, queues[igpu][1] );

                        magma_event_record( events[igpu][1], queues[igpu][1]);
                    }

                    magma_setdevice( igpu_p );

                    magma_zhemm( MagmaRight, uplo, n-(k+1)*nb, kb,
                                 c_neg_half, dA(igpu_p, k, k/ngpu),    ldda,
                                             dB_c(igpu_p, k+1, ind_k), lddbc,
                                 c_one,      dA(igpu_p, k+1, k/ngpu),  ldda, queues[igpu_p][1] );
                }
            }

            for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                magma_queue_sync( queues[igpu][1] );
            }

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 1; j < nblock; ++j) {
                    nloc[(j-1)%ngpu] += nb;

                    magma_int_t jb = min( nb, n-j*nb );

                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );
                        if (nloc[igpu] > 0) {
                            magma_zsetmatrix_async( (n-j*nb), jb,
                                                    B(j, j),            ldb,
                                                    dB_c(igpu, j, j%2), lddbc, queues[igpu][j%2] );

                            magma_queue_wait_event( queues[igpu][j%2], events[igpu][0] );

                            magma_ztrsm( MagmaLeft, uplo, MagmaNoTrans, MagmaNonUnit, jb, nloc[igpu], c_one, dB_c(igpu, j, j%2), lddbc,
                                        dA(igpu, j, 0), ldda, queues[igpu][j%2] );

                            if ( j < nblock-1 ) {
                                magma_zgemm( MagmaNoTrans, MagmaNoTrans, n-(j+1)*nb, nloc[igpu], nb,
                                             c_neg_one, dB_c(igpu, j+1, j%2), lddbc,
                                                        dA(igpu, j, 0),       ldda,
                                             c_one,     dA(igpu, j+1, 0),     ldda, queues[igpu][j%2] );
                            }
                            magma_event_record( events[igpu][0], queues[igpu][j%2]);
                        }
                    }

                    for (magma_int_t k = 0; k < j; ++k) {
                        magma_int_t igpu = k%ngpu;
                        magma_setdevice( igpu );
                        magma_int_t kb = min( nb, n-k*nb );
                        magma_zgetmatrix_async( jb, kb,
                                                dA(igpu, j, k/ngpu), ldda,
                                                A(j, k),             lda, queues[igpu][j%2] );
                    }
                }
            }
        }
    } else { // itype == 2 or 3
        if (upper) {
            /* Compute U*A*U' */
            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };
                magma_int_t iloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 0; j < nblock; ++j) {
                    magma_int_t jb = min( nb, n-j*nb );
                    nloc[j%ngpu] += jb;
                }

                for (magma_int_t k = 0; k < nblock; ++k) {
                    magma_int_t kb = min( nb, n-k*nb );

                    for (magma_int_t j = k; j < nblock; ++j) {
                        magma_int_t igpu = j%ngpu;
                        magma_setdevice( igpu );
                        magma_int_t jb = min( nb, n-j*nb );
                        magma_zsetmatrix_async( kb, jb,
                                                A(k, j),             lda,
                                                dA(igpu, k, j/ngpu), ldda, queues[igpu][k%2] );
                    }

                    magma_int_t igpu_p = k % ngpu;

                    ++iloc[igpu_p];
                    nloc[igpu_p] -= kb;

                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );
                        magma_zsetmatrix_async( k*nb + kb, kb,
                                                B(0, k),            ldb,
                                                dB_c(igpu, 0, k%2), lddbc, queues[igpu][k%2] );

                        magma_queue_wait_event( queues[igpu][k%2], events[igpu][0] );


                        if (igpu == igpu_p) {
                            magma_zhemm( MagmaRight, uplo, k*nb, kb,
                                         c_half, dA(igpu, k, k/ngpu), ldda,
                                                 dB_c(igpu, 0, k%2),  lddbc,
                                         c_one,  dA(igpu, 0, k/ngpu), ldda, queues[igpu][k%2] );
                        }

                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, k*nb, nloc[igpu], kb,
                                     c_one, dB_c(igpu, 0, k%2),      lddbc,
                                            dA(igpu, k, iloc[igpu]), ldda,
                                     c_one, dA(igpu, 0, iloc[igpu]), ldda, queues[igpu][k%2] );

                        magma_ztrmm( MagmaLeft, uplo, MagmaNoTrans, MagmaNonUnit, kb, nloc[igpu], c_one, dB_c(igpu, k, k%2), lddbc,
                                    dA(igpu, k, iloc[igpu]), ldda, queues[igpu][k%2] );

                        magma_event_record( events[igpu][0], queues[igpu][k%2]);

                        if (igpu == igpu_p) {
                            magma_zgetmatrix_async( k*nb, kb,
                                                    dA(igpu, 0, k/ngpu), ldda,
                                                    A(0, k),             lda, queues[igpu][k%2] );
                        }
                    }
                }
            }

            for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                magma_queue_sync( queues[igpu][0] );
                magma_queue_sync( queues[igpu][1] );
            }

            for (magma_int_t k = 0; k < nblock; ++k) {
                magma_int_t ind_k = k % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min( n-k*nb, nb );

                magma_int_t igpu_p = k%ngpu;

                if (k > 0) {
                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );

                        magma_queue_wait_event( queues[igpu][0], events[igpu][ind_k] ); // sync computation that use the B panel of next copy

                        magma_zsetmatrix_async( k*nb+kb, kb,
                                                B(0, k),              ldb,
                                                dB_c(igpu, 0, ind_k), lddbc, queues[igpu][0] );

                        magma_event_record( events[igpu][ind_k], queues[igpu][0] );

                        magma_zsetmatrix_async( k*nb, kb,
                                                A(0, k),           lda,
                                                dwork(igpu, 0, 0), n, queues[igpu][1] );

                        magma_queue_wait_event( queues[igpu][1], events[igpu][ind_k] ); // sync B copy
                    }

                    magma_setdevice( igpu_p );

                    magma_queue_wait_event( queues[igpu_p][2], events[igpu_p][ind_k1] ); // sync update of previous step
                    magma_queue_wait_event( queues[igpu_p][2], events[igpu_p][ind_k] ); // sync B copy

                    magma_zhemm( MagmaRight, uplo, k*nb, kb,
                                 c_half, dA(igpu_p, k, k/ngpu),  ldda,
                                         dB_c(igpu_p, 0, ind_k), lddbc,
                                 c_one,  dA(igpu_p, 0, k/ngpu),  ldda, queues[igpu_p][2] );

                    magma_ztrmm( MagmaRight, uplo, MagmaConjTrans, MagmaNonUnit,
                                k*nb, kb,
                                c_one, dB_c(igpu_p, k, ind_k), lddbc,
                                dA(igpu_p, 0, k/ngpu), ldda, queues[igpu_p][2] );

                    magma_event_record( events[igpu_p][ind_k], queues[igpu_p][2] );
                    magma_queue_wait_event( queues[igpu_p][1], events[igpu_p][ind_k] );

                    for (magma_int_t j = 0; j < k; ++j) {
                        magma_int_t igpu = j%ngpu;
                        magma_setdevice( igpu );

                        magma_zher2k( uplo, MagmaNoTrans, nb, kb,
                                      c_one, dwork(igpu, j, 0),    n,
                                             dB_c(igpu, j, ind_k), lddbc,
                                      d_one, dA(igpu, j, j/ngpu),  ldda, queues[igpu][1] );

                        magma_zgemm( MagmaNoTrans, MagmaConjTrans, j*nb, nb, kb,
                                     c_one, dB_c(igpu, 0, ind_k), lddbc,
                                            dwork(igpu, j, 0),    n,
                                     c_one, dA(igpu, 0, j/ngpu),  ldda, queues[igpu][1] );

                        magma_zgemm( MagmaNoTrans, MagmaConjTrans, j*nb, nb, kb,
                                     c_one, dwork(igpu, 0, 0),    n,
                                            dB_c(igpu, j, ind_k), lddbc,
                                     c_one, dA(igpu, 0, j/ngpu),  ldda, queues[igpu][1] );
                    }
                }

                for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                    magma_setdevice( igpu );
                    magma_event_record( events[igpu][ind_k], queues[igpu][1] );
                    magma_queue_sync( queues[igpu][0] ); // sync B copy (conflicts with zhegst)
                }

                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                magma_setdevice( igpu_p );

                magma_zsetmatrix_async( kb, kb,
                                        A(k, k),               lda,
                                        dA(igpu_p, k, k/ngpu), ldda, queues[igpu_p][1] );
            }

            for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                magma_queue_sync( queues[igpu][1] );
            }

            //copy A from mgpus
            for (magma_int_t j = 0; j < nblock; ++j) {
                magma_int_t igpu = j%ngpu;
                magma_setdevice( igpu );
                magma_int_t jb = min( nb, n-j*nb );
                magma_zgetmatrix_async( j*nb+jb, jb,
                                        dA(igpu, 0, j/ngpu), ldda,
                                        A(0, j),             lda, queues[igpu][0] );
            }
        } else {
            /* Compute L'*A*L */

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };
                magma_int_t iloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 0; j < nblock; ++j) {
                    magma_int_t jb = min( nb, n-j*nb );
                    nloc[j%ngpu] += jb;
                }

                for (magma_int_t k = 0; k < nblock; ++k) {
                    magma_int_t kb = min( nb, n-k*nb );

                    for (magma_int_t j = k; j < nblock; ++j) {
                        magma_int_t igpu = j%ngpu;
                        magma_setdevice( igpu );
                        magma_int_t jb = min( nb, n-j*nb );

                        magma_zsetmatrix_async( jb, kb,
                                                A(j, k),             lda,
                                                dA(igpu, j/ngpu, k), ldda, queues[igpu][k%2] );
                    }

                    magma_int_t igpu_p = k % ngpu;

                    ++iloc[igpu_p];
                    nloc[igpu_p] -= kb;

                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );
                        magma_zsetmatrix_async( kb, k*nb +kb,
                                                B(k, 0),            ldb,
                                                dB_r(igpu, k%2, 0), lddbr, queues[igpu][k%2] );

                        magma_queue_wait_event( queues[igpu][k%2], events[igpu][0] );


                        if (igpu == igpu_p) {
                            magma_zhemm( MagmaLeft, uplo, kb, k*nb,
                                         c_half, dA(igpu, k/ngpu, k), ldda,
                                                 dB_r(igpu, k%2, 0),  lddbr,
                                         c_one,  dA(igpu, k/ngpu, 0), ldda, queues[igpu][k%2] );
                        }

                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, nloc[igpu], k*nb, kb,
                                     c_one, dA(igpu, iloc[igpu], k), ldda,
                                            dB_r(igpu, k%2, 0),      lddbr,
                                     c_one, dA(igpu, iloc[igpu], 0), ldda, queues[igpu][k%2] );

                        magma_ztrmm( MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit, nloc[igpu], kb, c_one, dB_r(igpu, k%2, k), lddbr,
                                    dA(igpu, iloc[igpu], k), ldda, queues[igpu][k%2] );

                        magma_event_record( events[igpu][0], queues[igpu][k%2]);

                        if (igpu == igpu_p) {
                            magma_zgetmatrix_async( kb, k*nb,
                                                    dA(igpu, k/ngpu, 0), ldda,
                                                    A(k, 0),             lda, queues[igpu][k%2] );
                        }
                    }
                }
            }

            for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                magma_queue_sync( queues[igpu][0] );
                magma_queue_sync( queues[igpu][1] );
            }

            for (magma_int_t k = 0; k < nblock; ++k) {
                magma_int_t ind_k = k % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min( n-k*nb, nb );

                magma_int_t igpu_p = k%ngpu;

                if (k > 0) {
                    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                        magma_setdevice( igpu );

                        magma_queue_wait_event( queues[igpu][0], events[igpu][ind_k] ); // sync computation that use the B panel of next copy

                        magma_zsetmatrix_async( kb, k*nb+kb,
                                                B(k, 0),              ldb,
                                                dB_r(igpu, ind_k, 0), lddbr, queues[igpu][0] );

                        magma_event_record( events[igpu][ind_k], queues[igpu][0] );

                        magma_zsetmatrix_async( kb, k*nb,
                                                A(k, 0),           lda,
                                                dwork(igpu, 0, 0), nb, queues[igpu][1] );

                        magma_queue_wait_event( queues[igpu][1], events[igpu][ind_k] ); // sync B copy
                    }

                    magma_setdevice( igpu_p );

                    magma_queue_wait_event( queues[igpu_p][2], events[igpu_p][ind_k1] ); // sync update of previous step
                    magma_queue_wait_event( queues[igpu_p][2], events[igpu_p][ind_k] ); // sync B copy

                    magma_zhemm( MagmaLeft, uplo, kb, k*nb,
                                 c_half, dA(igpu_p, k/ngpu, k),  ldda,
                                         dB_r(igpu_p, ind_k, 0), lddbr,
                                 c_one,  dA(igpu_p, k/ngpu, 0),  ldda, queues[igpu_p][2] );

                    magma_ztrmm( MagmaLeft, uplo, MagmaConjTrans, MagmaNonUnit,
                                kb, k*nb,
                                c_one, dB_r(igpu_p, ind_k, k), lddbr,
                                dA(igpu_p, k/ngpu, 0), ldda, queues[igpu_p][2] );

                    magma_event_record( events[igpu_p][ind_k], queues[igpu_p][2] );
                    magma_queue_wait_event( queues[igpu_p][1], events[igpu_p][ind_k] );

                    for (magma_int_t j = 0; j < k; ++j) {
                        magma_int_t igpu = j%ngpu;
                        magma_setdevice( igpu );

                        magma_zher2k( uplo, MagmaConjTrans, nb, kb,
                                      c_one, dwork(igpu, 0, j),    nb,
                                             dB_r(igpu, ind_k, j), lddbr,
                                      d_one, dA(igpu, j/ngpu, j),  ldda, queues[igpu][1] );

                        magma_zgemm( MagmaConjTrans, MagmaNoTrans, nb, j*nb, kb,
                                     c_one, dwork(igpu, 0, j),    nb,
                                            dB_r(igpu, ind_k, 0), lddbr,
                                     c_one, dA(igpu, j/ngpu, 0),  ldda, queues[igpu][1] );

                        magma_zgemm( MagmaConjTrans, MagmaNoTrans, nb, j*nb, kb,
                                     c_one, dB_r(igpu, ind_k, j), lddbr,
                                            dwork(igpu, 0, 0),    nb,
                                     c_one, dA(igpu, j/ngpu, 0),  ldda, queues[igpu][1] );
                    }
                }

                for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                    magma_setdevice( igpu );
                    magma_event_record( events[igpu][ind_k], queues[igpu][1] );
                    magma_queue_sync( queues[igpu][0] ); // sync B copy (conflicts with zhegst)
                }

                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                magma_setdevice( igpu_p );

                magma_zsetmatrix_async( kb, kb,
                                        A(k, k),               lda,
                                        dA(igpu_p, k/ngpu, k), ldda, queues[igpu_p][1] );
            }

            for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
                magma_queue_sync( queues[igpu][1] );
            }

            //copy A from mgpus
            for (magma_int_t j = 0; j < nblock; ++j) {
                magma_int_t igpu = j%ngpu;
                magma_setdevice( igpu );
                magma_int_t jb = min( nb, n-j*nb );
                magma_zgetmatrix_async( jb, j*nb+jb,
                                        dA(igpu, j/ngpu, 0), ldda,
                                        A(j, 0),             lda, queues[igpu][0] );
            }
        }
    }

    for (magma_int_t igpu = 0; igpu < ngpu; ++igpu) {
        magma_setdevice( igpu );
        magma_event_destroy( events[igpu][0] );
        magma_event_destroy( events[igpu][1] );
        
        magma_queue_destroy( queues[igpu][0] );
        magma_queue_destroy( queues[igpu][1] );
        magma_queue_destroy( queues[igpu][2] );
        magma_free( dw[igpu] );
    }
    magma_setdevice( orig_dev );

    return *info;
} /* magma_zhegst_gpu */
