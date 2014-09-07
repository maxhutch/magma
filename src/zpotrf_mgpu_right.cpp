/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "trace.h"

#define get_time magma_wtime

extern "C" void
magma_zherk_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex **db, magma_int_t lddb, magma_int_t offset_b,
    double beta,
    magmaDoubleComplex **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_streams, magma_queue_t stream[][10]);

extern "C" void
magma_zherk_mgpu2(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex **db, magma_int_t lddb, magma_int_t offset_b,
    double beta,
    magmaDoubleComplex **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_streams, magma_queue_t stream[][10]);


/**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
       dA = U**H * U,   if UPLO = MagmaUpper, or
       dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    d_lA    COMPLEX_16 array of pointers on the GPU, dimension (num_gpus)
            On entry, the Hermitian matrix dA distributed over GPUs
            (dl_A[d] points to the local matrix on the d-th GPU).  
            It is distributed in 1D block column or row cyclic (with the
            block size of nb) if UPLO = MagmaUpper or MagmaLower, respectively.
            If UPLO = MagmaUpper, the leading N-by-N upper triangular 
            part of dA contains the upper triangular part of the matrix dA, 
            and the strictly lower triangular part of dA is not referenced.  
            If UPLO = MagmaLower, the leading N-by-N lower triangular part 
            of dA contains the lower triangular part of the matrix dA, and 
            the strictly upper triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_zposv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zpotrf_mgpu_right(magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t n,
                        magmaDoubleComplex **d_lA, magma_int_t ldda, magma_int_t *info )
{
    #define dlA(id, i, j)  (d_lA[(id)] + (j) * ldda + (i))
    #define dlP(id, i, j)  (d_lP[(id)] + (j) * ldda + (i))

    #define panel(j)  (panel + (j))
    #define tmppanel(j)  (tmppanel + (j))
    #define tmpprevpanel(j)  (tmpprevpanel + (j))
    #define STREAM_ID(i) (num_streams > 1 ? 1+((i)/nb)%(num_streams-1) : 0)

    magmaDoubleComplex z_one = MAGMA_Z_MAKE(  1.0, 0.0 );
    magmaDoubleComplex mz_one = MAGMA_Z_MAKE( -1.0, 0.0 );
    double             one =  1.0;
    double             m_one = -1.0;
    const char* uplo_ = lapack_uplo_const( uplo );

    magma_int_t j, nb, d, id, j_local, blkid, crosspoint, prevj, prevtrsmrows, num_streams = 5;
    magmaDoubleComplex *panel, *tmppanel0, *tmppanel1, *tmppanel, *tmpprevpanel;
    magmaDoubleComplex *d_lP[MagmaMaxGPUs], *dlpanel, *dlpanels[MagmaMaxGPUs];
    magma_int_t rows, trsmrows, ngpu, n_local[MagmaMaxGPUs], ldpanel;
    magma_queue_t stream[MagmaMaxGPUs][10];

    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );

    nb = magma_get_zpotrf_nb(n);

    ldpanel = ldda;
    magma_setdevice(0);
    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &panel, 2 * nb * ldpanel )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    tmppanel0 = panel;
    tmppanel1 = tmppanel0 + nb * ldpanel;

    if ((nb <= 1) || (nb >= n)) {
        // Use unblocked code.
        magma_zgetmatrix( n, n, dlA(0, 0, 0), ldda, panel, ldpanel);
        lapackf77_zpotrf( uplo_, &n, panel, &ldpanel, info);
        magma_zsetmatrix( n, n, panel, ldpanel, dlA(0, 0, 0), ldda );
    } else {
        for( d = 0; d < num_gpus; d++ ) {
            // local-n and local-ld
            n_local[d] = ((n / nb) / num_gpus) * nb;
            if (d < (n / nb) % num_gpus)
                n_local[d] += nb;
            else if (d == (n / nb) % num_gpus)
                n_local[d] += n % nb;

            magma_setdevice(d);
            magma_device_sync();
            if (MAGMA_SUCCESS != magma_zmalloc( &d_lP[d], nb * ldda )) {
                for( j = 0; j < d; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_lP[d] );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            for( j=0; j < num_streams; j++ ) {
                magma_queue_create( &stream[d][j] );
            }
        }

        //#define ENABLE_TIMER
        #if defined (ENABLE_TIMER)
        real_Double_t therk[4], tmtc, tcchol, tctrsm, tctm, tmnp, tcnp;
        real_Double_t ttot_herk[4] = {0,0,0,0}, ttot_mtc = 0, ttot_cchol = 0, ttot_ctrsm = 0, ttot_ctm = 0, ttot_mnp = 0, ttot_cnp = 0;
        printf("\n\n %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                "j", "nb", "row", "mtc", "CPU_np", "panel", "ctrsm", "CH+TRSM", "CPU", "dsyrk[0]", "dsyrk[1]", "dsyrk[2]", "dsyrk[3]", "ctm P", "gpu_np");
        printf("     ====================================================================================================\n");
        #endif

        // Use blocked code.
        if (uplo == MagmaUpper) {
            printf( " === not supported, yet ===\n" );
        } else {
            blkid = -1;
            if (num_gpus == 4)
                crosspoint = n;
            else if (num_gpus == 3)
                crosspoint = n;
            else if (num_gpus == 2)
                crosspoint = 20160;
            else
                crosspoint = 0;
            crosspoint = 0; //n; //n -- > gpu always does next panel, 0 --> cpu always does next panel
            crosspoint = n;

            #if defined (ENABLE_TIMER)
            real_Double_t tget = get_time(), tset = 0.0, ttot = 0.0;
            #endif
            if ( n > nb ) {
                // send first panel to cpu
                magma_setdevice(0);
                tmppanel = tmppanel0;
                magma_zgetmatrix_async(n, nb,
                        dlA(0, 0, 0), ldda,
                        tmppanel(0),  ldpanel,
                        stream[0][0] );
            }
            #if defined (ENABLE_TIMER)
            for( d=0; d < num_gpus; d++ ) {
                magma_setdevice(d);
                magma_device_sync();
            }
            tget = get_time()-tget;
            #endif

            // Compute the Cholesky factorization A = L*L'
            for (j = 0; (j + nb) < n; j += nb) {
                #if defined (ENABLE_TIMER)
                therk[0] = therk[1] = therk[2] = therk[3] = tmtc = tcchol = tctrsm = tctm = tmnp = tcnp = 0.0;
                #endif

                blkid += 1;
                tmppanel = (blkid % 2 == 0) ? tmppanel0 : tmppanel1;
                // Set the gpu number that holds the current panel
                id = (j / nb) % num_gpus;
                magma_setdevice(id);

                // Set the local index where the current panel is
                j_local = j / (nb * num_gpus) * nb;
                
                rows = n - j;
                // Wait for the panel on cpu
                magma_queue_sync( stream[id][0] );
                if (j > 0 && prevtrsmrows > crosspoint) {
                    #if defined (ENABLE_TIMER)
                    tcnp = get_time();
                    #endif

                    tmpprevpanel = ((blkid - 1) % 2) == 0 ? tmppanel0 : tmppanel1;

                    blasf77_zgemm( MagmaNoTransStr, MagmaConjTransStr,
                            &rows, &nb, &nb,
                            &mz_one, tmpprevpanel(j), &ldpanel,
                                     tmpprevpanel(j), &ldpanel,
                            &z_one,      tmppanel(j), &ldpanel );

                    #if defined (ENABLE_TIMER)
                    tcnp = get_time() - tcnp;
                    ttot_cnp += tcnp;
                    #endif
                }

                #if defined (ENABLE_TIMER)
                tcchol = get_time();
                #endif
                lapackf77_zpotrf(MagmaLowerStr, &nb, tmppanel(j), &ldpanel, info);
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }

                #if defined (ENABLE_TIMER)
                tcchol = get_time() - tcchol;
                ttot_cchol += tcchol;
                tctrsm = get_time();
                #endif

                trsmrows = rows - nb;

                if (trsmrows > 0) {
                    blasf77_ztrsm(MagmaRightStr, MagmaLowerStr, MagmaConjTransStr, MagmaNonUnitStr,
                                  &trsmrows, &nb,
                                  &z_one, tmppanel(j), &ldpanel,
                                          tmppanel(j + nb), &ldpanel);
                }

                #if defined (ENABLE_TIMER)
                tctrsm = get_time() - tctrsm;
                ttot_ctrsm += tctrsm;
                tctm = get_time();
                #endif

                d = (id + 1) % num_gpus;
                // send current panel to gpus
                for (ngpu = 0; ngpu < num_gpus; ngpu++, d = (d + 1) % num_gpus ) {
                    magma_int_t myrows = 0;
                    magma_int_t row_offset = 0;
                    if ( d == id ) {
                        dlpanel = dlA(d, j, j_local);
                        myrows = rows;
                        row_offset = 0;
                    } else {
                        dlpanel = dlP(d, 0, 0);
                        myrows = trsmrows;
                        row_offset = nb;
                    }

                    if (myrows > 0) {
                        magma_setdevice(d);
                        magma_zsetmatrix_async(myrows, nb,
                                tmppanel(j + row_offset),    ldpanel,
                                dlpanel, ldda, stream[d][0] );
                    }
                }
                /* make sure panel is on GPUs */
                d = (id + 1) % num_gpus;
                for (ngpu = 0; ngpu < num_gpus; ngpu++, d = (d + 1) % num_gpus ) {
                    magma_setdevice(d);
                    magma_queue_sync( stream[d][0] );
                }

                #if defined (ENABLE_TIMER)
                tctm = get_time() - tctm;
                ttot_ctm += tctm;
                #endif

                if ( (j + nb) < n) {
                    magma_int_t offset = 0;
                    magma_int_t row_offset = 0;
                    if (j + nb + nb < n) {
                        d = (id + 1) % num_gpus;
                        magma_setdevice(d);
                        magma_int_t j_local2 = (j + nb) / (nb * num_gpus) * nb;
                        if (trsmrows <= crosspoint) {
                            #if defined (ENABLE_TIMER)
                            tmnp = get_time();
                            #endif

                            // do gemm on look ahead panel
                            if ( d == id ) {
                                dlpanel = dlA(d, j + nb, j_local);
                            } else {
                                dlpanel = dlP(d, 0, 0);
                            }

                            magmablasSetKernelStream(stream[d][STREAM_ID(j_local2)]);
                            #define ZHERK_ON_DIAG
                            #ifdef  ZHERK_ON_DIAG
                            magma_zherk( MagmaLower, MagmaNoTrans,
                                    nb, nb,
                                    m_one, dlpanel, ldda,
                                     one,  dlA(d, j + nb, j_local2), ldda);
                            magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                    trsmrows-nb, nb, nb,
                                    mz_one, dlpanel+nb, ldda,
                                            dlpanel,    ldda,
                                     z_one, dlA(d, j + nb +nb, j_local2), ldda);
                            #else
                            magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                    trsmrows, nb, nb,
                                    mz_one, dlpanel, ldda,
                                            dlpanel, ldda,
                                     z_one, dlA(d, j + nb, j_local2), ldda);
                            #endif

                            #if defined (ENABLE_TIMER)
                            magma_device_sync();
                            tmnp = get_time() - tmnp;
                            ttot_mnp += tmnp;
                            #endif
                        }
                        // send next panel to cpu
                        magma_queue_sync( stream[d][STREAM_ID(j_local2)] ); // make sure lookahead is done
                        tmppanel = ((blkid+1) % 2 == 0) ? tmppanel0 : tmppanel1;
                        magma_zgetmatrix_async(rows-nb, nb,
                                dlA(d, j+nb, j_local2), ldda,
                                tmppanel(j+nb),  ldpanel,
                                stream[d][0] );
                        tmppanel = (blkid % 2 == 0) ? tmppanel0 : tmppanel1;

                        offset = j + nb + nb;
                        row_offset = nb;
                    } else {
                        offset = j + nb;
                        row_offset = 0;
                    }

                    if (n - offset > 0) {
                        // syrk on multiple gpu
                        for (d = 0; d < num_gpus; d++ ) {
                            if ( d == id ) {
                                dlpanels[d] = dlA(d, j + nb + row_offset, j_local);
                            } else {
                                dlpanels[d] = dlP(d, row_offset, 0);
                            }
                        }

                        #if defined (ENABLE_TIMER)
                        for( d=0; d < num_gpus; d++ ) therk[d] = get_time();
                        #endif

                        //magmablasSetKernelStream(stream[d]);
                        //magma_zherk(MagmaLower, MagmaNoTrans, n - offset, nb,
                        //        m_one, dlpanel, ldda,
                        //        one, &d_lA[d][offset + offset*ldda], ldda );
                        #ifdef  ZHERK_ON_DIAG
                        magma_zherk_mgpu
                        #else
                        magma_zherk_mgpu2
                        #endif
                                        (num_gpus, MagmaLower, MagmaNoTrans,
                                         nb, n - offset, nb,
                                         m_one, dlpanels, ldda, 0,
                                         one,   d_lA,     ldda, offset,
                                         num_streams, stream );
                        #if defined (ENABLE_TIMER)
                        for( d=0; d < num_gpus; d++ ) {
                            magma_setdevice(d);
                            magma_device_sync();
                            therk[d] = get_time() - therk[d];
                            ttot_herk[d] += therk[d];
                        }
                        #endif
                    }

                    prevtrsmrows = trsmrows;
                    prevj = j;

                    #if defined (ENABLE_TIMER)
                    ttot += (tcnp+tcchol+tctrsm+therk[0]+therk[1]+therk[2]+tctm+tmnp);
                    printf("%10d %10d %10d %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf(%d) %10.3lf\n",
                            j, nb, rows, tmtc,
                            tcnp,     // gemm
                            tcchol,   // potrf
                            tctrsm,   // trsm
                            (tcchol + tctrsm),
                            (tmtc+tcnp+tcchol+tctrsm),
                            therk[0], therk[1], therk[2], therk[3], // syrk
                            tctm, // copy panel to GPU
                            tmnp, // lookahead on GPU
                            (id + 1) % num_gpus,
                            (tcnp+tcchol+tctrsm+therk[0]+therk[1]+therk[2]+tctm+tmnp));
                    fflush(0);
                    #endif
                }
            }
            for( d = 0; d < num_gpus; d++ ) {
                magma_setdevice(d);
                for( id=0; id < num_streams; id++ ) {
                    magma_queue_sync( stream[d][id] );
                }
            }
            #if defined (ENABLE_TIMER)
            printf("\n%10d %10d %10d %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf(-) %10.3lf\n",
                    n, n, 0, ttot_mtc,
                    ttot_cnp,     // gemm
                    ttot_cchol,   // potrf
                    ttot_ctrsm,   // trsm
                    (ttot_cchol + ttot_ctrsm),
                    (ttot_mtc+ttot_cnp+ttot_cchol+ttot_ctrsm),
                    ttot_herk[0], ttot_herk[1], ttot_herk[2], ttot_herk[3], // syrk
                    ttot_ctm, // copy panel to GPU
                    ttot_mnp, // lookahead on GPU
                    (ttot_cnp+ttot_cchol+ttot_ctrsm+ttot_herk[0]+ttot_herk[1]+ttot_herk[2]+ttot_ctm+ttot_mnp));
            printf("%10d %10d %10d %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf(-) %10.3lf (ratio)\n",
                    n, n, 0, ttot_mtc/ttot,
                    ttot_cnp/ttot,     // gemm
                    ttot_cchol/ttot,   // potrf
                    ttot_ctrsm/ttot,   // trsm
                    (ttot_cchol + ttot_ctrsm)/ttot,
                    (ttot_mtc+ttot_cnp+ttot_cchol+ttot_ctrsm)/ttot,
                    ttot_herk[0]/ttot, ttot_herk[1]/ttot, ttot_herk[2]/ttot, ttot_herk[3]/ttot, // syrk
                    ttot_ctm/ttot, // copy panel to GPU
                    ttot_mnp/ttot, // lookahead on GPU
                    (ttot_cnp+ttot_cchol+ttot_ctrsm+ttot_herk[0]+ttot_herk[1]+ttot_herk[2]+ttot_ctm+ttot_mnp)/ttot);
            #endif

            // cholesky for the last block
            if (j < n && *info == 0) {
                rows = n - j;
                id = (j / nb) % num_gpus;

                // Set the local index where the current panel is
                j_local = j / (nb * num_gpus) * nb;
                
                magma_setdevice(id);
                #if defined (ENABLE_TIMER)
                tset = get_time();
                #endif
                magma_zgetmatrix(rows, rows, dlA(id, j, j_local), ldda, panel(j), ldpanel);
                lapackf77_zpotrf(MagmaLowerStr, &rows, panel(j), &ldpanel, info);
                magma_zsetmatrix(rows, rows, panel(j), ldpanel, dlA(id, j, j_local), ldda);
                #if defined (ENABLE_TIMER)
                tset = get_time() - tset;
                #endif
            }
            #if defined (ENABLE_TIMER)
            printf( " matrix_get,set: %10.3lf %10.3lf -> %10.3lf\n",tget,tset,ttot+tget+tset );
            #endif
        } // end of else not upper

        // clean up
        for( d = 0; d < num_gpus; d++ ) {
            magma_setdevice(d);
            for( j=0; j < num_streams; j++ ) {
                magma_queue_destroy( stream[d][j] );
            }
            magma_free( d_lP[d] );
        }
    } // end of not lapack

    // free workspace
    magma_free_pinned( panel );
    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );

    return *info;
} /* magma_zpotrf_mgpu_right */


extern "C" void
magma_zherk_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex **db, magma_int_t lddb, magma_int_t offset_b,
    double beta,
    magmaDoubleComplex **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_streams, magma_queue_t stream[][10])
{
#define dB(id, i, j)  (db[(id)]+(j)*lddb + (i)+offset_b)
#define dC(id, i, j)  (dc[(id)]+(j)*lddc + (i))

    const char* uplo_  = lapack_uplo_const( uplo  );
    magma_int_t i, id, ib, ii, kk, n1;
    magmaDoubleComplex z_alpha = MAGMA_Z_MAKE(alpha,0.0);
    magmaDoubleComplex z_beta  = MAGMA_Z_MAKE(beta, 0.0);

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    /* diagonal update */
    for( i=0; i < n; i += nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = STREAM_ID( i+offset );

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));

        /* zher2k on diagonal block */
        magma_setdevice(id);
        magmablasSetKernelStream(stream[id][kk]);
        trace_gpu_start( id, kk, "syr2k", "syr2k" );
        magma_zherk(uplo, trans, ib, k,
                    alpha,  dB(id, i,        0 ), lddb,
                     beta,  dC(id, i+offset, ii), lddc);
        trace_gpu_end( id, kk );
    }

    /* off-diagonal update */
    if (uplo == MagmaUpper) {
        for( i=nb; i < n; i += nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = STREAM_ID( i+offset );

            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));

            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, i, ib, k,
                        z_alpha, dB(id, 0, 0 ), lddb,
                                 dB(id, i, 0 ), lddb,
                        z_beta,  dC(id, 0, ii), lddc);
        }
    }
    else {
        for( i=0; i < n-nb; i += nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = STREAM_ID( i+offset );

            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            n1 = n-i-ib;

            /* zgemm on off-diagonal blocks */
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                        z_alpha, dB(id, i+ib,         0 ), lddb,
                                 dB(id,  i,           0 ), lddb,
                        z_beta,  dC(id,  i+offset+ib, ii), lddc);
            trace_gpu_end( id, kk );
        }
    }

    // TODO why not sync?
    //for( id=0; id < num_gpus; id++ ) {
    //    magma_setdevice(id);
    //    //for( kk=0; kk < num_streams; kk++ )
    //    //    magma_queue_sync(stream[id][kk]);
    //}
    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );
}


extern "C" void
magma_zherk_mgpu2(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex **db, magma_int_t lddb, magma_int_t offset_b,
    double beta,
    magmaDoubleComplex **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_streams, magma_queue_t stream[][10])
{
#define dB(id, i, j)  (db[(id)]+(j)*lddb + (i)+offset_b)
#define dC(id, i, j)  (dc[(id)]+(j)*lddc + (i))

    const char* uplo_  = lapack_uplo_const( uplo  );
    magma_int_t i, id, ib, ii, kk, n1;
    magmaDoubleComplex z_alpha = MAGMA_Z_MAKE(alpha,0.0);
    magmaDoubleComplex z_beta  = MAGMA_Z_MAKE(beta, 0.0);

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    /* diagonal update */
    for( i=0; i < n; i += nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = STREAM_ID( i+offset );

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));
    }

    if (uplo == MagmaUpper) {
        for( i=0; i < n; i += nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = STREAM_ID( i+offset );

            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            n1 = i+ib;

            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);

            /* zgemm on diag and off-diagonal blocks */
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                        z_alpha, dB(id, 0, 0 ), lddb,
                                 dB(id, i, 0 ), lddb,
                        z_beta,  dC(id, 0, ii), lddc);
        }
    }
    else {
        for( i=0; i < n; i += nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = STREAM_ID( i+offset );

            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            n1 = n-i;

            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            /* zgemm on diag and off-diagonal blocks */
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                        z_alpha, dB(id, i,         0), lddb,
                                 dB(id, i,         0), lddb,
                        z_beta,  dC(id, i+offset, ii), lddc);
            trace_gpu_end( id, kk );
        }
    }

    // TODO: why not sync?
    //for( id=0; id < num_gpus; id++ ) {
    //    magma_setdevice(id);
    //    //for( kk=0; kk < num_streams; kk++ )
    //    //    magma_queue_sync(stream[id][kk]);
    //}
    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );
}
