/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"
#include "trace.h"

/* === Define what BLAS to use ============================================ */
#define PRECISION_s

#if defined(PRECISION_s) || defined(PRECISION_d)
//#define STRSM_WORK
#endif
/* === End defining what BLAS to use ======================================= */


//#define dlA(id, i, j)  (d_lA[id] + (j)*ldda + (i))
//#define dlAT(id, i, j)  (d_lA[id] + (j)*lddat + (i))

//#define dlP(id, i, j)  (d_lP[id] + (j)*lddp + (i))
//#define dlPT(id, i, j)  (d_lP[id] + (j)*nb    + (i))

#define Alo(i, j)  (a   +            ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i))
#define Aup(i, j)  (a   +(nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i))

#define dlA(id, i, j)     (d_lA[(id)] + (j)*ldda + (i))
#define dlP(id, i, j, k)  (d_lP[(id)] + (k)*nb*lddp + (j)*lddp + (i))
#define dlPT(id, i, j, k) (d_lP[(id)] + (k)*nb*lddp + (j)*nb   + (i))


extern "C" magma_int_t
magma_spotrf2_mgpu(int num_gpus, char uplo, magma_int_t m, magma_int_t n,
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   float **d_lA,  magma_int_t ldda,
                   float **d_lP,  magma_int_t lddp,
                   float *a,      magma_int_t lda,   magma_int_t h,
                   magma_queue_t stream[][3], magma_event_t event[][5],
                   magma_int_t *info )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    SPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
       dA = U**T * U,  if UPLO = 'U', or
       dA = L  * L**T,  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of dA is stored;
            = 'L':  Lower triangle of dA is stored.

    N       (input) INTEGER
            The order of the matrix dA.  N >= 0.

    dA      (input/output) REAL array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = 'U', the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**T * U or dA = L * L**T.

    LDDA     (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.
    =====================================================================   */


    magma_int_t     j, jb, nb0, nb2, dd, d, id, j_local, j_local2, buf;
    char            uplo_[2] = {uplo, 0};
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float          d_one     =  1.0;
    float          d_neg_one = -1.0;
    int upper = lapackf77_lsame(uplo_, "U");
    float *dlpanel;
    //magma_event_t event0[MagmaMaxGPUs], // syrk
    //            event1[MagmaMaxGPUs], // send off-diagonal
    //            event2[MagmaMaxGPUs], // send diagonal
    //            event3[MagmaMaxGPUs]; // trsm
    magma_int_t n_local[MagmaMaxGPUs], ldpanel;
    int stream0 = 0, stream1 = 1;
    #ifdef STRSM_WORK
    float *d_dinvA[MagmaMaxGPUs][2], *d_x[MagmaMaxGPUs][2]; /* used by strsm_work */
    #endif
    
    *info = 0;
    if ( (! upper) && (! lapackf77_lsame(uplo_, "L")) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (!upper && num_gpus*ldda < max(1,n)) {
        *info = -4;
    } else if (upper && ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    for( d=0; d<num_gpus; d++ ) {
        /* local-n and local-ld */
        if (upper) {
            n_local[d] = ((n/nb)/num_gpus)*nb;
            if (d < (n/nb)%num_gpus)
                n_local[d] += nb;
            else if (d == (n/nb)%num_gpus)
                n_local[d] += n%nb;
        } else {
            n_local[d] = ((m/nb)/num_gpus)*nb;
            if (d < (m/nb)%num_gpus)
                n_local[d] += nb;
            else if (d == (m/nb)%num_gpus)
                n_local[d] += m%nb;
        }
        //magma_setdevice(d);
        //magma_event_create( &event0[d] );
        //magma_event_create( &event1[d] );
        //magma_event_create( &event2[d] );
        //magma_event_create( &event3[d] );
    }
    magma_setdevice(0);

    /* == initialize the trace */
    trace_init( 1, num_gpus, 3, (magma_queue_t*)stream );

    /* Use blocked code. */
    if (upper) {
        /* ---------------------------------------------- */
        /* Upper-triangular case                          */
        /* > Compute the Cholesky factorization A = U'*U. */
        /* ---------------------------------------------- */
        
#if defined(PRECISION_d) && defined(STRSM_WORK)
        /* invert the diagonals
         * Allocate device memory for the inversed diagonal blocks, size=m*NB
         */
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            for( j=0; j<2; j++ ) {
                magma_smalloc( &d_dinvA[d][j], nb*nb );
                magma_smalloc( &d_x[d][j],      n*nb );
                cudaMemset(d_dinvA[d][j], 0, nb*nb*sizeof(float));
                cudaMemset(d_x[d][j],     0,  n*nb*sizeof(float));
            }
        }
        magma_setdevice(0);
#endif
        
        for (j=0; j<m; j+=nb) {

            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%num_gpus;
            buf = (j/nb)%num_gpus;
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*num_gpus);
            jb = min(nb, (m-j));
            
            if( j > 0 ) {
                /* needed on pluto... */
                magma_setdevice(id);
                magma_queue_sync( stream[id][stream0] ); // wait for the column on CPU

                /* broadcast off-diagonal column to all gpus */
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    if( d != id ) {
                        magma_setdevice(d);
                
                        /* wait for it on CPU */
                        magma_queue_wait_event( stream[d][stream0], event[id][1] );
                
                        /* send it to GPU */
                        trace_gpu_start( d, stream0, "comm", "rows to GPUs" );
                        magma_ssetmatrix_async( j, jb,
                                                Aup(0,j),        lda,
                                                dlP(d,jb,0,buf), lddp,
                                                stream[d][stream0] );
                        trace_gpu_end( d, stream0 );
                        magma_event_record( event[d][1], stream[d][stream0] );
                    }
                    d = (d+1)%num_gpus;
                }
            }
            
            /* Update the current diagonal block */
            magma_setdevice(id);
            if( j > 0 ) {
                magmablasSetKernelStream(stream[id][stream1]);
                trace_gpu_start( id, stream1, "syrk", "syrk" );
                magma_ssyrk(MagmaUpper, MagmaTrans, jb, j,
                            d_neg_one, dlA(id, 0, nb*j_local), ldda,
                            d_one,     dlA(id, j, nb*j_local), ldda);
                trace_gpu_end( id, stream1 );
                magma_event_record( event[id][0], stream[id][stream1] );
            }

            /* send the diagonal to cpu */
            magma_queue_wait_event( stream[id][stream0], event[id][0] ); // wait for syrk
            trace_gpu_start( id, stream0, "comm", "D to CPU" );
            magma_sgetmatrix_async( jb, jb,
                                    dlA(id, j, nb*j_local), ldda,
                                    Aup(j,j),               lda,
                                    stream[id][stream0] );
            trace_gpu_end( id, stream0 );

            if ( j > 0 ) {

                /* Compute the local block column of the panel. */
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    j_local2 = j_local+1;
                    if( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
                
                    if( n_local[d] > nb0 ) {
                        /* wait for the off-diagonal */
                        if( d != id ) {
                            //magma_queue_sync( stream[id][3] );
                            dlpanel = dlP(d, jb, 0, buf);
                            ldpanel = lddp;
                
                            /* wait for the offdiagonal column */
                            magma_queue_wait_event( stream[d][stream1], event[d][1] );
                        } else {
                            dlpanel = dlA(d, 0, nb*j_local);
                            ldpanel = ldda;
                        }
                        
                        /* update the panel */
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][stream1]);
                        trace_gpu_start( d, stream1, "gemm", "gemm" );
                        magma_sgemm(MagmaTrans, MagmaNoTrans,
                                    jb, n_local[d]-nb0, j,
                                    c_neg_one, dlpanel,        ldpanel,
                                               dlA(d, 0, nb0), ldda,
                                    c_one,     dlA(d, j, nb0), ldda);
                        trace_gpu_end( d, stream1 );
                    }
                    d = (d+1)%num_gpus;
                }
            }
            
            /* factor the diagonal */
            magma_setdevice(id);
            magma_queue_sync( stream[id][stream0] ); // wait for the diagonal
            trace_cpu_start( 0, "getrf", "getrf" );
            lapackf77_spotrf(MagmaUpperStr, &jb, Aup(j,j), &lda, info);
            trace_cpu_end( 0 );
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus */
            if ( (j+jb) < n) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    magma_setdevice(d);
                    if( d == id ) {
                        dlpanel = dlA(d, j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d, 0, 0, buf);
                        ldpanel = lddp;
                    }
                    
                    trace_gpu_start( d, stream0, "comm", "D to GPUs" );
                    magma_ssetmatrix_async( jb, jb,
                                            Aup(j,j), lda,
                                            dlpanel,  ldpanel,
                                            stream[d][stream0] );
                    trace_gpu_end( d, stream0 );
                    magma_event_record( event[d][2], stream[d][stream0] );
                    d = (d+1)%num_gpus;
                }
            } else {
                magma_setdevice(id);
                trace_gpu_start( id, stream0, "comm", "D to GPUs" );
                magma_ssetmatrix_async( jb, jb,
                                        Aup(j,j),               lda,
                                        dlA(id, j, nb*j_local), ldda,
                                        stream[id][stream0] );
                trace_gpu_end( id, stream0 );
            }
            
            /* panel-factorize the off-diagonal */
            if ( (j+jb) < n) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if( d > id ) j_local2--;
                    if( d == id ) {
                        dlpanel = dlA(d, j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d, 0, 0, buf);
                        ldpanel = lddp;
                    }
                    nb2 = n_local[d]-nb*j_local2;
                    nb0 = min(nb, nb2 );
                    
                    magma_setdevice(d);
                    magmablasSetKernelStream(stream[d][stream1]);
                    magma_queue_wait_event( stream[d][stream1], event[d][2] ); // wait for the diagonal
                    if( j+jb < m && d == (j/nb+1)%num_gpus ) {
                        /* owns the next column, look-ahead the column */
                        trace_gpu_start( d, stream1, "trsm", "trsm" );
#if defined(PRECISION_d) && defined(STRSM_WORK)
                        magmablas_strsm_work( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                              jb, nb0, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                              d_dinvA[d][0], d_x[d][0] );
                        /*nb2 = n_local[d] - j_local2*nb;
                        magmablas_strsm_work( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                              d_dinvA[d], d_x[d] );*/
#else
                        /*nb2 = n_local[d] - j_local2*nb;
                        magma_strsm( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel,                ldda,
                                     dlA(d, j, nb*j_local2), ldda);
                        */
                        magma_strsm( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                     jb, nb0, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, j, nb*j_local2), ldda);
#endif
                        trace_gpu_end( d, stream1 );
                        magma_event_record( event[d][3], stream[d][stream1] );
                        
                        /* send the column to cpu */
                        if( j+jb < m ) {
                            trace_gpu_start( d, stream0, "comm", "rows to CPU" );
                            magma_queue_wait_event( stream[d][stream0], event[d][3] ); // wait for lookahead
                            magma_sgetmatrix_async( (j+jb), nb0,
                                                    dlA(d, 0, nb*j_local2), ldda,
                                                    Aup(0,j+jb),            lda,
                                                    stream[d][stream0] );
                            trace_gpu_end( d, stream0 );
                            magma_event_record( event[d][1], stream[d][stream0] );
                        }
                        
                        /* update the remaining blocks */
                        nb2 = nb2 - nb0;
#if defined(PRECISION_d) && defined(STRSM_WORK)
                        magmablas_strsm_work( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel,                    ldpanel,
                                              dlA(d, j, nb*j_local2+nb0), ldda,
                                              d_dinvA[d][1], d_x[d][1] );
#else
                        magma_strsm( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel,                    ldpanel,
                                     dlA(d, j, nb*j_local2+nb0), ldda);
#endif
                    } else if( nb2 > 0 ) {
                        /* update the entire trailing matrix */
                        trace_gpu_start( d, stream1, "trsm", "trsm" );
#if defined(PRECISION_d) && defined(STRSM_WORK)
                        magmablas_strsm_work( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                    d_dinvA[d][1], d_x[d][1] );
#else
                        magma_strsm( MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, j, nb*j_local2), ldda);
#endif
                        trace_gpu_end( d, stream1 );
                    }
                    d = (d+1)%num_gpus;
                }
            } /* end of strsm */
        } /* end of for j=1, .., n */
    } else {
        /* -------------------------------------------- */
        /* Lower-triangular case                        */
        /* Compute the Cholesky factorization A = L*L'. */
        /* -------------------------------------------- */
#if defined(PRECISION_d) && defined(STRSM_WORK)
        /*
         * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE
         */
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            for( j=0; j<2; j++ ) {
                magma_smalloc( &d_dinvA[d][j], nb*nb );
                magma_smalloc( &d_x[d][j],     nb*m  );
                cudaMemset(d_dinvA[d][j], 0, nb*nb*sizeof(float));
                cudaMemset(d_x[d][j],     0, nb* m*sizeof(float));
            }
        }
        magma_setdevice(0);
#endif

        for (j=0; j<n; j+=nb) {

            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%num_gpus;
            buf = (j/nb)%num_gpus;
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*num_gpus);
            jb = min(nb, (n-j));
            
            if( j > 0 ) {
                /* needed on pluto... */
                magma_setdevice(id);
                magma_queue_sync( stream[id][stream0] ); // wait for the column on CPU

                /* broadcast offdiagonal row to all gpus */
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    if( d != id ) {
                        magma_setdevice(d);
                        /* wait for it on CPU */
                        magma_queue_wait_event( stream[d][stream0], event[id][1] );
            
                        /* send it to GPU */
                        magma_ssetmatrix_async( jb, j,
                                                Alo(j,0),         lda,
                                                dlPT(d,0,jb,buf), nb,
                                                stream[d][stream0] );
                        magma_event_record( event[d][1], stream[d][stream0] );
                    }
                    d = (d+1)%num_gpus;
                }
            }

            /* Update the current diagonal block */
            magma_setdevice(id);
            if( j > 0 ) {
                magmablasSetKernelStream(stream[id][stream1]);
                magma_ssyrk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dlA(id, nb*j_local, 0), ldda,
                            d_one,     dlA(id, nb*j_local, j), ldda);
                magma_event_record( event[id][0], stream[id][stream1] );
            }
            
            /* send the diagonal to cpu */
            magma_queue_wait_event( stream[id][stream0], event[id][0] ); // wait for syrk
            magma_sgetmatrix_async( jb, jb,
                                    dlA(id, nb*j_local, j), ldda,
                                    Alo(j,j),               lda,
                                    stream[id][stream0] );

            /* update the offdiagonal blocks */
            if ( j > 0 ) {
                /* compute the block-rows of the panel */
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    j_local2 = j_local+1;
                    if( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
            
                    if( nb0 < n_local[d] ) {
                        if( d != id ) {
                            dlpanel = dlPT(d, 0, jb, buf);
                            ldpanel = nb;
            
                            /* wait for offdiagonal row */
                            magma_queue_wait_event( stream[d][stream1], event[d][1] );
                        } else {
                            dlpanel = dlA(d, nb*j_local, 0);
                            ldpanel = ldda;
                        }
            
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][stream1]);
                        magma_sgemm( MagmaNoTrans, MagmaTrans,
                                     n_local[d]-nb0, jb, j,
                                     c_neg_one, dlA(d, nb0, 0), ldda,
                                                dlpanel,        ldpanel,
                                     c_one,     dlA(d, nb0, j), ldda);
                    }
                    d = (d+1)%num_gpus;
                }
            }

            /* factor the diagonal */
            magma_setdevice(id);
            magma_queue_sync( stream[id][stream0] );
            lapackf77_spotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus */
            if ( (j+jb) < m ) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    magma_setdevice(d);
                    if( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    magma_ssetmatrix_async( jb, jb,
                                            Alo(j,j), lda,
                                            dlpanel,  ldpanel,
                                            stream[d][stream0] );
                    magma_event_record( event[d][2], stream[d][stream0] );
                    d = (d+1)%num_gpus;
                }
            } else {
                magma_setdevice(id);
                magma_ssetmatrix_async( jb, jb,
                                        Alo(j,j),               lda,
                                        dlA(id, nb*j_local, j), ldda,
                                        stream[id][stream0] );
            }

            /* factorize off-diagonal blocks */
            if ( (j+jb) < m ) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if( d > id ) j_local2--;
                    if( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2 );
            
                    magma_setdevice(d);
                    magmablasSetKernelStream(stream[d][stream1]);
                    magma_queue_wait_event( stream[d][stream1], event[d][2] ); // wait for the diagonal
                    if( j+jb < n && d == (j/nb+1)%num_gpus ) {
                        /* owns the next column, look-ahead the column */
#if defined(PRECISION_d) && defined(STRSM_WORK)
                        magmablas_strsm_work( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                                              nb0, jb, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, nb*j_local2, j), ldda,
                                              d_dinvA[d][0], d_x[d][0]);
#else
                        magma_strsm( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                                     nb0, jb, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, nb*j_local2, j), ldda);
#endif
                        magma_event_record( event[d][3], stream[d][stream1] );

                        /* send the column to cpu */
                        if( j+jb < n ) {
                            magma_queue_wait_event( stream[d][stream0], event[d][3] ); // wait for lookahead
                            magma_sgetmatrix_async( nb0, j+jb,
                                                    dlA(d, nb*j_local2, 0), ldda,
                                                    Alo(j+jb,0),            lda,
                                                    stream[d][stream0] );
                            magma_event_record( event[d][1], stream[d][stream0] );
                        }

                        /* update the remaining blocks */
                        nb2 = nb2 - nb0;
#if defined(PRECISION_d) && defined(STRSM_WORK)
                        magmablas_strsm_work( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                                              nb2, jb, c_one,
                                              dlpanel,                    ldpanel,
                                              dlA(d, nb*j_local2+nb0, j), ldda,
                                              d_dinvA[d][1], d_x[d][1] );
#else
                        magma_strsm( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                                     nb2, jb, c_one,
                                     dlpanel,                    ldpanel,
                                     dlA(d, nb*j_local2+nb0, j), ldda);
#endif
                    } else if( nb2 > 0 ) {
                        /* update the entire trailing matrix */
#if defined(PRECISION_d) && defined(STRSM_WORK)
                        magmablas_strsm_work( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                                              nb2, jb, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, nb*j_local2, j), ldda,
                                              d_dinvA[d][1], d_x[d][1] );
#else
                        magma_strsm( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                                     nb2, jb, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, nb*j_local2, j), ldda);
#endif
                    }
                    d = (d+1)%num_gpus;
                }
            }
        }
    } /* end of else not upper */

    /* == finalize the trace == */
    trace_finalize( "spotrf.svg","trace.css" );

    /* clean up */
    for( d=0; d<num_gpus; d++ ) {
        magma_setdevice(d);
        magma_queue_sync( stream[d][0] );
        magma_queue_sync( stream[d][1] );
        magmablasSetKernelStream(NULL);

        //magma_event_destroy( event0[d] );
        //magma_event_destroy( event1[d] );
        //magma_event_destroy( event2[d] );
        //magma_event_destroy( event3[d] );
    }
    magma_setdevice(0);

    return *info;
} /* magma_spotrf_mgpu */
