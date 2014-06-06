/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:36 2013
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_d

extern "C" magma_int_t
magma_dgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    double *T,
    magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    DGEHRD reduces a DOUBLE_PRECISION general matrix A to upper Hessenberg form H by
    an orthogonal similarity transformation:  Q' * A * Q = H . This version
    stores the triangular matrices used in the factorization so that they can
    be applied directly (i.e., without being recomputed) later. As a result,
    the application of Q is much faster.

    Arguments
    =========
    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    ILO     (input) INTEGER
    IHI     (input) INTEGER
            It is assumed that A is already upper triangular in rows
            and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
            set by a previous call to DGEBAL; otherwise they should be
            set to 1 and N respectively. See Further Details.
            1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.

    A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the N-by-N general matrix to be reduced.
            On exit, the upper triangle and the first subdiagonal of A
            are overwritten with the upper Hessenberg matrix H, and the
            elements below the first subdiagonal, with the array TAU,
            represent the orthogonal matrix Q as a product of elementary
            reflectors. See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    TAU     (output) DOUBLE_PRECISION array, dimension (N-1)
            The scalar factors of the elementary reflectors (see Further
            Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
            zero.

    WORK    (workspace/output) DOUBLE_PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.  LWORK >= max(1,N).
            For optimum performance LWORK >= N*NB, where NB is the
            optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    T       (output)  DOUBLE_PRECISION array, dimension NB*N,
            where NB is the optimal blocksize. It stores the NB*NB blocks
            of the triangular T matrices used in the reduction.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ===============
    The matrix Q is represented as a product of (ihi-ilo) elementary
    reflectors

       Q = H(ilo) H(ilo+1) . . . H(ihi-1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
    exit in A(i+2:ihi,i), and tau in TAU(i).

    The contents of A are illustrated by the following example, with
    n = 7, ilo = 2 and ihi = 6:

    on entry,                        on exit,

    ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
    (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
    (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
    (                         a )    (                          a )

    where a denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an
    element of the vector defining H(i).

    This implementation follows the hybrid algorithm and notations described in

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.

    This version stores the T matrices, for later use in magma_dorghr.

    =====================================================================    */

    #define  A( i, j )    (A + (i) + (j)*lda)
    #define dA( d, i, j ) (data.A[d] + (i) + (j)*ldda)

    double c_one  = MAGMA_D_ONE;
    double c_zero = MAGMA_D_ZERO;

    magma_int_t nb = magma_get_dgehrd_nb(n);

    magma_int_t nh, iws, ldda, min_lblocks, max_lblocks, last_dev, d;
    magma_int_t dpanel, di, nlocal, i, i2, ib, ldwork;
    magma_int_t iinfo;
    magma_int_t lquery;
    struct dgehrd_data data;

    int ngpu = magma_num_gpus();
    
    *info = 0;
    iws = n*(nb + nb*ngpu);
    work[0] = MAGMA_D_MAKE( iws, 0 );

    lquery = lwork == -1;
    if (n < 0) {
        *info = -1;
    } else if (ilo < 1 || ilo > max(1,n)) {
        *info = -2;
    } else if (ihi < min(ilo,n) || ihi > n) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if (lwork < max(1,n) && ! lquery) {
        *info = -8;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;
    
    magma_device_t cdevice;
    magma_getdevice( &cdevice );

    // Adjust from 1-based indexing
    ilo -= 1;
    
    // Quick return if possible
    nh = ihi - ilo;
    if (nh <= 1) {
        work[0] = c_one;
        return *info;
    }

    // Set elements 0:ILO-1 and IHI-1:N-2 of TAU to zero
    for(i = 0; i < ilo; ++i)
        tau[i] = c_zero;

    for(i = max(0,ihi-1); i < n-1; ++i)
        tau[i] = c_zero;

    // set T to zero
    lapackf77_dlaset( "Full", &nb, &n, &c_zero, &c_zero, T, &nb );

    // set to null, to simplify cleanup code
    for( d = 0; d < ngpu; ++d ) {
        data.A[d]       = NULL;
        data.streams[d] = NULL;
    }
    
    // If not enough workspace, use unblocked code
    if ( lwork < iws ) {
        nb = 1;
    }

    if (nb == 1 || nb >= nh) {
        // Use unblocked code below
        i = ilo;
    }
    else {
        // Use blocked code
        // allocate memory on GPUs for A and workspaces
        ldda = ((n+31)/32)*32;
        min_lblocks = (n     / nb) / ngpu;
        max_lblocks = ((n-1) / nb) / ngpu + 1;
        last_dev    = (n     / nb) % ngpu;
        
        // V and Vd need to be padded for copying in mdlahr2
        data.ngpu = ngpu;
        data.ldda = ldda;
        data.ldv  = nb*max_lblocks*ngpu;
        data.ldvd = nb*max_lblocks;
        
        for( d = 0; d < ngpu; ++d ) {
            magma_setdevice( d );
            nlocal = min_lblocks*nb;
            if ( d < last_dev ) {
                nlocal += nb;
            }
            else if ( d == last_dev ) {
                nlocal += (n % nb);
            }
            
            ldwork = nlocal*ldda   // A
                   + nb*data.ldv   // V
                   + nb*data.ldvd  // Vd
                   + nb*ldda       // Y
                   + nb*ldda       // W
                   + nb*nb;        // Ti
            if ( MAGMA_SUCCESS != magma_dmalloc( &data.A[d], ldwork )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto CLEANUP;
            }
            data.V [d] = data.A [d] + nlocal*ldda;
            data.Vd[d] = data.V [d] + nb*data.ldv;
            data.Y [d] = data.Vd[d] + nb*data.ldvd;
            data.W [d] = data.Y [d] + nb*ldda;
            data.Ti[d] = data.W [d] + nb*ldda;
            
            magma_queue_create( &data.streams[d] );
        }
        
        // Copy the matrix to GPUs
        magma_dsetmatrix_1D_col_bcyclic( n, n, A, lda, data.A, ldda, ngpu, nb );
        
        // round ilo down to block boundary
        ilo = (ilo/nb)*nb;
        for (i = ilo; i < ihi - 1 - nb; i += nb) {
            //   Reduce columns i:i+nb-1 to Hessenberg form, returning the
            //   matrices V and T of the block reflector H = I - V*T*V'
            //   which performs the reduction, and also the matrix Y = A*V*T
            
            //   Get the current panel (no need for the 1st iteration)
            dpanel =  (i / nb) % ngpu;
            di     = ((i / nb) / ngpu) * nb;
            if ( i > ilo ) {
                magma_setdevice( dpanel );
                magma_dgetmatrix( ihi-i, nb,
                                  dA(dpanel, i, di), ldda,
                                  A(i,i),            lda );
            }
            
            // add 1 to i for 1-based index
            magma_dlahr2_m( ihi, i+1, nb, A(0,i), lda,
                            &tau[i], &T[i*nb], nb, work, n, &data );
            
            magma_dlahru_m( n, ihi, i, nb, A, lda, &data );
            
            // copy first i rows above panel to host
            magma_setdevice( dpanel );
            magma_dgetmatrix_async( i, nb,
                                    dA(dpanel, 0, di), ldda,
                                    A(0,i),            lda, data.streams[dpanel] );
        }
        
        // Copy remainder to host, block-by-block
        for( i2 = i; i2 < n; i2 += nb ) {
            ib = min( nb, n-i2 );
            d  = (i2 / nb) % ngpu;
            di = (i2 / nb) / ngpu * nb;
            magma_setdevice( d );
            magma_dgetmatrix( n, ib,
                              dA(d, 0, di), ldda,
                              A(0,i2),      lda );
        }
    }

    // Use unblocked code to reduce the rest of the matrix
    // add 1 to i for 1-based index
    i += 1;
    lapackf77_dgehd2(&n, &i, &ihi, A, &lda, tau, work, &iinfo);
    work[0] = MAGMA_D_MAKE( iws, 0 );
    
CLEANUP:
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magmablasSetKernelStream( NULL );
        magma_free( data.A[d] );
        data.A[d] = NULL;
        if ( data.streams[d] != NULL ) {
            magma_queue_destroy( data.streams[d] );
        }
    }
    magma_setdevice( cdevice );
    
    return *info;
} // magma_dgehrd
