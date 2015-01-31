/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @author Raffaele Solca
       @author Ichitaro Yamazaki
       @author Mark Gates

       @generated from zlatrd_mgpu.cpp normal z -> c, Fri Jan 30 19:00:17 2015

*/
#include "common_magma.h"
#include "trace.h"

#define PRECISION_c
#define COMPLEX

/**
    Purpose
    -------
    CLATRD reduces NB rows and columns of a complex Hermitian matrix A to
    Hermitian tridiagonal form by an orthogonal similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = MagmaUpper, CLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = MagmaLower, CLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by CHETRD.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is stored:
      -     = MagmaUpper: Upper triangular
      -     = MagmaLower: Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.

    @param[in]
    nb      INTEGER
            The number of rows and columns to be reduced.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit:
      -     if UPLO = MagmaUpper, the last NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements above the diagonal
              with the array TAU, represent the orthogonal matrix Q as a
              product of elementary reflectors;
      -     if UPLO = MagmaLower, the first NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements below the diagonal
              with the array TAU, represent the  orthogonal matrix Q as a
              product of elementary reflectors.
            See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= (1,N).

    @param[out]
    e       COMPLEX array, dimension (N-1)
            If UPLO = MagmaUpper, E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = MagmaLower, E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    @param[out]
    tau     COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = MagmaUpper, and in TAU(1:nb) if UPLO = MagmaLower.
            See Further Details.

    @param[out]
    W       COMPLEX array, dimension (LDW,NB)
            The n-by-nb matrix W required to update the unreduced part
            of A.

    @param[in]
    ldw     INTEGER
            The leading dimension of the array W. LDW >= max(1,N).

    @param
    dA

    @param[in]
    ldda

    @param[in]
    offset

    @param
    dW

    @param[in]
    lddw

    @param
    hwork

    @param[in]
    lhwork

    @param
    dwork

    @param[in]
    ldwork
             
    @param[in]
    queues  magma_queue_t array of dimension (ngpu).
            queues[dev] is an execution queue on GPU dev.
    
    Further Details
    ---------------
    If UPLO = MagmaUpper, the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n) H(n-1) . . . H(n-nb+1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
    and tau in TAU(i-1).

    If UPLO = MagmaLower, the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
    and tau in TAU(i).

    The elements of the vectors v together form the n-by-nb matrix V
    which is needed, with W, to apply the transformation to the unreduced
    part of the matrix, using a Hermitian rank-2k update of the form:
    A := A - V*W' - W*V'.

    The contents of A on exit are illustrated by the following examples
    with n = 5 and nb = 2:

    if UPLO = MagmaUpper:                       if UPLO = MagmaLower:

      (  a   a   a   v4  v5 )              (  d                  )
      (      a   a   v4  v5 )              (  1   d              )
      (          a   1   v5 )              (  v1  1   a          )
      (              d   1  )              (  v1  v2  a   a      )
      (                  d  )              (  v1  v2  a   a   a  )

    where d denotes a diagonal element of the reduced matrix, a denotes
    an element of the original matrix that is unchanged, and vi denotes
    an element of the vector defining H(i).

    @ingroup magma_cheev_aux
    ********************************************************************/
extern "C" magma_int_t
magma_clatrd_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb, magma_int_t nb0,
    magmaFloatComplex *A,  magma_int_t lda,
    float *e, magmaFloatComplex *tau,
    magmaFloatComplex *W,          magma_int_t ldw,
    magmaFloatComplex_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex_ptr dW[],    magma_int_t lddw,
    magmaFloatComplex    *hwork,   magma_int_t lhwork,
    magmaFloatComplex_ptr dwork[], magma_int_t ldwork,
    magma_queue_t queues[] )
{
#define A(i, j) (A + (j)*lda + (i))
#define W(i, j) (W + (j)*ldw + (i))

#define dA(dev, i, j)  (dA[(dev)] + ((j)+loffset)*ldda + (i) + offset)
#define dW(dev, i, j)  (dW[(dev)] + (j)          *lddw + (i))
#define dW1(dev, i, j) (dW[(dev)] + ((j)+nb)     *lddw + (i))

    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    const magma_int_t ione = 1;

    magmaFloatComplex alpha, value;
    magma_int_t dev;
    magma_int_t i, n_i, n_i_1, ip1, iw;

    // TODO check arguments
    magma_int_t info = 0;
    if (n <= 0) {
        return info;
    }
    
    // TODO allocate f in chetrd and pass into clatrd. (e.g., expand hwork a bit)
    magmaFloatComplex *f;
    magma_cmalloc_cpu( &f, n );
    if ( f == NULL ) {
        info = MAGMA_ERR_HOST_ALLOC;
        return info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    if (uplo == MagmaUpper) {
        /* Reduce last NB columns of upper triangle */
        for (i = n-1; i >= n - nb; --i) {
            ip1 = i + 1;
            n_i_1 = n - i - 1;
            iw = i - n + nb;
            if (i < n-1) {
                /* Update A(1:i,i) */
                magmaFloatComplex wii = -conj( *W(i, iw+1) );
                blasf77_caxpy( &ip1, &wii, A(0, i+1), &ione, A(0, i), &ione );

                wii = -conj( *A(i, i+1) );
                blasf77_caxpy( &ip1, &wii, W(0, iw+1), &ione, A(0, i), &ione );
            }
            if (i > 0) {
                /* Generate elementary reflector H(i) to annihilate A(1:i-2,i) */
                alpha = *A(i-1, i);
                lapackf77_clarfg( &i, &alpha, A(0, i), &ione, &tau[i - 1] );

                e[i-1] = MAGMA_C_REAL( alpha );
                *A(i-1,i) = MAGMA_C_ONE;
                
                // TODO Previously, this set dx2[dev] = dW1(dev, 0, iw); and used dx2 in chemv.
                // TODO Now chemv handles broadcasting x to the GPUs, but data in dW1 is
                // TODO apparently still used in chetrd_mgpu / cher2k_mgpu.
                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    magma_csetvector_async( n, A(0,i), 1, dW1(dev, 0, iw), 1, queues[dev] );
                }
                magmablas_chemv_mgpu(
                    MagmaUpper, i, c_one, dA, ldda, 0,
                    A(0,i), 1, c_zero, W(0, iw), 1,
                    hwork, lhwork, dwork, ldwork, ngpu, nb0, queues );

                if (i < n-1) {
                    blasf77_cgemv( MagmaConjTransStr, &i, &n_i_1, &c_one,
                                   W(0,   iw+1), &ldw,
                                   A(0,   i),    &ione, &c_zero,
                                   W(i+1, iw),   &ione );
                }

                /* overlap update */
                if ( i < n-1 && i-1 >= n - nb ) {
                    /* Update A(1:i,i) */
                    #ifdef COMPLEX
                    lapackf77_clacgv( &n_i_1, W(i-1, iw+1), &ldw );
                    #endif
                    blasf77_cgemv( "No transpose", &i, &n_i_1, &c_neg_one,
                                   A(0,   i+1),  &lda,
                                   W(i-1, iw+1), &ldw, &c_one,
                                   A(0,   i-1),  &ione );
                    #ifdef COMPLEX
                    lapackf77_clacgv( &n_i_1, W(i-1, iw+1), &ldw );
                    lapackf77_clacgv( &n_i_1, A(i-1, i +1), &lda );
                    #endif
                    blasf77_cgemv( "No transpose", &i, &n_i_1, &c_neg_one,
                                   W(0,   iw+1), &ldw,
                                   A(i-1, i+1),  &lda, &c_one,
                                   A(0,   i-1),  &ione );
                    #ifdef COMPLEX
                    lapackf77_clacgv( &n_i_1, A(i-1, i+1), &lda );
                    #endif
                }

                // synchronize to get chemv result W(0, iw)
                magmablas_chemv_mgpu_sync(
                    MagmaUpper, i, c_one, dA, ldda, 0,
                    A(0,i), 1, c_zero, W(0, iw), 1,
                    hwork, lhwork, dwork, ldwork, ngpu, nb0, queues );

                if (i < n-1) {
                    blasf77_cgemv( "No transpose", &i, &n_i_1, &c_neg_one,
                                   A(0,   i+1), &lda,
                                   W(i+1, iw),  &ione, &c_one,
                                   W(0,   iw),  &ione );

                    blasf77_cgemv( MagmaConjTransStr, &i, &n_i_1, &c_one,
                                   A(0,   i+1), &lda,
                                   A(0,   i),   &ione, &c_zero,
                                   W(i+1, iw),  &ione );

                    blasf77_cgemv( "No transpose", &i, &n_i_1, &c_neg_one,
                                   W(0,   iw+1), &ldw,
                                   W(i+1, iw),   &ione, &c_one,
                                   W(0,   iw),   &ione );
                }

                blasf77_cscal( &i, &tau[i - 1], W(0, iw), &ione );

                value = magma_cblas_cdotc( i, W(0,iw), ione, A(0,i), ione );
                alpha = tau[i - 1] * -0.5f * value;
                blasf77_caxpy( &i, &alpha, A(0, i), &ione, W(0, iw), &ione );

                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    magma_csetvector_async( n, W(0,iw), 1, dW(dev, 0, iw), 1, queues[dev] );
                }
            }
        }
    } else {
        /*  Reduce first NB columns of lower triangle */
        for (i = 0; i < nb; ++i) {
            /* Update A(i:n,i) */
            n_i = n - i;
            //idw = ((offset+i)/nb)%ngpu;
            if ( i > 0 ) {
                trace_cpu_start( 0, "gemv", "gemv" );
                magmaFloatComplex wii = -conj( *W(i, i-1) );
                blasf77_caxpy( &n_i, &wii, A(i, i-1), &ione, A(i, i), &ione );

                wii = -conj( *A(i, i-1) );
                blasf77_caxpy( &n_i, &wii, W(i, i-1), &ione, A(i, i), &ione );
            }

            if (i < n-1) {
                /* Generate elementary reflector H(i) to annihilate A(i+2:n,i) */
                n_i_1 = n - i - 1;
                trace_cpu_start( 0, "larfg", "larfg" );
                alpha = *A(i+1, i);
                lapackf77_clarfg( &n_i_1, &alpha, A(min(i+2,n-1), i), &ione, &tau[i] );
                e[i] = MAGMA_C_REAL( alpha );
                *A(i+1,i) = MAGMA_C_ONE;
                trace_cpu_end( 0 );

                /* Compute W(i+1:n,i) */
                // TODO Previously, this set dx2[id] = dW1(id, 0, i)-offset; and used dx2 in chemv.
                // TODO Now chemv handles broadcasting x to the GPUs, but data in dW1 is
                // TODO apparently still used in chetrd_mgpu / cher2k_mgpu.
                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    magma_csetvector_async( n, A(0,i), 1, dW1(dev, 0, i), 1, queues[dev] );
                }
                
                magmablas_chemv_mgpu(
                    MagmaLower, n_i_1, c_one, dA, ldda, offset+i+1,
                    A(i+1, i), 1, c_zero, W(i+1, i), 1,
                    hwork, lhwork, dwork, ldwork, ngpu, nb0, queues );
                
                trace_cpu_start( 0, "gemv", "gemv" );
                blasf77_cgemv( MagmaConjTransStr, &n_i_1, &i, &c_one,
                               W(i+1, 0), &ldw,
                               A(i+1, i), &ione, &c_zero,
                               W(0,   i), &ione );
                
                blasf77_cgemv( "No transpose", &n_i_1, &i, &c_neg_one,
                               A(i+1, 0), &lda,
                               W(0,   i), &ione, &c_zero,
                               f,         &ione );
                
                blasf77_cgemv( MagmaConjTransStr, &n_i_1, &i, &c_one,
                               A(i+1, 0), &lda,
                               A(i+1, i), &ione, &c_zero,
                               W(0,   i), &ione );
                trace_cpu_end( 0 );

                /* overlap update */
                if ( i > 0 && i+1 < n ) {
                    trace_cpu_start( 0, "gemv", "gemv" );
                    #ifdef COMPLEX
                    lapackf77_clacgv( &i, W(i+1, 0), &ldw );
                    #endif
                    blasf77_cgemv( "No transpose", &n_i_1, &i, &c_neg_one,
                                   A(i+1, 0),   &lda,
                                   W(i+1, 0),   &ldw, &c_one,
                                   A(i+1, i+1), &ione );
                    #ifdef COMPLEX
                    lapackf77_clacgv( &i, W(i+1, 0), &ldw );
                    lapackf77_clacgv( &i, A(i+1, 0), &lda );
                    #endif
                    blasf77_cgemv( "No transpose", &n_i_1, &i, &c_neg_one,
                                   W(i+1, 0),   &ldw,
                                   A(i+1, 0),   &lda, &c_one,
                                   A(i+1, i+1), &ione );
                    #ifdef COMPLEX
                    lapackf77_clacgv( &i, A(i+1, 0), &lda );
                    #endif
                    trace_cpu_end( 0 );
                }

                // synchronize to get chemv result W(i+1, i)
                magmablas_chemv_mgpu_sync(
                    MagmaLower, n_i_1, c_one, dA, ldda, offset+i+1,
                    A(i+1, i), 1, c_zero, W(i+1, i), 1,
                    hwork, lhwork, dwork, ldwork, ngpu, nb0, queues );
                
                trace_cpu_start( 0, "axpy", "axpy" );
                if (i != 0) {
                    blasf77_caxpy( &n_i_1, &c_one, f, &ione, W(i+1, i), &ione );
                }

                blasf77_cgemv( "No transpose", &n_i_1, &i, &c_neg_one,
                               W(i+1, 0), &ldw,
                               W(0,   i), &ione, &c_one,
                               W(i+1, i), &ione );
                blasf77_cscal( &n_i_1, &tau[i], W(i+1,i), &ione );

                value = magma_cblas_cdotc( n_i_1, W(i+1,i), ione, A(i+1,i), ione );
                alpha = tau[i] * -0.5f * value;
                blasf77_caxpy( &n_i_1, &alpha, A(i+1, i), &ione, W(i+1,i), &ione );
                trace_cpu_end( 0 );
                for( dev=0; dev < ngpu; dev++ ) {
                    magma_setdevice( dev );
                    magma_csetvector_async( n, W(0,i), 1, dW(dev, 0, i), 1, queues[dev] );
                }
            }
        }
    }

    magma_free_cpu( f );

    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );
    
    return info;
} /* magma_clatrd_mgpu */

#undef A
#undef W
#undef dA
#undef dW
#undef dW1
