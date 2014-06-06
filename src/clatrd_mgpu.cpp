/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Stan Tomov
       @author Raffaele Solca

       @generated c Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"
#include "trace.h"

#include <cblas.h>

#define PRECISION_c

#define MAGMABLAS_CHEMV_MGPU

#define A(i, j) (a+(j)*lda + (i))
#define W(i, j) (w+(j)*ldw + (i))

#define dA(id, i, j)  (da[(id)] + ((j)+loffset)*ldda + (i) + offset)
#define dW(id, i, j)  (dw[(id)] + (j)          *lddw + (i))
#define dW1(id, i, j) (dw[(id)] + ((j)+nb)     *lddw + (i))

extern "C" float
magma_clatrd_mgpu(magma_int_t num_gpus, char uplo,
                  magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
                  magmaFloatComplex *a,  magma_int_t lda,
                  float *e, magmaFloatComplex *tau,
                  magmaFloatComplex *w,   magma_int_t ldw,
                  magmaFloatComplex **da, magma_int_t ldda, magma_int_t offset,
                  magmaFloatComplex **dw, magma_int_t lddw,
                  magmaFloatComplex *dwork[MagmaMaxGPUs], magma_int_t ldwork,
                  magma_int_t k,
                  magmaFloatComplex  *dx[MagmaMaxGPUs], magmaFloatComplex *dy[MagmaMaxGPUs],
                  magmaFloatComplex *work,
                  magma_queue_t stream[][10],
                  float *times)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    CLATRD reduces NB rows and columns of a complex Hermitian matrix A to
    Hermitian tridiagonal form by an orthogonal similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = 'U', CLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = 'L', CLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by CHETRD.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is stored:
            = 'U': Upper triangular
            = 'L': Lower triangular

    N       (input) INTEGER
            The order of the matrix A.

    NB      (input) INTEGER
            The number of rows and columns to be reduced.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit:
            if UPLO = 'U', the last NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements above the diagonal
              with the array TAU, represent the orthogonal matrix Q as a
              product of elementary reflectors;
            if UPLO = 'L', the first NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements below the diagonal
              with the array TAU, represent the  orthogonal matrix Q as a
              product of elementary reflectors.
            See Further Details.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= (1,N).

    E       (output) COMPLEX array, dimension (N-1)
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    TAU     (output) COMPLEX array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.
            See Further Details.

    W       (output) COMPLEX array, dimension (LDW,NB)
            The n-by-nb matrix W required to update the unreduced part
            of A.

    LDW     (input) INTEGER
            The leading dimension of the array W. LDW >= max(1,N).

    Further Details
    ===============
    If UPLO = 'U', the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n) H(n-1) . . . H(n-nb+1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
    and tau in TAU(i-1).

    If UPLO = 'L', the matrix Q is represented as a product of elementary
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

    if UPLO = 'U':                       if UPLO = 'L':

      (  a   a   a   v4  v5 )              (  d                  )
      (      a   a   v4  v5 )              (  1   d              )
      (          a   1   v5 )              (  v1  1   a          )
      (              d   1  )              (  v1  v2  a   a      )
      (                  d  )              (  v1  v2  a   a   a  )

    where d denotes a diagonal element of the reduced matrix, a denotes
    an element of the original matrix that is unchanged, and vi denotes
    an element of the vector defining H(i).
    =====================================================================    */

    char uplo_[2]  = {uplo, 0};

    float mv_time = 0.0;
    magma_int_t i;
#ifndef MAGMABLAS_CHEMV_MGPU
    magma_int_t loffset = nb0*((offset/nb0)/num_gpus);
#endif

    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    magmaFloatComplex value     = MAGMA_C_ZERO;
    magma_int_t id, idw, i_one = 1;

    //magma_int_t kk;
    magma_int_t ione = 1;

    magma_int_t i_n, i_1, iw;

    magmaFloatComplex alpha;

    magmaFloatComplex *dx2[MagmaMaxGPUs];
    magmaFloatComplex *f = (magmaFloatComplex *)malloc(n*sizeof(magmaFloatComplex ));

    if (n <= 0) {
        return 0;
    }

//#define PROFILE_SYMV
#ifdef PROFILE_SYMV
    magma_event_t start, stop;
    float etime;
    magma_timestr_t cpu_start, cpu_end;
    magma_setdevice(0);
    magma_event_create( &start );
    magma_event_create( &stop  );
#endif

    if (lapackf77_lsame(uplo_, "U")) {
        /* Reduce last NB columns of upper triangle */
        for (i = n-1; i >= n - nb ; --i) {
            i_1 = i + 1;
            i_n = n - i - 1;
            iw = i - n + nb;
            if (i < n-1) {
                /* Update A(1:i,i) */
                magmaFloatComplex wii = *W(i, iw+1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_clacgv(&i_one, &wii, &ldw);
                #endif
                wii = -wii;
                blasf77_caxpy(&i_1, &wii, A(0, i+1), &i_one, A(0, i), &ione);

                wii = *A(i, i+1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_clacgv(&i_one, &wii, &ldw);
                #endif
                wii = -wii;
                blasf77_caxpy(&i_1, &wii, W(0, iw+1), &i_one, A(0, i), &ione);
            }
            if (i > 0) {
                /* Generate elementary reflector H(i) to annihilate A(1:i-2,i) */
                alpha = *A(i-1, i);
                lapackf77_clarfg(&i, &alpha, A(0, i), &ione, &tau[i - 1]);

                e[i-1] = MAGMA_C_REAL( alpha );
                *A(i-1,i) = MAGMA_C_MAKE( 1, 0 );
                for( id=0; id<num_gpus; id++ ) {
                    magma_setdevice(id);
                    dx2[id] = dW1(id, 0, iw);
                    magma_csetvector_async( n, A(0,i), 1, dW1(id, 0, iw), 1, stream[id][0]);
#ifndef  MAGMABLAS_CHEMV_MGPU
                    magma_csetvector_async( i, A(0,i), 1, dx[id], 1, stream[id][0] );
#endif
                }
                magmablas_chemv_mgpu(num_gpus, k, 'U', i, nb0, c_one, da, ldda, 0,
                                     dx2, ione, c_zero, dy, ione, dwork, ldwork,
                                     work, W(0, iw), stream );

                if (i < n-1) {
                    blasf77_cgemv(MagmaConjTransStr, &i, &i_n, &c_one, W(0, iw+1), &ldw,
                                  A(0, i), &ione, &c_zero, W(i+1, iw), &ione);
                }

                /* overlap update */
                if( i < n-1 && i-1 >= n - nb )
                {
                    magma_int_t im1_1 = i_1 - 1;
                    magma_int_t im1   = i-1;
                    /* Update A(1:i,i) */
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        magma_int_t im1_n = i_n + 1;
                        lapackf77_clacgv(&im1_n, W(im1, iw+1), &ldw);
                    #endif
                    blasf77_cgemv("No transpose", &im1_1, &i_n, &c_neg_one, A(0, i+1), &lda,
                                  W(im1, iw+1), &ldw, &c_one, A(0, i-1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_clacgv(&im1_n, W(im1, iw+1), &ldw);
                        lapackf77_clacgv(&im1_n, A(im1, i +1), &lda);
                    #endif
                    blasf77_cgemv("No transpose", &im1_1, &i_n, &c_neg_one, W(0, iw+1), &ldw,
                                  A(im1, i+1), &lda, &c_one, A(0, i-1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_clacgv(&im1_n, A(im1, i+1), &lda);
                    #endif
                }

                // 3. Here is where we need it // TODO find the right place
                magmablas_chemv_sync(num_gpus, k, i, work, W(0, iw), stream );

                if (i < n-1) {
                    blasf77_cgemv("No transpose", &i, &i_n, &c_neg_one, A(0, i+1), &lda,
                                  W(i+1, iw), &ione, &c_one, W(0, iw), &ione);

                    blasf77_cgemv(MagmaConjTransStr, &i, &i_n, &c_one, A(0, i+1), &lda,
                                  A(0, i), &ione, &c_zero, W(i+1, iw), &ione);

                    blasf77_cgemv("No transpose", &i, &i_n, &c_neg_one, W(0, iw+1), &ldw,
                                  W(i+1, iw), &ione, &c_one, W(0, iw), &ione);
                }

                blasf77_cscal(&i, &tau[i - 1], W(0, iw), &ione);

                #if defined(PRECISION_z) || defined(PRECISION_c)
                cblas_cdotc_sub( i, W(0,iw), ione, A(0,i), ione, &value );
                #else
                value = cblas_cdotc( i, W(0,iw), ione, A(0,i), ione );
                #endif
                alpha = tau[i - 1] * -.5f * value;
                blasf77_caxpy(&i, &alpha, A(0, i), &ione, W(0, iw), &ione);

                for( id=0; id<num_gpus; id++ ) {
                    magma_setdevice(id);
                    if( k > 1 ) {
                        magma_csetvector_async( n, W(0,iw), 1, dW(id, 0, iw), 1, stream[id][1] );
                    } else {
                        magma_csetvector_async( n, W(0,iw), 1, dW(id, 0, iw), 1, stream[id][0] );
                    }
                }
            }
        }
    } else {
        /*  Reduce first NB columns of lower triangle */
        for (i = 0; i < nb; ++i) {
            /* Update A(i:n,i) */
            i_n = n - i;
            idw = ((offset+i)/nb)%num_gpus;
            if( i > 0 ) {
                trace_cpu_start( 0, "gemv", "gemv" );
                magmaFloatComplex wii = *W(i, i-1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_clacgv(&i_one, &wii, &ldw);
                #endif
                wii = -wii;
                blasf77_caxpy( &i_n, &wii, A(i, i-1), &ione, A(i, i), &ione);

                wii = *A(i, i-1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_clacgv(&i_one, &wii, &lda);
                #endif
                wii = -wii;
                blasf77_caxpy( &i_n, &wii, W(i, i-1), &ione, A(i, i), &ione);
            }

            if (i < n-1) {
                /* Generate elementary reflector H(i) to annihilate A(i+2:n,i) */
                i_n = n - i - 1;
                trace_cpu_start( 0, "larfg", "larfg" );
                alpha = *A(i+1, i);
#ifdef PROFILE_SYMV
                cpu_start = get_current_time();
#endif
                lapackf77_clarfg(&i_n, &alpha, A(min(i+2,n-1), i), &ione, &tau[i]);
#ifdef PROFILE_SYMV
                cpu_end = get_current_time();
                times[0] += GetTimerValue(cpu_start,cpu_end)/1000.0;
#endif
                e[i] = MAGMA_C_REAL( alpha );
                *A(i+1,i) = MAGMA_C_MAKE( 1, 0 );
                trace_cpu_end( 0 );

                /* Compute W(i+1:n,i) */
                // 1. Send the block reflector  A(i+1:n,i) to the GPU
                //trace_gpu_start(  idw, 0, "comm", "comm1" );
#ifndef  MAGMABLAS_CHEMV_MGPU
                magma_setdevice(idw);
                magma_csetvector( i_n, A(i+1,i), 1, dA(idw, i+1, i), 1 );
#endif
                for( id=0; id<num_gpus; id++ ) {
                    magma_setdevice(id);
                    trace_gpu_start( id, 0, "comm", "comm" );
#ifdef MAGMABLAS_CHEMV_MGPU
                    dx2[id] = dW1(id, 0, i)-offset;
#else
                    dx2[id] = dx[id];
                    magma_csetvector( i_n, A(i+1,i), 1, dx[id], 1 );
#endif
                    magma_csetvector_async( n, A(0,i), 1, dW1(id, 0, i), 1, stream[id][0] );
                    trace_gpu_end( id, 0 );
                }
                /* mat-vec on multiple GPUs */
#ifdef PROFILE_SYMV
                magma_setdevice(0);
                magma_event_record(start, stream[0][0]);
#endif
                magmablas_chemv_mgpu(num_gpus, k, 'L', i_n, nb0, c_one, da, ldda, offset+i+1,
                                       dx2, ione, c_zero, dy, ione, dwork, ldwork,
                                       work, W(i+1,i), stream );
#ifdef PROFILE_SYMV
                magma_setdevice(0);
                magma_event_record(stop, stream[0][0]);
#endif
                trace_cpu_start( 0, "gemv", "gemv" );
                blasf77_cgemv(MagmaConjTransStr, &i_n, &i, &c_one, W(i+1, 0), &ldw,
                              A(i+1, i), &ione, &c_zero, W(0, i), &ione);
                blasf77_cgemv("No transpose", &i_n, &i, &c_neg_one, A(i+1, 0), &lda,
                              W(0, i), &ione, &c_zero, f, &ione);
                blasf77_cgemv(MagmaConjTransStr, &i_n, &i, &c_one, A(i+1, 0), &lda,
                              A(i+1, i), &ione, &c_zero, W(0, i), &ione);
                trace_cpu_end( 0 );

                /* overlap update */
                if( i > 0 && i+1 < n )
                {
                    magma_int_t ip1 = i+1;
                    trace_cpu_start( 0, "gemv", "gemv" );
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_clacgv(&i, W(ip1, 0), &ldw);
                    #endif
                    blasf77_cgemv("No transpose", &i_n, &i, &c_neg_one, A(ip1, 0), &lda,
                                  W(ip1, 0), &ldw, &c_one, A(ip1, ip1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_clacgv(&i, W(ip1, 0), &ldw);
                        lapackf77_clacgv(&i, A(ip1 ,0), &lda);
                    #endif
                    blasf77_cgemv("No transpose", &i_n, &i, &c_neg_one, W(ip1, 0), &ldw,
                                  A(ip1, 0), &lda, &c_one, A(ip1, ip1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_clacgv(&i, A(ip1, 0), &lda);
                    #endif
                    trace_cpu_end( 0 );
                }

                /* synchronize */
                magmablas_chemv_sync(num_gpus, k, i_n, work, W(i+1,i), stream );
#ifdef PROFILE_SYMV
                cudaEventElapsedTime(&etime, start, stop);
                mv_time += (etime/1000.0);
                times[1+(i_n/(n0/10))] += (etime/1000.0);
#endif
                trace_cpu_start( 0, "axpy", "axpy" );
                if (i!=0)
                    blasf77_caxpy(&i_n, &c_one, f, &ione, W(i+1, i), &ione);

                blasf77_cgemv("No transpose", &i_n, &i, &c_neg_one, W(i+1, 0), &ldw,
                              W(0, i), &ione, &c_one, W(i+1, i), &ione);
                blasf77_cscal(&i_n, &tau[i], W(i+1,i), &ione);

                #if defined(PRECISION_z) || defined(PRECISION_c)
                    cblas_cdotc_sub( i_n, W(i+1,i), ione, A(i+1,i), ione, &value );
                #else
                    value = cblas_cdotc( i_n, W(i+1,i), ione, A(i+1,i), ione );
                #endif
                alpha = tau[i]* -.5f * value;
                blasf77_caxpy(&i_n, &alpha, A(i+1, i), &ione, W(i+1,i), &ione);
                trace_cpu_end( 0 );
                for( id=0; id<num_gpus; id++ ) {
                    magma_setdevice(id);
                    if( k > 1 ) {
                        magma_csetvector_async( n, W(0,i), 1, dW(id, 0, i), 1, stream[id][1] );
                    } else {
                        magma_csetvector_async( n, W(0,i), 1, dW(id, 0, i), 1, stream[id][0] );
                    }
                }
            }
        }
    }

#ifdef PROFILE_SYMV
    magma_setdevice(0);
    magma_event_destory( start );
    magma_event_destory( stop  );
#endif
    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        if( k > 1) magma_queue_sync(stream[id][1]);
    }
    free(f);

    return mv_time;
} /* clatrd_ */

extern "C"
magma_int_t
magmablas_chemv_mgpu( magma_int_t num_gpus, magma_int_t k, char uplo,
                      magma_int_t n, magma_int_t nb,
                      magmaFloatComplex alpha,
                      magmaFloatComplex **da, magma_int_t ldda, magma_int_t offset,
                      magmaFloatComplex **dx, magma_int_t incx,
                      magmaFloatComplex beta,
                      magmaFloatComplex **dy, magma_int_t incy,
                      magmaFloatComplex **dwork, magma_int_t ldwork,
                      magmaFloatComplex *work, magmaFloatComplex *w,
                      magma_queue_t stream[][10] )
{

#define dX(id, i)    (dx[(id)]+incx*(i))
#define dY(id, i, j) (dy[(id)]+incy*(i)+n*(j))

    magma_int_t id;

#ifdef MAGMABLAS_CHEMV_MGPU
    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        magmablasSetKernelStream(stream[id][0]);
        trace_gpu_start( id, 0, "memset", "memset" );
        cudaMemset( dwork[id], 0, ldwork*sizeof(magmaFloatComplex) );
        trace_gpu_end( id, 0 );
        trace_gpu_start( id, 0, "symv", "symv" );
    }

    if( nb == 32 ) {
        magmablas_chemv_mgpu_32_offset( uplo, offset+n, alpha, da, ldda,
                                        dx, incx,
                                        beta,
                                        dy, incy,
                                        dwork, ldwork,
                                        num_gpus, nb, offset,
                                        stream );
    } else {
        magmablas_chemv_mgpu_offset( uplo, offset+n, alpha, da, ldda,
                                     dx, incx,
                                     beta,
                                     dy, incy,
                                     dwork, ldwork,
                                     num_gpus, nb, offset,
                                     stream );
    }
    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        trace_gpu_end( id, 0 );
        magmablasSetKernelStream(NULL);
    }
    //magma_setdevice(0);
    //magmablasSetKernelStream(stream[0][0]);
    //magma_chemv('L', n, alpha, &da[0][offset+offset*ldda], ldda, &dx[0][offset], incx, beta, &dy[0][offset], incy );
    //magmablasSetKernelStream(NULL);

    /* send to CPU */
    magma_setdevice(0);
    trace_gpu_start( 0, 0, "comm", "comm" );
    magma_cgetvector_async( n, dY(0, offset, 0), 1, w, 1, stream[0][0] );
    trace_gpu_end( 0, 0 );
    magmablasSetKernelStream(NULL);

    for( id=1; id<num_gpus; id++ ) {
        magma_setdevice(id);
        trace_gpu_start(  id, 0, "comm", "comm" );
        magma_cgetvector_async( n, dY(id, offset, 0), 1, &work[id*n], 1, stream[id][0] );
        trace_gpu_end( id, 0 );
        magmablasSetKernelStream(NULL);
    }
#else
    magmaFloatComplex c_one = MAGMA_C_ONE;
    char uplo_[2]  = {uplo, 0};
    magma_int_t i, ii, j, kk, ib, ib0, i_1, i_local, idw;
    magma_int_t i_0=n;
    magma_int_t loffset0 = nb*(offset/(nb*num_gpus));
    magma_int_t loffset1 = offset%nb;
    magma_int_t loffset;    
    
    //magma_chemv(uplo, n, alpha, da, ldda, dx, incx, beta, dy, incy );

    idw = (offset/nb)%num_gpus;

    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        magmablasSetKernelStream(stream[id][0]);
        cudaMemset( dy[id], 0, n*k*sizeof(magmaFloatComplex) );
    }

    if( lapackf77_lsame( uplo_, "L" ) ) {
        /* the first block */
        if( loffset1 > 0 ) {
            id = idw;
            kk = 0;

            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);

            loffset = loffset0+loffset1;
            ib0 = min(nb-loffset1,n);
            // diagonal
            magma_chemv(MagmaLower, ib0, c_one, dA(id, 0, 0 ), ldda,
                        dX(id, 0), incx, c_one, dY(id, 0, kk), incy);
            // off-diagonl
            if( ib0 < n ) {
                for( j=ib0; j<n; j+= i_0 ) {
                    i_1 = min(i_0, n-j);
                    magma_cgemv(MagmaNoTrans, i_1, ib0, c_one, dA(id, j, 0), ldda,
                                dX(id, 0), incx, c_one, dY(id, j, kk), incy);
                    magma_cgemv(MagmaConjTrans, i_1, ib0, c_one, dA(id, j, 0), ldda,
                                dX(id, j), incx, c_one, dY(id, 0, kk), incy);
                }
            }
        }
        else {
            ib0 = 0;
        }

        /* diagonal */
        for( i=ib0; i<n; i+=nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = ((i+loffset1)/(nb*num_gpus))%k;

            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);

            i_local = (i+loffset1)/(nb*num_gpus);
            ib = min(nb,n-i);

            ii = nb*i_local;

            loffset = loffset0;
            if( id < idw ) loffset += nb;
            magma_chemv(MagmaLower,  ib, c_one, dA(id, i, ii), ldda,
                        dX(id, i), incx, c_one, dY(id, i, kk), incy);
        }

        /* off-diagonal */
        for( i=ib0; i<n-nb; i+=nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = ((i+loffset1)/(nb*num_gpus))%k;
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);

            i_local = ((i+loffset1)/nb)/num_gpus;
            ii = nb*i_local;
            ib = min(nb,n-i);
            loffset = loffset0;
            if( id < idw ) loffset += nb;

            for( j=i+ib; j<n; j+= i_0 ) {
                i_1 = min(i_0, n-j);
                magma_cgemv(MagmaNoTrans, i_1, ib, c_one, dA(id, j, ii), ldda,
                            dX(id, i), incx, c_one, dY(id, j, kk), incy);
                magma_cgemv(MagmaConjTrans, i_1, ib, c_one, dA(id, j, ii), ldda,
                            dX(id, j), incx, c_one, dY(id, i, kk), incy);
            }
        }
    } else { /* upper-triangular storage */
        loffset = 0;
        /* diagonal */
        for( i=0; i<n; i+=nb ) {
            id = (i/nb)%num_gpus;
            kk = (i/(nb*num_gpus))%k;
            ib = min(nb,n-i);

            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);

            i_local = i/(nb*num_gpus);
            ii = nb*i_local;

            magma_chemv(MagmaUpper, ib, c_one, dA(id, i, ii), ldda,
                        dX(id, i), incx, c_one, dY(id, i, kk), incy);
        }

        /* off-diagonal */
        for( i=nb; i<n; i+=nb ) {
            id = (i/nb)%num_gpus;
            kk = (i/(nb*num_gpus))%k;
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);

            i_local = (i/nb)/num_gpus;
            ii = nb*i_local;
            ib = min(nb,n-i);

            magma_cgemv(MagmaNoTrans, i, ib, c_one, dA(id, 0, ii), ldda,
                        dX(id, i), incx, c_one, dY(id, 0, kk), incy);
            magma_cgemv(MagmaConjTrans, i, ib, c_one, dA(id, 0, ii), ldda,
                        dX(id, 0), incx, c_one, dY(id, i, kk), incy);
        }
    }
    /* send to CPU */
    magma_setdevice(0);
    magma_cgetvector_async( n, dY(0, 0, 0), 1, w, 1, stream[0][0] );
    for( kk=1; kk<k; kk++ ) {
        magma_cgetvector_async( n, dY(0, 0, kk), 1, &work[kk*n], 1, stream[0][kk] );
    }
    magmablasSetKernelStream(NULL);

    for( id=1; id<num_gpus; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk<k; kk++ ) {
            magma_cgetvector_async( n, dY(id, 0, kk), 1, &work[id*k*n + kk*n], 1, stream[id][kk] );
        }
        magmablasSetKernelStream(NULL);
    }
#endif
    return 0;
}

extern "C"
magma_int_t
magmablas_chemv_sync( magma_int_t num_gpus, magma_int_t k,
                      magma_int_t n, magmaFloatComplex *work, magmaFloatComplex *w,
                      magma_queue_t stream[][10] )
{

    magmaFloatComplex c_one = MAGMA_C_ONE;
    magma_int_t ione = 1;
    magma_int_t id, kk;

    /* reduce on CPU */
    magma_setdevice(0);
    magma_queue_sync(stream[0][0]);
    for( kk=1; kk<k; kk++ ) {
        magma_queue_sync(stream[0][kk]);
        blasf77_caxpy( &n, &c_one, &work[kk*n], &ione, w, &ione );
    }
    for( id=1; id<num_gpus; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk<k; kk++ ) {
            magma_queue_sync(stream[id][kk]);
            blasf77_caxpy( &n, &c_one, &work[id*k*n + kk*n], &ione, w, &ione );
        }
    }

    return 0;
}
