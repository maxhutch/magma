/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @author Azzam Haidar

       @generated from dtrevc3_mt.cpp normal d -> s, Fri Jan 30 19:00:19 2015
*/
#include "thread_queue.hpp"
#include "magma_timer.h"

#include "common_magma.h"  // after thread.hpp, so max, min are defined

#define REAL

// ---------------------------------------------
// stores arguments and executes call to slaqtrsd (on CPU)
class magma_slaqtrsd_task: public magma_task
{
public:
    magma_slaqtrsd_task(
        magma_trans_t in_trans, magma_int_t in_n,
        const float *in_T, magma_int_t in_ldt,
        float       *in_x, magma_int_t in_incx,
        const float *in_cnorm
    ):
        trans( in_trans ),
        n    ( in_n     ),
        T    ( in_T     ),
        ldt  ( in_ldt   ),
        x    ( in_x     ),
        incx ( in_incx  ),
        cnorm( in_cnorm )
    {}
    
    virtual void run()
    {
        magma_int_t info = 0;
        magma_slaqtrsd( trans, n, T, ldt, x, incx, cnorm, &info );
        if ( info != 0 ) {
            fprintf( stderr, "slaqtrsd info %d\n", (int) info );
        }
    }
    
private:
    magma_trans_t trans;
    magma_int_t   n;
    const float *T;
    magma_int_t   ldt;
    float       *x;
    magma_int_t   incx;
    const float *cnorm;
};


// ---------------------------------------------
// stores arguments and executes call to sgemm (on CPU)
// todo - better to store magma_trans_t and use lapack_trans_const, since there
// is no guarantee that the char* string is still valid when the task gets executed.
class sgemm_task: public magma_task
{
public:
    sgemm_task(
        magma_trans_t in_transA, magma_trans_t in_transB,
        magma_int_t in_m, magma_int_t in_n, magma_int_t in_k,
        float  in_alpha,
        const float *in_A, magma_int_t in_lda,
        const float *in_B, magma_int_t in_ldb,
        float  in_beta,
        float *in_C, magma_int_t in_ldc
    ):
        transA( in_transA ),
        transB( in_transB ),
        m     ( in_m      ),
        n     ( in_n      ),
        k     ( in_k      ),
        alpha ( in_alpha  ),
        A     ( in_A      ),
        lda   ( in_lda    ),
        B     ( in_B      ),
        ldb   ( in_ldb    ),
        beta  ( in_beta   ),
        C     ( in_C      ),
        ldc   ( in_ldc    )
    {}
    
    virtual void run()
    {
        blasf77_sgemm( lapack_trans_const(transA), lapack_trans_const(transB),
                       &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    }
    
private:
    magma_trans_t transA;
    magma_trans_t transB;
    magma_int_t   m;
    magma_int_t   n;
    magma_int_t   k;
    float        alpha;
    const float *A;
    magma_int_t   lda;
    const float *B;
    magma_int_t   ldb;
    float        beta;
    float       *C;
    magma_int_t   ldc;
};


/**
    Purpose
    -------
    STREVC3_MT computes some or all of the right and/or left eigenvectors of
    a real upper quasi-triangular matrix T.
    Matrices of this type are produced by the Schur factorization of
    a real general matrix:  A = Q*T*Q**T, as computed by SHSEQR.

    The right eigenvector x and the left eigenvector y of T corresponding
    to an eigenvalue w are defined by:

       T*x = w*x,     (y**T)*T = w*(y**T)

    where y**T denotes the transpose of the vector y.
    The eigenvalues are not input to this routine, but are read directly
    from the diagonal blocks of T.

    This routine returns the matrices X and/or Y of right and left
    eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
    input matrix. If Q is the orthogonal factor that reduces a matrix
    A to Schur form T, then Q*X and Q*Y are the matrices of right and
    left eigenvectors of A.

    This uses a Level 3 BLAS version of the back transformation.
    This uses a multi-threaded (mt) implementation.

    Arguments
    ---------
    @param[in]
    side     magma_side_t
       -     = MagmaRight:      compute right eigenvectors only;
       -     = MagmaLeft:       compute left eigenvectors only;
       -     = MagmaBothSides:  compute both right and left eigenvectors.

    @param[in]
    howmany  magma_vec_t
       -     = MagmaAllVec:        compute all right and/or left eigenvectors;
       -     = MagmaBacktransVec:  compute all right and/or left eigenvectors,
                     backtransformed by the matrices in VR and/or VL;
       -     = MagmaSomeVec:       compute selected right and/or left eigenvectors,
                     as indicated by the logical array select.

    @param[in,out]
    select   LOGICAL array, dimension (n)
             If howmany = MagmaSomeVec, select specifies the eigenvectors to be
             computed.
             If w(j) is a real eigenvalue, the corresponding real
             eigenvector is computed if select(j) is true.
             If w(j) and w(j+1) are the real and imaginary parts of a
             complex eigenvalue, the corresponding complex eigenvector is
             computed if either select(j) or select(j+1) is true, and
             on exit select(j) is set to true and select(j+1) is set to
             false.
             Not referenced if howmany = MagmaAllVec or MagmaBacktransVec.

    @param[in]
    n        INTEGER
             The order of the matrix T. n >= 0.

    @param[in]
    T        REAL array, dimension (ldt,n)
             The upper quasi-triangular matrix T in Schur canonical form.

    @param[in]
    ldt      INTEGER
             The leading dimension of the array T. ldt >= max(1,n).

    @param[in,out]
    VL       REAL array, dimension (ldvl,mm)
             On entry, if side = MagmaLeft or MagmaBothSides and howmany = MagmaBacktransVec, VL must
             contain an n-by-n matrix Q (usually the orthogonal matrix Q
             of Schur vectors returned by SHSEQR).
             On exit, if side = MagmaLeft or MagmaBothSides, VL contains:
             if howmany = MagmaAllVec, the matrix Y of left eigenvectors of T;
             if howmany = MagmaBacktransVec, the matrix Q*Y;
             if howmany = MagmaSomeVec, the left eigenvectors of T specified by
                               select, stored consecutively in the columns
                               of VL, in the same order as their
                               eigenvalues.
             A complex eigenvector corresponding to a complex eigenvalue
             is stored in two consecutive columns, the first holding the
             real part, and the second the imaginary part.
             Not referenced if side = MagmaRight.

    @param[in]
    ldvl     INTEGER
             The leading dimension of the array VL.
             ldvl >= 1, and if side = MagmaLeft or MagmaBothSides, ldvl >= n.

    @param[in,out]
    VR       REAL array, dimension (ldvr,mm)
             On entry, if side = MagmaRight or MagmaBothSides and howmany = MagmaBacktransVec, VR must
             contain an n-by-n matrix Q (usually the orthogonal matrix Q
             of Schur vectors returned by SHSEQR).
             On exit, if side = MagmaRight or MagmaBothSides, VR contains:
             if howmany = MagmaAllVec, the matrix X of right eigenvectors of T;
             if howmany = MagmaBacktransVec, the matrix Q*X;
             if howmany = MagmaSomeVec, the right eigenvectors of T specified by
                               select, stored consecutively in the columns
                               of VR, in the same order as their
                               eigenvalues.
             A complex eigenvector corresponding to a complex eigenvalue
             is stored in two consecutive columns, the first holding the
             real part and the second the imaginary part.
             Not referenced if side = MagmaLeft.

    @param[in]
    ldvr     INTEGER
             The leading dimension of the array VR.
             ldvr >= 1, and if side = MagmaRight or MagmaBothSides, ldvr >= n.

    @param[in]
    mm       INTEGER
             The number of columns in the arrays VL and/or VR. mm >= mout.

    @param[out]
    mout     INTEGER
             The number of columns in the arrays VL and/or VR actually
             used to store the eigenvectors.
             If howmany = MagmaAllVec or MagmaBacktransVec, mout is set to n.
             Each selected real eigenvector occupies one column and each
             selected complex eigenvector occupies two columns.

    @param
    work     (workspace) REAL array, dimension (max(1,lwork))

    @param[in]
    lwork    INTEGER
             The dimension of array work. lwork >= max(1,3*n).
             For optimum performance, lwork >= (1 + 2*nb)*n, where nb is
             the optimal blocksize.

    @param[out]
    info     INTEGER
       -     = 0:  successful exit
       -     < 0:  if info = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The algorithm used in this program is basically backward (forward)
    substitution, with scaling to make the the code robust against
    possible overflow.

    Each eigenvector is normalized so that the element of largest
    magnitude has magnitude 1; here the magnitude of a complex number
    (x,y) is taken to be |x| + |y|.

    @ingroup magma_sgeev_comp
    ********************************************************************/
extern "C"
magma_int_t magma_strevc3_mt(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select,  // logical in fortran
    magma_int_t n,
    float *T,  magma_int_t ldt,
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info )
{
#define T(i,j)  (T  + (i) + (j)*ldt)
#define VL(i,j) (VL + (i) + (j)*ldvl)
#define VR(i,j) (VR + (i) + (j)*ldvr)
#define X(i,j)  (X  + (i)-1 + ((j)-1)*2)  // still as 1-based indices
#define work(i,j) (work + (i) + (j)*n)

    // constants
    const magma_int_t ione = 1;
    const float c_zero = 0;
    const float c_one  = 1;
    const magma_int_t nbmin = 16, nbmax = 256;
    
    // .. Local Scalars ..
    magma_int_t allv, bothv, leftv, over, pair, rightv, somev;
    magma_int_t i, ii, ip, is, j, k, ki, ki2,
                iv, n2, nb, nb2, version;
    float emax, remax;
    
    // .. Local Arrays ..
    // since iv is a 1-based index, allocate one extra here
    magma_int_t iscomplex[ nbmax+1 ];

    // Decode and test the input parameters
    bothv  = (side == MagmaBothSides);
    rightv = (side == MagmaRight) || bothv;
    leftv  = (side == MagmaLeft ) || bothv;

    allv  = (howmany == MagmaAllVec);
    over  = (howmany == MagmaBacktransVec);
    somev = (howmany == MagmaSomeVec);

    *info = 0;
    if ( ! rightv && ! leftv )
        *info = -1;
    else if ( ! allv && ! over && ! somev )
        *info = -2;
    else if ( n < 0 )
        *info = -4;
    else if ( ldt < max( 1, n ) )
        *info = -6;
    else if ( ldvl < 1 || ( leftv && ldvl < n ) )
        *info = -8;
    else if ( ldvr < 1 || ( rightv && ldvr < n ) )
        *info = -10;
    else if ( lwork < max( 1, 3*n ) )
        *info = -14;
    else {
        // Set mout to the number of columns required to store the selected
        // eigenvectors, standardize the array select if necessary, and
        // test mm.
        if ( somev ) {
            *mout = 0;
            pair = false;
            for( j=0; j < n; ++j ) {
                if ( pair ) {
                    pair = false;
                    select[j] = false;
                }
                else {
                    if ( j < n-1 ) {
                        if ( *T(j+1,j) == c_zero ) {
                            if ( select[j] ) {
                                *mout += 1;
                            }
                        }
                        else {
                            pair = true;
                            if ( select[j] || select[j+1] ) {
                                select[j] = true;
                                *mout += 2;
                            }
                        }
                    }
                    else if ( select[n-1] ) {
                        *mout += 1;
                    }
                }
            }
        }
        else {
            *mout = n;
        }
        if ( mm < *mout ) {
            *info = -11;
        }
    }
    
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible.
    if ( n == 0 ) {
        return *info;
    }
    
    // Use blocked version (2) if sufficient workspace.
    // Requires 1 vector for 1-norms, and 2*nb vectors for x and Q*x.
    // Zero-out the workspace to avoid potential NaN propagation.
    nb = 2;
    if ( lwork >= n + 2*n*nbmin ) {
        version = 2;
        nb = (lwork - n) / (2*n);
        nb = min( nb, nbmax );
        nb2 = 1 + 2*nb;
        lapackf77_slaset( "F", &n, &nb2, &c_zero, &c_zero, work, &n );
    }
    else {
        version = 1;
    }

    // Compute 1-norm of each column of strictly upper triangular
    // part of T to control overflow in triangular solver.
    *work(0,0) = c_zero;
    for( j=1; j < n; ++j ) {
        *work(j,0) = c_zero;
        for( i=0; i < j; ++i ) {
            *work(j,0) += fabsf( *T(i,j) );
        }
    }
        
    // launch threads -- each single-threaded MKL
    magma_int_t nthread = magma_get_parallel_numthreads();
    magma_int_t lapack_nthread = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads( 1 );
    magma_thread_queue queue;
    queue.launch( nthread );
    //printf( "nthread %d, %d\n", nthread, lapack_nthread );
    
    // gemm_nb = N/thread, rounded up to multiple of 16,
    // but avoid multiples of page size, e.g., 512*8 bytes = 4096.
    magma_int_t gemm_nb = magma_int_t( ceil( ceil( ((float)n) / nthread ) / 16. ) * 16. );
    if ( gemm_nb % 512 == 0 ) {
        gemm_nb += 32;
    }
    
    magma_timer_t time_total=0, time_trsv=0, time_gemm=0, time_gemv=0, time_trsv_sum=0, time_gemm_sum=0, time_gemv_sum=0;
    timer_start( time_total );

    // Index ip is used to specify the real or complex eigenvalue:
    // ip =  0, real eigenvalue (wr),
    //    =  1, first  of conjugate complex pair: (wr,wi)
    //    = -1, second of conjugate complex pair: (wr,wi)
    // iscomplex array stores ip for each column in current block.
    if ( rightv ) {
        // ============================================================
        // Compute right eigenvectors.
        // iv is index of column in current block (1-based).
        // For complex right vector, uses iv-1 for real part and iv for complex part.
        // Non-blocked version always uses iv=2;
        // blocked     version starts with iv=nb, goes down to 1 or 2.
        // (Note the "0-th" column is used for 1-norms computed above.)
        iv = 2;
        if ( version == 2 ) {
            iv = nb;
        }

        timer_start( time_trsv );
        ip = 0;
        is = *mout - 1;
        for( ki=n-1; ki >= 0; --ki ) {
            if ( ip == -1 ) {
                // previous iteration (ki+1) was second of conjugate pair,
                // so this ki is first of conjugate pair; skip to end of loop
                ip = 1;
                continue;
            }
            else if ( ki == 0 ) {
                // last column, so this ki must be real eigenvalue
                ip = 0;
            }
            else if ( *T(ki,ki-1) == c_zero ) {
                // zero on sub-diagonal, so this ki is real eigenvalue
                ip = 0;
            }
            else {
                // non-zero on sub-diagonal, so this ki is second of conjugate pair
                ip = -1;
            }

            if ( somev ) {
                if ( ip == 0 ) {
                    if ( ! select[ki] ) {
                        continue;
                    }
                }
                else {
                    if ( ! select[ki-1] ) {
                        continue;
                    }
                }
            }

            if ( ip == 0 ) {
                // ------------------------------------------------------------
                // Real right eigenvector
                // Solve upper quasi-triangular system:
                // [ T(0:ki-1,0:ki-1) - wr ]*X = -T(0:ki-1,ki)
                queue.push_task( new magma_slaqtrsd_task(
                    MagmaNoTrans, ki+1, T(0,0), ldt, work(0,iv), n, work(0,0) ));
                
                // Copy the vector x or Q*x to VR and normalize.
                if ( ! over ) {
                    // ------------------------------
                    // no back-transform: copy x to VR and normalize.
                    queue.sync();
                    n2 = ki+1;
                    blasf77_scopy( &n2, work(0,iv), &ione, VR(0,is), &ione );

                    ii = blasf77_isamax( &n2, VR(0,is), &ione ) - 1;  // subtract 1; ii is 0-based
                    remax = c_one / fabsf( *VR(ii,is) );
                    blasf77_sscal( &n2, &remax, VR(0,is), &ione );

                    for( k=ki + 1; k < n; ++k ) {
                        *VR(k,is) = c_zero;
                    }
                }
                else if ( version == 1 ) {
                    // ------------------------------
                    // version 1: back-transform each vector with GEMV, Q*x.
                    queue.sync();
                    time_trsv_sum += timer_stop( time_trsv );
                    timer_start( time_gemv );
                    if ( ki > 0 ) {
                        n2 = ki;
                        blasf77_sgemv( "n", &n, &n2, &c_one,
                                       VR, &ldvr,
                                       work(0, iv), &ione,
                                       work(ki,iv), VR(0,ki), &ione );
                    }
                    time_gemv_sum += timer_stop( time_gemv );
                    ii = blasf77_isamax( &n, VR(0,ki), &ione ) - 1;  // subtract 1; ii is 0-based
                    remax = c_one / fabsf( *VR(ii,ki) );
                    blasf77_sscal( &n, &remax, VR(0,ki), &ione );
                    timer_start( time_trsv );
                }
                else if ( version == 2 ) {
                    // ------------------------------
                    // version 2: back-transform block of vectors with GEMM
                    // zero out below vector
                    for( k=ki + 1; k < n; ++k ) {
                        *work(k,iv) = c_zero;
                    }
                    iscomplex[ iv ] = ip;
                    // back-transform and normalization is done below
                }
            }  // end real eigenvector
            else {
                // ------------------------------------------------------------
                // Complex right eigenvector
                // Solve upper quasi-triangular system:
                // [ T(0:ki-2,0:ki-2) - (wr+i*wi) ]*x = u
                queue.push_task( new magma_slaqtrsd_task(
                    MagmaNoTrans, ki+1, T(0,0), ldt, work(0,iv-1), n, work(0,0) ));

                // Copy the vector x or Q*x to VR and normalize.
                if ( ! over ) {
                    // ------------------------------
                    // no back-transform: copy x to VR and normalize.
                    queue.sync();
                    n2 = ki+1;
                    blasf77_scopy( &n2, work(0,iv-1), &ione, VR(0,is-1), &ione );
                    blasf77_scopy( &n2, work(0,iv  ), &ione, VR(0,is  ), &ione );

                    emax = c_zero;
                    for( k=0; k <= ki; ++k ) {
                        emax = max( emax, fabsf(*VR(k,is-1)) + fabsf(*VR(k,is)) );
                    }
                    remax = c_one / emax;
                    blasf77_sscal( &n2, &remax, VR(0,is-1), &ione );
                    blasf77_sscal( &n2, &remax, VR(0,is  ), &ione );

                    for( k=ki + 1; k < n; ++k ) {
                        *VR(k,is-1) = c_zero;
                        *VR(k,is  ) = c_zero;
                    }
                }
                else if ( version == 1 ) {
                    // ------------------------------
                    // version 1: back-transform each vector with GEMV, Q*x.
                    queue.sync();
                    time_trsv_sum += timer_stop( time_trsv );
                    timer_start( time_gemv );
                    if ( ki > 1 ) {
                        n2 = ki-1;
                        blasf77_sgemv( "n", &n, &n2, &c_one,
                                       VR, &ldvr,
                                       work(0,   iv-1), &ione,
                                       work(ki-1,iv-1), VR(0,ki-1), &ione );
                        blasf77_sgemv( "n", &n, &n2, &c_one,
                                       VR, &ldvr,
                                       work(0, iv), &ione,
                                       work(ki,iv), VR(0,ki), &ione );
                    }
                    else {
                        blasf77_sscal( &n, work(ki-1,iv-1), VR(0,ki-1), &ione );
                        blasf77_sscal( &n, work(ki,  iv  ), VR(0,ki  ), &ione );
                    }
                    time_gemv_sum += timer_stop( time_gemv );

                    emax = c_zero;
                    for( k=0; k < n; ++k ) {
                        emax = max( emax, fabsf(*VR(k,ki-1)) + fabsf(*VR(k,ki)) );
                    }
                    remax = c_one / emax;
                    blasf77_sscal( &n, &remax, VR(0,ki-1), &ione );
                    blasf77_sscal( &n, &remax, VR(0,ki  ), &ione );
                    timer_start( time_trsv );
                }
                else if ( version == 2 ) {
                    // ------------------------------
                    // version 2: back-transform block of vectors with GEMM
                    // zero out below vector
                    for( k=ki + 1; k < n; ++k ) {
                        *work(k,iv-1) = c_zero;
                        *work(k,iv  ) = c_zero;
                    }
                    iscomplex[ iv-1 ] = -ip;
                    iscomplex[ iv   ] =  ip;
                    iv -= 1;
                    // back-transform and normalization is done below
                }
            }  // end real or complex vector

            if ( version == 2 ) {
                // ------------------------------------------------------------
                // Blocked version of back-transform
                // For complex case, ki2 includes both vectors (ki-1 and ki)
                if ( ip == 0 ) {
                    ki2 = ki;
                }
                else {
                    ki2 = ki - 1;
                }

                // Columns iv:nb of work are valid vectors.
                // When the number of vectors stored reaches nb-1 or nb,
                // or if this was last vector, do the GEMM
                if ( (iv <= 2) || (ki2 == 0) ) {
                    queue.sync();
                    time_trsv_sum += timer_stop( time_trsv );
                    timer_start( time_gemm );
                    nb2 = nb-iv+1;
                    n2  = ki2+nb-iv+1;
                    
                    // split gemm into multiple tasks, each doing one block row
                    for( i=0; i < n; i += gemm_nb ) {
                        magma_int_t ib = min( gemm_nb, n-i );
                        queue.push_task( new sgemm_task(
                            MagmaNoTrans, MagmaNoTrans, ib, nb2, n2, c_one,
                            VR(i,0), ldvr,
                            work(0,iv), n, c_zero,
                            work(i,nb+iv), n ));
                    }
                    queue.sync();
                    time_gemm_sum += timer_stop( time_gemm );

                    // normalize vectors
                    // TODO if somev, should copy vectors individually to correct location.
                    for( k=iv; k <= nb; ++k ) {
                        if ( iscomplex[k] == 0 ) {
                            // real eigenvector
                            ii = blasf77_isamax( &n, work(0,nb+k), &ione ) - 1;  // subtract 1; ii is 0-based
                            remax = c_one / fabsf( *work(ii,nb+k) );
                        }
                        else if ( iscomplex[k] == 1 ) {
                            // first eigenvector of conjugate pair
                            emax = c_zero;
                            for( ii=0; ii < n; ++ii ) {
                                emax = max( emax, fabsf( *work(ii,nb+k  ) )
                                                + fabsf( *work(ii,nb+k+1) ) );
                            }
                            remax = c_one / emax;
                        // else if iscomplex[k] == -1
                        //     second eigenvector of conjugate pair
                        //     reuse same remax as previous k
                        }
                        blasf77_sscal( &n, &remax, work(0,nb+k), &ione );
                    }
                    nb2 = nb-iv+1;
                    lapackf77_slacpy( "F", &n, &nb2,
                                      work(0,nb+iv), &n,
                                      VR(0,ki2), &ldvr );
                    iv = nb;
                    timer_start( time_trsv );
                }
                else {
                    iv -= 1;
                }
            }  // end blocked back-transform

            is -= 1;
            if ( ip != 0 ) {
                is -= 1;
            }
        }
    }
    timer_stop( time_trsv );
    
    timer_stop( time_total );
    timer_printf( "trevc trsv %.4f, gemm %.4f, gemv %.4f, total %.4f\n",
                  time_trsv_sum, time_gemm_sum, time_gemv_sum, time_total );

    if ( leftv ) {
        // ============================================================
        // Compute left eigenvectors.
        // iv is index of column in current block (1-based).
        // For complex left vector, uses iv for real part and iv+1 for complex part.
        // Non-blocked version always uses iv=1;
        // blocked     version starts with iv=1, goes up to nb-1 or nb.
        // (Note the "0-th" column is used for 1-norms computed above.)
        iv = 1;
        ip = 0;
        is = 0;
        for( ki=0; ki < n; ++ki ) {
            if ( ip == 1 ) {
                // previous iteration (ki-1) was first of conjugate pair,
                // so this ki is second of conjugate pair; skip to end of loop
                ip = -1;
                continue;
            }
            else if ( ki == n-1 ) {
                // last column, so this ki must be real eigenvalue
                ip = 0;
            }
            else if ( *T(ki+1,ki) == c_zero ) {
                // zero on sub-diagonal, so this ki is real eigenvalue
                ip = 0;
            }
            else {
                // non-zero on sub-diagonal, so this ki is first of conjugate pair
                ip = 1;
            }
    
            if ( somev ) {
                if ( ! select[ki] ) {
                    continue;
                }
            }
    
            if ( ip == 0 ) {
                // ------------------------------------------------------------
                // Real left eigenvector
                // Solve transposed quasi-triangular system:
                // [ T(ki+1:n,ki+1:n) - wr ]**T * X = -T(ki+1:n,ki)
                queue.push_task( new magma_slaqtrsd_task(
                    MagmaTrans, n-ki, T(ki,ki), ldt, work(ki,iv), n, work(ki,0) ));
    
                // Copy the vector x or Q*x to VL and normalize.
                if ( ! over ) {
                    // ------------------------------
                    // no back-transform: copy x to VL and normalize.
                    queue.sync();
                    n2 = n-ki;
                    blasf77_scopy( &n2, work(ki,iv), &ione, VL(ki,is), &ione );
    
                    ii = blasf77_isamax( &n2, VL(ki,is), &ione ) + ki - 1;  // subtract 1; ii is 0-based
                    remax = c_one / fabsf( *VL(ii,is) );
                    blasf77_sscal( &n2, &remax, VL(ki,is), &ione );
    
                    for( k=0; k < ki; ++k ) {
                        *VL(k,is) = c_zero;
                    }
                }
                else if ( version == 1 ) {
                    // ------------------------------
                    // version 1: back-transform each vector with GEMV, Q*x.
                    queue.sync();
                    if ( ki < n-1 ) {
                        n2 = n-ki-1;
                        blasf77_sgemv( "n", &n, &n2, &c_one,
                                       VL(0,ki+1), &ldvl,
                                       work(ki+1,iv), &ione,
                                       work(ki,  iv), VL(0,ki), &ione );
                    }
                    ii = blasf77_isamax( &n, VL(0,ki), &ione ) - 1;  // subtract 1; ii is 0-based
                    remax = c_one / fabsf( *VL(ii,ki) );
                    blasf77_sscal( &n, &remax, VL(0,ki), &ione );
                }
                else if ( version == 2 ) {
                    // ------------------------------
                    // version 2: back-transform block of vectors with GEMM
                    // zero out above vector
                    // could go from (ki+1)-NV+1 to ki
                    for( k=0; k < ki; ++k ) {
                        *work(k,iv) = c_zero;
                    }
                    iscomplex[ iv ] = ip;
                    // back-transform and normalization is done below
                }
            }  // end real eigenvector
            else {
                // ------------------------------------------------------------
                // Complex left eigenvector
                // Solve transposed quasi-triangular system:
                // [ T(ki+2:n,ki+2:n)**T - (wr-i*wi) ]*X = V
                queue.push_task( new magma_slaqtrsd_task(
                    MagmaTrans, n-ki, T(ki,ki), ldt, work(ki,iv), n, work(ki,0) ));
    
                // Copy the vector x or Q*x to VL and normalize.
                if ( ! over ) {
                    // ------------------------------
                    // no back-transform: copy x to VL and normalize.
                    queue.sync();
                    n2 = n-ki;
                    blasf77_scopy( &n2, work(ki,iv  ), &ione, VL(ki,is  ), &ione );
                    blasf77_scopy( &n2, work(ki,iv+1), &ione, VL(ki,is+1), &ione );
    
                    emax = c_zero;
                    for( k=ki; k < n; ++k ) {
                        emax = max( emax, fabsf(*VL(k,is))+ fabsf(*VL(k,is+1)) );
                    }
                    remax = c_one / emax;
                    blasf77_sscal( &n2, &remax, VL(ki,is  ), &ione );
                    blasf77_sscal( &n2, &remax, VL(ki,is+1), &ione );
    
                    for( k=0; k < ki; ++k ) {
                        *VL(k,is  ) = c_zero;
                        *VL(k,is+1) = c_zero;
                    }
                }
                else if ( version == 1 ) {
                    // ------------------------------
                    // version 1: back-transform each vector with GEMV, Q*x.
                    queue.sync();
                    if ( ki < n-2 ) {
                        n2 = n-ki-2;
                        blasf77_sgemv( "n", &n, &n2, &c_one,
                                       VL(0,ki+2), &ldvl,
                                       work(ki+2,iv), &ione,
                                       work(ki,  iv), VL(0,ki), &ione );
                        blasf77_sgemv( "n", &n, &n2, &c_one,
                                       VL(0,ki+2), &ldvl,
                                       work(ki+2,iv+1), &ione,
                                       work(ki+1,iv+1), VL(0,ki+1), &ione );
                    }
                    else {
                        blasf77_sscal( &n, work(ki,  iv  ), VL(0, ki  ), &ione );
                        blasf77_sscal( &n, work(ki+1,iv+1), VL(0, ki+1), &ione );
                    }
    
                    emax = c_zero;
                    for( k=0; k < n; ++k ) {
                        emax = max( emax, fabsf(*VL(k,ki))+ fabsf(*VL(k,ki+1)) );
                    }
                    remax = c_one / emax;
                    blasf77_sscal( &n, &remax, VL(0,ki  ), &ione );
                    blasf77_sscal( &n, &remax, VL(0,ki+1), &ione );
                }
                else if ( version == 2 ) {
                    // ------------------------------
                    // version 2: back-transform block of vectors with GEMM
                    // zero out above vector
                    // could go from (ki+1)-NV+1 to ki
                    for( k=0; k < ki; ++k ) {
                        *work(k,iv  ) = c_zero;
                        *work(k,iv+1) = c_zero;
                    }
                    iscomplex[ iv   ] =  ip;
                    iscomplex[ iv+1 ] = -ip;
                    iv += 1;
                    // back-transform and normalization is done below
                }
            }  // end real or complex eigenvector
    
            if ( version == 2 ) {
                // -------------------------------------------------
                // Blocked version of back-transform
                // For complex case, (ki2+1) includes both vectors (ki+1) and (ki+2)
                if ( ip == 0 ) {
                    ki2 = ki;
                }
                else {
                    ki2 = ki + 1;
                }
    
                // Columns 1:iv of work are valid vectors.
                // When the number of vectors stored reaches nb-1 or nb,
                // or if this was last vector, do the GEMM
                if ( (iv >= nb-1) || (ki2 == n-1) ) {
                    queue.sync();
                    n2 = n-(ki2+1)+iv;
                    
                    // split gemm into multiple tasks, each doing one block row
                    for( i=0; i < n; i += gemm_nb ) {
                        magma_int_t ib = min( gemm_nb, n-i );
                        queue.push_task( new sgemm_task(
                            MagmaNoTrans, MagmaNoTrans, ib, iv, n2, c_one,
                            VL(i,ki2-iv+1), ldvl,
                            work(ki2-iv+1,1), n, c_zero,
                            work(i,nb+1), n ));
                    }
                    queue.sync();
                    // normalize vectors
                    for( k=1; k <= iv; ++k ) {
                        if ( iscomplex[k] == 0 ) {
                            // real eigenvector
                            ii = blasf77_isamax( &n, work(0,nb+k), &ione ) - 1;  // subtract 1; ii is 0-based
                            remax = c_one / fabsf( *work(ii,nb+k) );
                        }
                        else if ( iscomplex[k] == 1) {
                            // first eigenvector of conjugate pair
                            emax = c_zero;
                            for( ii=0; ii < n; ++ii ) {
                                emax = max( emax, fabsf( *work(ii,nb+k  ) )
                                                + fabsf( *work(ii,nb+k+1) ) );
                            }
                            remax = c_one / emax;
                        // else if iscomplex[k] == -1
                        //     second eigenvector of conjugate pair
                        //     reuse same remax as previous k
                        }
                        blasf77_sscal( &n, &remax, work(0,nb+k), &ione );
                    }
                    lapackf77_slacpy( "F", &n, &iv,
                                      work(0,nb+1), &n,
                                      VL(0,ki2-iv+1), &ldvl );
                    iv = 1;
                }
                else {
                    iv += 1;
                }
            } // blocked back-transform
    
            is += 1;
            if ( ip != 0 ) {
                is += 1;
            }
        }
    }
    
    // close down threads
    queue.quit();
    magma_set_lapack_numthreads( lapack_nthread );
    
    return *info;
}  // end of STREVC3
