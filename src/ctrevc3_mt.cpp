/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Mark Gates
       @author Azzam Haidar
       
       @generated from ztrevc3_mt.cpp normal z -> c, Fri Jan 30 19:00:19 2015
*/
#include "thread_queue.hpp"
#include "magma_timer.h"

#include "common_magma.h"  // after thread.hpp, so max, min are defined

#define COMPLEX

// ---------------------------------------------
// stores arguments and executes call to clatrsd (on CPU)
class magma_clatrsd_task: public magma_task
{
public:
    magma_clatrsd_task(
        magma_uplo_t in_uplo, magma_trans_t in_trans, magma_diag_t in_diag, magma_bool_t in_normin,
        magma_int_t in_n,
        const magmaFloatComplex *in_T, magma_int_t in_ldt,
        magmaFloatComplex  in_lambda,
        magmaFloatComplex *in_x, //magma_int_t in_incx,
        magmaFloatComplex *in_scale,
        float *in_cnorm
    ):
        uplo  ( in_uplo   ),
        trans ( in_trans  ),
        diag  ( in_diag   ),
        normin( in_normin ),
        n     ( in_n      ),
        T     ( in_T      ),
        ldt   ( in_ldt    ),
        lambda( in_lambda ),
        x     ( in_x      ),
        //incx  ( in_incx   ),
        scale ( in_scale  ),
        cnorm ( in_cnorm  )
    {}
    
    virtual void run()
    {
        // clatrs takes scale as float, but in ctrevc it eventually gets
        // stored in a complex; it's easiest to do that conversion here,
        // rather than storing a vector float scales[ nbmax+1 ] in ctrevc.
        magma_int_t info = 0;
        float s;
        magma_clatrsd( uplo, trans, diag, normin, n,
                       T, ldt, lambda, x, &s, cnorm, &info );
        *scale = MAGMA_C_MAKE( s, 0 );
        if ( info != 0 ) {
            fprintf( stderr, "clatrsd info %d\n", (int) info );
        }
    }
    
private:
    magma_uplo_t  uplo;
    magma_trans_t trans;
    magma_diag_t  diag;
    magma_bool_t  normin;
    magma_int_t   n;
    const magmaFloatComplex *T;
    magma_int_t   ldt;
    magmaFloatComplex lambda;
    magmaFloatComplex *x;
    magmaFloatComplex *scale;
    //magma_int_t   incx;
    float *cnorm;
};


// ---------------------------------------------
// stores arguments and executes call to cgemm (on CPU)
// todo - better to store magma_trans_t and use lapack_trans_const, since there
// is no guarantee that the char* string is still valid when the task gets executed.
class cgemm_task: public magma_task
{
public:
    cgemm_task(
        magma_trans_t in_transA, magma_trans_t in_transB,
        magma_int_t in_m, magma_int_t in_n, magma_int_t in_k,
        magmaFloatComplex  in_alpha,
        const magmaFloatComplex *in_A, magma_int_t in_lda,
        const magmaFloatComplex *in_B, magma_int_t in_ldb,
        magmaFloatComplex  in_beta,
        magmaFloatComplex *in_C, magma_int_t in_ldc
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
        blasf77_cgemm( lapack_trans_const(transA), lapack_trans_const(transB),
                       &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    }
    
private:
    magma_trans_t transA;
    magma_trans_t transB;
    magma_int_t   m;
    magma_int_t   n;
    magma_int_t   k;
    magmaFloatComplex        alpha;
    const magmaFloatComplex *A;
    magma_int_t   lda;
    const magmaFloatComplex *B;
    magma_int_t   ldb;
    magmaFloatComplex        beta;
    magmaFloatComplex       *C;
    magma_int_t   ldc;
};


/**
    Purpose
    -------
    CTREVC3_MT computes some or all of the right and/or left eigenvectors of
    a complex upper triangular matrix T.
    Matrices of this type are produced by the Schur factorization of
    a complex general matrix:  A = Q*T*Q**H, as computed by CHSEQR.

    The right eigenvector x and the left eigenvector y of T corresponding
    to an eigenvalue w are defined by:

                 T*x = w*x,     (y**H)*T = w*(y**H)

    where y**H denotes the conjugate transpose of the vector y.
    The eigenvalues are not input to this routine, but are read directly
    from the diagonal of T.

    This routine returns the matrices X and/or Y of right and left
    eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
    input matrix. If Q is the unitary factor that reduces a matrix
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

    @param[in]
    select   LOGICAL array, dimension (n)
             If howmany = MagmaSomeVec, select specifies the eigenvectors to be
             computed.
             The eigenvector corresponding to the j-th eigenvalue is
             computed if select[j] = true.
             Not referenced if howmany = MagmaAllVec or MagmaBacktransVec.

    @param[in]
    n        INTEGER
             The order of the matrix T. n >= 0.

    @param[in,out]
    T        COMPLEX array, dimension (ldt,n)
             The upper triangular matrix T.
             Unlike LAPACK's ctrevc, T is not modified, not even temporarily.

    @param[in]
    ldt      INTEGER
             The leading dimension of the array T. ldt >= max(1,n).

    @param[in,out]
    VL       COMPLEX array, dimension (ldvl,mm)
             On entry, if side = MagmaLeft or MagmaBothSides and howmany = MagmaBacktransVec, VL must
             contain an n-by-n matrix Q (usually the unitary matrix Q
             of Schur vectors returned by CHSEQR).
             On exit, if side = MagmaLeft or MagmaBothSides, VL contains:
             if howmany = MagmaAllVec, the matrix Y of left eigenvectors of T;
             if howmany = MagmaBacktransVec, the matrix Q*Y;
             if howmany = MagmaSomeVec, the left eigenvectors of T specified by
                              select, stored consecutively in the columns
                              of VL, in the same order as their eigenvalues.
             Not referenced if side = MagmaRight.

    @param[in]
    ldvl     INTEGER
             The leading dimension of the array VL.
             ldvl >= 1, and if side = MagmaLeft or MagmaBothSides, ldvl >= n.

    @param[in,out]
    VR       COMPLEX array, dimension (ldvr,mm)
             On entry, if side = MagmaRight or MagmaBothSides and howmany = MagmaBacktransVec, VR must
             contain an n-by-n matrix Q (usually the unitary matrix Q
             of Schur vectors returned by CHSEQR).
             On exit, if side = MagmaRight or MagmaBothSides, VR contains:
             if howmany = MagmaAllVec, the matrix X of right eigenvectors of T;
             if howmany = MagmaBacktransVec, the matrix Q*X;
             if howmany = MagmaSomeVec, the right eigenvectors of T specified by
                               select, stored consecutively in the columns
                               of VR, in the same order as their eigenvalues.
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
             Each selected eigenvector occupies one column.

    @param[out]
    work     COMPLEX array, dimension (max(1,lwork))

    @param[in]
    lwork    INTEGER
             The dimension of array work. lwork >= max(1,2*n).
             For optimum performance, lwork >= (1 + 2*nb)*n, where nb is
             the optimal blocksize.

    @param[out]
    rwork    float array, dimension (n)

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

    @ingroup magma_cgeev_comp
    ********************************************************************/
extern "C"
magma_int_t magma_ctrevc3_mt(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select,  // logical in Fortran
    magma_int_t n,
    magmaFloatComplex *T,  magma_int_t ldt,
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info )
{
    #define  T(i,j)  ( T + (i) + (j)*ldt )
    #define VL(i,j)  (VL + (i) + (j)*ldvl)
    #define VR(i,j)  (VR + (i) + (j)*ldvr)
    #define work(i,j) (work + (i) + (j)*n)

    // .. Parameters ..
    const magmaFloatComplex c_zero = MAGMA_C_ZERO;
    const magmaFloatComplex c_one  = MAGMA_C_ONE;
    const magma_int_t  nbmin = 16, nbmax = 128;
    const magma_int_t  ione = 1;
    
    // .. Local Scalars ..
    magma_int_t            allv, bothv, leftv, over, rightv, somev;
    magma_int_t            i, ii, is, j, k, ki, iv, n2, nb, nb2, version;
    float                 ovfl, remax, unfl;  //smlnum, smin, ulp
    
    // Decode and test the input parameters
    bothv  = (side == MagmaBothSides);
    rightv = (side == MagmaRight) || bothv;
    leftv  = (side == MagmaLeft ) || bothv;

    allv  = (howmany == MagmaAllVec);
    over  = (howmany == MagmaBacktransVec);
    somev = (howmany == MagmaSomeVec);

    // Set mout to the number of columns required to store the selected
    // eigenvectors.
    if ( somev ) {
        *mout = 0;
        for( j=0; j < n; ++j ) {
            if ( select[j] ) {
                *mout += 1;
            }
        }
    }
    else {
        *mout = n;
    }

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
    else if ( mm < *mout )
        *info = -11;
    else if ( lwork < max( 1, 2*n ) )
        *info = -14;
    
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible.
    if ( n == 0 ) {
        return *info;
    }
    
    // Use blocked version (2) if sufficient workspace.
    // Requires 1 vector to save diagonal elements, and 2*nb vectors for x and Q*x.
    // (Compared to dtrevc3, rwork stores 1-norms.)
    // Zero-out the workspace to avoid potential NaN propagation.
    nb = 2;
    if ( lwork >= n + 2*n*nbmin ) {
        version = 2;
        nb = (lwork - n) / (2*n);
        nb = min( nb, nbmax );
        nb2 = 1 + 2*nb;
        lapackf77_claset( "F", &n, &nb2, &c_zero, &c_zero, work, &n );
    }
    else {
        version = 1;
    }

    // Set the constants to control overflow.
    unfl = lapackf77_slamch( "Safe minimum" );
    ovfl = 1. / unfl;
    lapackf77_slabad( &unfl, &ovfl );
    //ulp = lapackf77_slamch( "Precision" );
    //smlnum = unfl*( n / ulp );

    // Store the diagonal elements of T in working array work.
    for( i=0; i < n; ++i ) {
        *work(i,0) = *T(i,i);
    }

    // Compute 1-norm of each column of strictly upper triangular
    // part of T to control overflow in triangular solver.
    rwork[0] = 0.;
    for( j=1; j < n; ++j ) {
        rwork[j] = magma_cblas_scasum( j, T(0,j), ione );
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

    if ( rightv ) {
        // ============================================================
        // Compute right eigenvectors.
        // iv is index of column in current block.
        // Non-blocked version always uses iv=1;
        // blocked     version starts with iv=nb, goes down to 1.
        // (Note the "0-th" column is used to store the original diagonal.)
        iv = 1;
        if ( version == 2 ) {
            iv = nb;
        }
        
        timer_start( time_trsv );
        is = *mout - 1;
        for( ki=n-1; ki >= 0; --ki ) {
            if ( somev ) {
                if ( ! select[ki] ) {
                    continue;
                }
            }
            //smin = max( ulp*MAGMA_C_ABS1( *T(ki,ki) ), smlnum );

            // --------------------------------------------------------
            // Complex right eigenvector
            *work(ki,iv) = c_one;

            // Form right-hand side.
            for( k=0; k < ki; ++k ) {
                *work(k,iv) = -(*T(k,ki));
            }

            // Solve upper triangular system:
            // [ T(1:ki-1,1:ki-1) - T(ki,ki) ]*X = scale*work.
            if ( ki > 0 ) {
                queue.push_task( new magma_clatrsd_task(
                    MagmaUpper, MagmaNoTrans, MagmaNonUnit, MagmaTrue,
                    ki, T, ldt, *T(ki,ki),
                    work(0,iv), work(ki,iv), rwork ));
            }

            // Copy the vector x or Q*x to VR and normalize.
            if ( ! over ) {
                // ------------------------------
                // no back-transform: copy x to VR and normalize
                queue.sync();
                n2 = ki+1;
                blasf77_ccopy( &n2, work(0,iv), &ione, VR(0,is), &ione );

                ii = blasf77_icamax( &n2, VR(0,is), &ione ) - 1;
                remax = 1. / MAGMA_C_ABS1( *VR(ii,is) );
                blasf77_csscal( &n2, &remax, VR(0,is), &ione );

                for( k=ki+1; k < n; ++k ) {
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
                    blasf77_cgemv( "n", &n, &ki, &c_one,
                                   VR, &ldvr,
                                   work(0, iv), &ione,
                                   work(ki,iv), VR(0,ki), &ione );
                }
                time_gemv_sum += timer_stop( time_gemv );
                ii = blasf77_icamax( &n, VR(0,ki), &ione ) - 1;
                remax = 1. / MAGMA_C_ABS1( *VR(ii,ki) );
                blasf77_csscal( &n, &remax, VR(0,ki), &ione );
                timer_start( time_trsv );
            }
            else if ( version == 2 ) {
                // ------------------------------
                // version 2: back-transform block of vectors with GEMM
                // zero out below vector
                for( k=ki+1; k < n; ++k ) {
                    *work(k,iv) = c_zero;
                }

                // Columns iv:nb of work are valid vectors.
                // When the number of vectors stored reaches nb,
                // or if this was last vector, do the GEMM
                if ( (iv == 1) || (ki == 0) ) {
                    queue.sync();
                    time_trsv_sum += timer_stop( time_trsv );
                    timer_start( time_gemm );
                    nb2 = nb-iv+1;
                    n2  = ki+nb-iv+1;
                    
                    // split gemm into multiple tasks, each doing one block row
                    for( i=0; i < n; i += gemm_nb ) {
                        magma_int_t ib = min( gemm_nb, n-i );
                        queue.push_task( new cgemm_task(
                            MagmaNoTrans, MagmaNoTrans, ib, nb2, n2, c_one,
                            VR(i,0), ldvr,
                            work(0,iv   ), n, c_zero,
                            work(i,nb+iv), n ));
                    }
                    queue.sync();
                    time_gemm_sum += timer_stop( time_gemm );
                    
                    // normalize vectors
                    // TODO if somev, should copy vectors individually to correct location.
                    for( k = iv; k <= nb; ++k ) {
                        ii = blasf77_icamax( &n, work(0,nb+k), &ione ) - 1;
                        remax = 1. / MAGMA_C_ABS1( *work(ii,nb+k) );
                        blasf77_csscal( &n, &remax, work(0,nb+k), &ione );
                    }
                    lapackf77_clacpy( "F", &n, &nb2, work(0,nb+iv), &n, VR(0,ki), &ldvr );
                    iv = nb;
                    timer_start( time_trsv );
                }
                else {
                    iv -= 1;
                }
            } // blocked back-transform

            is -= 1;
        }
    }
    timer_stop( time_trsv );
    
    timer_stop( time_total );
    timer_printf( "trevc trsv %.4f, gemm %.4f, gemv %.4f, total %.4f\n",
                  time_trsv_sum, time_gemm_sum, time_gemv_sum, time_total );

    if ( leftv ) {
        // ============================================================
        // Compute left eigenvectors.
        // iv is index of column in current block.
        // Non-blocked version always uses iv=1;
        // blocked     version starts with iv=1, goes up to nb.
        // (Note the "0-th" column is used to store the original diagonal.)
        iv = 1;
        is = 0;
        for( ki=0; ki < n; ++ki ) {
            if ( somev ) {
                if ( ! select[ki] ) {
                    continue;
                }
            }
            //smin = max( ulp*MAGMA_C_ABS1( *T(ki,ki) ), smlnum );
        
            // --------------------------------------------------------
            // Complex left eigenvector
            *work(ki,iv) = c_one;
        
            // Form right-hand side.
            for( k = ki + 1; k < n; ++k ) {
                *work(k,iv) = -MAGMA_C_CNJG( *T(ki,k) );
            }
            
            // Solve conjugate-transposed triangular system:
            // [ T(ki+1:n,ki+1:n) - T(ki,ki) ]**H * X = scale*work.
            // TODO what happens with T(k,k) - lambda is small? Used to have < smin test.
            if ( ki < n-1 ) {
                n2 = n-ki-1;
                queue.push_task( new magma_clatrsd_task(
                    MagmaUpper, MagmaConjTrans, MagmaNonUnit, MagmaTrue,
                    n2, T(ki+1,ki+1), ldt, *T(ki,ki),
                    work(ki+1,iv), work(ki,iv), rwork ));
            }
            
            // Copy the vector x or Q*x to VL and normalize.
            if ( ! over ) {
                // ------------------------------
                // no back-transform: copy x to VL and normalize
                queue.sync();
                n2 = n-ki;
                blasf77_ccopy( &n2, work(ki,iv), &ione, VL(ki,is), &ione );
        
                ii = blasf77_icamax( &n2, VL(ki,is), &ione ) + ki - 1;
                remax = 1. / MAGMA_C_ABS1( *VL(ii,is) );
                blasf77_csscal( &n2, &remax, VL(ki,is), &ione );
        
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
                    blasf77_cgemv( "n", &n, &n2, &c_one,
                                   VL(0,ki+1), &ldvl,
                                   work(ki+1,iv), &ione,
                                   work(ki,  iv), VL(0,ki), &ione );
                }
                ii = blasf77_icamax( &n, VL(0,ki), &ione ) - 1;
                remax = 1. / MAGMA_C_ABS1( *VL(ii,ki) );
                blasf77_csscal( &n, &remax, VL(0,ki), &ione );
            }
            else if ( version == 2 ) {
                // ------------------------------
                // version 2: back-transform block of vectors with GEMM
                // zero out above vector
                // could go from (ki+1)-NV+1 to ki
                for( k=0; k < ki; ++k ) {
                    *work(k,iv) = c_zero;
                }
        
                // Columns 1:iv of work are valid vectors.
                // When the number of vectors stored reaches nb,
                // or if this was last vector, do the GEMM
                if ( (iv == nb) || (ki == n-1) ) {
                    queue.sync();
                    n2 = n-(ki+1)+iv;
                    
                    // split gemm into multiple tasks, each doing one block row
                    for( i=0; i < n; i += gemm_nb ) {
                        magma_int_t ib = min( gemm_nb, n-i );
                        queue.push_task( new cgemm_task(
                            MagmaNoTrans, MagmaNoTrans, ib, iv, n2, c_one,
                            VL(i,ki-iv+1), ldvl,
                            work(ki-iv+1,1), n, c_zero,
                            work(i,nb+1), n ));
                    }
                    queue.sync();
                    // normalize vectors
                    for( k=1; k <= iv; ++k ) {
                        ii = blasf77_icamax( &n, work(0,nb+k), &ione ) - 1;
                        remax = 1. / MAGMA_C_ABS1( *work(ii,nb+k) );
                        blasf77_csscal( &n, &remax, work(0,nb+k), &ione );
                    }
                    lapackf77_clacpy( "F", &n, &iv, work(0,nb+1), &n, VL(0,ki-iv+1), &ldvl );
                    iv = 1;
                }
                else {
                    iv += 1;
                }
            } // blocked back-transform
        
            is += 1;
        }
    }
    
    // close down threads
    queue.quit();
    magma_set_lapack_numthreads( lapack_nthread );
    
    return *info;
}  // End of CTREVC
