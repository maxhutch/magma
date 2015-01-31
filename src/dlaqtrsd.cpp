/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @precisions normal d -> s
*/
#include "common_magma.h"

// Version 1 is LAPACK dlaln2. This is not thread safe.
// Version 2 is MAGMA  dlaln2, which is exactly the same, but thread safe.
#define VERSION 2

/**
    Purpose
    -------
    DLAQTRSD is used by DTREVC to solve one of the (singular) quasi-triangular
    systems with modified diagonal
        (T - lambda*I)    * x = 0  or
        (T - lambda*I)**T * x = 0
    with scaling to prevent overflow. Here T is an upper quasi-triangular
    matrix with 1x1 or 2x2 diagonal blocks, A**T denotes the transpose of A,
    and x is an n-element real or complex vector. The eigenvalue lambda is
    computed from the block diagonal of T.
    It does not modify T during the computation.
    
    If trans = MagmaNoTrans, lambda is an eigenvalue for the lower 1x1 or 2x2 block,
    and it solves
        ( [ That u      ] - lambda*I ) * x = 0,
        ( [ 0    lambda ]            )
    which becomes (That - lambda*I) * w = -s*u, with x = [ w; 1 ] and scaling s.
    If the lower block is 1x1, lambda and x are real;
    if the lower block is 2x2, lambda and x are complex.
    
    If trans = MagmaTrans, lambda is an eigenvalue for the upper 1x1 or 2x2 block,
    and it solves
        ( [ lambda v^T  ] - lambda I )**T * x = 0,
        ( [ 0      That ]            )
    which becomes (That - lambda*I)**T * w = -s*v, with x = [ 1; w ] and scaling s.
    If the upper block is 1x1, lambda and x are real;
    if the upper block is 2x2, lambda and x are complex.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the operation applied to T.
      -     = MagmaNoTrans:    Solve (T - lambda*I)    * x = 0  (No transpose)
      -     = MagmaTrans:      Solve (T - lambda*I)**T * x = 0  (Transpose)

    @param[in]
    n       INTEGER
            The order of the matrix T.  N >= 0.

    @param[in]
    T       DOUBLE PRECISION array, dimension (LDT,N)
            The triangular matrix T.  The leading n by n
            upper triangular part of the array T contains the upper
            triangular matrix, and the strictly lower triangular part of
            T is not referenced.

    @param[in]
    ldt     INTEGER
            The leading dimension of the array T.  LDT >= max (1,N).

    @param[out]
    x       DOUBLE PRECISION array, dimension (LDX,1) or (LDX,2).
            On exit, X is overwritten by the solution vector x.
            If LAMBDAI .EQ. 0, X is real    and has dimension (LDX,1).
            If LAMBDAI .NE. 0, X is complex and has dimension (LDX,2);
            the real part is in X(:,0), the imaginary part in X(:,1).
    
    @param[in]
    ldx     INTEGER
            The leading dimension of the array X.  LDX >= max(1,N).
    
    @param[in,out]
    cnorm   (input) DOUBLE PRECISION array, dimension (N)
            CNORM(j) contains the norm of the off-diagonal part of the j-th column
            of T.  If TRANS = MagmaNoTrans, CNORM(j) must be greater than or equal
            to the infinity-norm, and if TRANS = MagmaTrans or MagmaConjTrans, CNORM(j)
            must be greater than or equal to the 1-norm.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -k, the k-th argument had an illegal value

    @ingroup magma_dgeev_aux
    ********************************************************************/
extern "C"
magma_int_t magma_dlaqtrsd(
    magma_trans_t trans, magma_int_t n,
    const double *T, magma_int_t ldt,
    double *x,       magma_int_t ldx,
    const double *cnorm,
    magma_int_t *info)
{
#define T(i,j)  (T + (i) + (j)*ldt)
#define x(i,j)  (x + (i) + (j)*ldx)
#define W(i,j)  (W + (i) + (j)*2)

    // constants
    const magma_int_t c_false = false;
    const magma_int_t c_true  = true;
    const magma_int_t ione = 1;
    const magma_int_t itwo = 2;
    const double   c_zero = 0.;
    const double   c_one  = 1.;
    
    // .. Local Scalars ..
    magma_int_t notran;
    magma_int_t ierr, j, j1, j2, jnxt, k, len;
    double beta, bignum, ovfl, rec, smin,
           smlnum, ulp, unfl, vcrit, vmax, scale, wr, wi, xnorm;
    double tmp;
    
    // .. Local Arrays ..
    double W[4];
    
    // Decode and test the input parameters
    notran = (trans == MagmaNoTrans);
    
    *info = 0;
    if ( ! notran && trans != MagmaTrans ) {
        *info = -1;
    }
    else if ( n < 0 ) {
        *info = -2;
    }
    else if ( ldt < max(1,n) ) {
        *info = -4;
    }
    else if ( ldx < max(1,n) ) {
        *info = -8;
    }
    
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    // Set the constants to control overflow.
    unfl = lapackf77_dlamch( "Safe minimum" );
    ovfl = 1. / unfl;
    lapackf77_dlabad( &unfl, &ovfl );
    ulp = lapackf77_dlamch( "Precision" );
    smlnum = unfl*( n / ulp );
    bignum = (1. - ulp) / smlnum;
    
    //char buf[ 8192 ];

    if ( notran ) {
        // ============================================================
        // Compute right eigenvectors.
        // Compute the (n-1)-st eigenvalue (wr,wi).
        wr = *T(n-1,n-1);
        wi = 0.;
        if ( n >= 2 && *T(n-1,n-2) != c_zero ) {
            wi = sqrt( fabs(*T(n-1,n-2)) ) * sqrt( fabs(*T(n-2,n-1)) );
        }
        smin = max( ulp*(fabs(wr) + fabs(wi)), smlnum );
        
        //printf( "n %d, w %5.2f + %5.2fi\n", n, wr, wi );

        if ( wi == c_zero ) {
            // ------------------------------------------------------------
            // Real eigenvalue
            // Form right-hand side.
            *x(n-1,0) = c_one;
            for( k=0; k < n-1; ++k ) {
                *x(k,0) = -(*T(k,n-1));
            }
            
            //magma_d_snprint( buf, sizeof(buf), n, 1, x, ldx );
            
            // Solve upper quasi-triangular system:
            // [ T(0:n-2,0:n-2) - wr ]*x = scale*b.
            jnxt = n-2;
            for( j=n-2; j >= 0; --j ) {
                if ( j > jnxt ) {
                    continue;
                }
                j1 = j;
                j2 = j;
                jnxt = j - 1;
                if ( j > 0 ) {
                    if ( *T(j,j-1) != c_zero ) {
                        j1   = j - 1;
                        jnxt = j - 2;
                    }
                }
                
                if ( j1 == j2 ) {
                    // 1-by-1 diagonal block
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_false, &ione, &ione, &smin, &c_one,
                            T(j,j), &ldt, &c_one, &c_one, x(j,0), &ldx,
                            &wr, &c_zero, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_false, ione, ione, smin, c_one,
                        T(j,j), ldt, c_one, c_one, x(j,0), ldx,
                        wr, c_zero, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale W(0,0) to avoid overflow when updating
                    // the right-hand side.
                    if ( xnorm > 1. ) {
                        if ( cnorm[j] > bignum / xnorm ) {
                            *W(0,0) /= xnorm;
                            scale   /= xnorm;
                        }
                    }
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                    }
                    *x(j,0) = *W(0,0);
                    
                    // Update right-hand side
                    len = j;
                    tmp = -(*W(0,0));
                    blasf77_daxpy( &len, &tmp, T(0,j), &ione, x(0,0), &ione );
                }
                else {
                    // 2-by-2 diagonal block
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_false, &itwo, &ione, &smin, &c_one,
                            T(j-1,j-1), &ldt, &c_one, &c_one, x(j-1,0), &ldx,
                            &wr, &c_zero, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_false, itwo, ione, smin, c_one,
                        T(j-1,j-1), ldt, c_one, c_one, x(j-1,0), ldx,
                        wr, c_zero, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale W(0,0) and W(1,0) to avoid overflow when
                    // updating the right-hand side.
                    if ( xnorm > 1. ) {
                        beta = max( cnorm[j-1], cnorm[j] );
                        if ( beta > bignum / xnorm ) {
                            *W(0,0) /= xnorm;
                            *W(1,0) /= xnorm;
                            scale   /= xnorm;
                        }
                    }
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                    }
                    *x(j-1,0) = *W(0,0);
                    *x(j,  0) = *W(1,0);
                    
                    // Update right-hand side
                    len = (j+1)-2;
                    tmp = -(*W(0,0));  blasf77_daxpy( &len, &tmp, T(0,j-1), &ione, x(0,0), &ione );
                    tmp = -(*W(1,0));  blasf77_daxpy( &len, &tmp, T(0,j  ), &ione, x(0,0), &ione );
                }
            }
            
            //printf( "real x=\n%s\n", buf );
            
        }  // end real eigenvalue
        else {
            // ------------------------------------------------------------
            // Complex eigenvalue
            // Initial solve
            // [ ( T(n-2,n-2) T(n-2,n-1) ) - (wr + i*wi) ]*X = 0.
            // [ ( T(n-1,n-2) T(n-1,n-1) )               ]
            if ( fabs(*T(n-2,n-1)) >= fabs(*T(n-1,n-2)) ) {
                *x(n-2,0) = c_one;
                *x(n-1,1) = wi / *T(n-2,n-1);
            }
            else {
                *x(n-2,0) = -wi / *T(n-1,n-2);
                *x(n-1,1) = c_one;
            }
            *x(n-1,0) = c_zero;
            *x(n-2,1) = c_zero;

            // Form right-hand side.
            for( k=0; k < n-2; ++k ) {
                *x(k,0) = -(*x(n-2,0)) * (*T(k,n-2));
                *x(k,1) = -(*x(n-1,1)) * (*T(k,n-1));
            }
            
            //magma_d_snprint( buf, sizeof(buf), n, 2, x, ldx );
            
            // Solve upper quasi-triangular system:
            // [ T(0:n-3,0:n-3) - (wr + i*wi) ]*x = scale*b
            jnxt = n-3;
            for( j=n-3; j >= 0; --j ) {
                if ( j > jnxt ) {
                    continue;
                }
                j1 = j;
                j2 = j;
                jnxt = j - 1;
                if ( j > 0 ) {
                    if ( *T(j,j-1) != c_zero ) {
                        j1   = j - 1;
                        jnxt = j - 2;
                    }
                }
                
                if ( j1 == j2 ) {
                    // 1-by-1 diagonal block
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_false, &ione, &itwo, &smin, &c_one,
                            T(j,j), &ldt, &c_one, &c_one, x(j,0), &ldx,
                            &wr, &wi, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_false, ione, itwo, smin, c_one,
                        T(j,j), ldt, c_one, c_one, x(j,0), ldx,
                        wr, wi, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale W(0,0) and W(0,1) to avoid overflow when
                    // updating the right-hand side.
                    if ( xnorm > 1. ) {
                        if ( cnorm[j] > bignum / xnorm ) {
                            *W(0,0) /= xnorm;
                            *W(0,1) /= xnorm;
                            scale   /= xnorm;
                        }
                    }
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                        blasf77_dscal( &len, &scale, x(0,1), &ione );
                    }
                    *x(j,0) = *W(0,0);
                    *x(j,1) = *W(0,1);
                    
                    // Update the right-hand side
                    len = j;
                    tmp = -(*W(0,0));  blasf77_daxpy( &len, &tmp, T(0,j), &ione, x(0,0), &ione );
                    tmp = -(*W(0,1));  blasf77_daxpy( &len, &tmp, T(0,j), &ione, x(0,1), &ione );
                }
                else {
                    // 2-by-2 diagonal block
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_false, &itwo, &itwo, &smin, &c_one,
                            T(j-1,j-1), &ldt, &c_one, &c_one, x(j-1,0), &ldx,
                            &wr, &wi, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_false, itwo, itwo, smin, c_one,
                        T(j-1,j-1), ldt, c_one, c_one, x(j-1,0), ldx,
                        wr, wi, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale W to avoid overflow when updating
                    // the right-hand side.
                    if ( xnorm > 1. ) {
                        beta = max( cnorm[j-1], cnorm[j] );
                        if ( beta > bignum / xnorm ) {
                            rec = c_one / xnorm;
                            *W(0,0) *= rec;
                            *W(0,1) *= rec;
                            *W(1,0) *= rec;
                            *W(1,1) *= rec;
                            scale   *= rec;
                        }
                    }
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                        blasf77_dscal( &len, &scale, x(0,1), &ione );
                    }
                    *x(j-1,0) = *W(0,0);
                    *x(j,  0) = *W(1,0);
                    *x(j-1,1) = *W(0,1);
                    *x(j,  1) = *W(1,1);
                    
                    // Update the right-hand side
                    len = (j+1)-2;
                    tmp = -(*W(0,0));  blasf77_daxpy( &len, &tmp, T(0,j-1), &ione, x(0,0), &ione );
                    tmp = -(*W(1,0));  blasf77_daxpy( &len, &tmp, T(0,j  ), &ione, x(0,0), &ione );
                    tmp = -(*W(0,1));  blasf77_daxpy( &len, &tmp, T(0,j-1), &ione, x(0,1), &ione );
                    tmp = -(*W(1,1));  blasf77_daxpy( &len, &tmp, T(0,j  ), &ione, x(0,1), &ione );
                }
            }
            
            //printf( "complex x=\n%s\n", buf );
            
        }  // end complex eigenvalue
    }  // end notran
    else { // transposed
        // ============================================================
        // Compute left eigenvectors.
        // Compute the 0-th eigenvalue (wr,wi).
        wr = *T(0,0);
        wi = 0.;
        if ( n >= 2 && *T(1,0) != c_zero ) {
            wi = sqrt( fabs(*T(0,1)) ) * sqrt( fabs(*T(1,0)) );
        }
        smin = max( ulp*(fabs(wr) + fabs(wi)), smlnum );
        
        if ( wi == c_zero ) {
            // ------------------------------------------------------------
            // Real eigenvalue
            // Form right-hand side.
            *x(0,0) = c_one;
            for( k=1; k < n; ++k ) {
                *x(k,0) = -(*T(0,k));
            }
            
            // Solve transposed quasi-triangular system:
            // [ T(1:n-1,1:n-1) - wr ]**T * x = scale*b
            vmax = c_one;
            vcrit = bignum;
            
            jnxt = 1;
            for( j=1; j < n; ++j ) {
                if ( j < jnxt ) {
                    continue;
                }
                j1 = j;
                j2 = j;
                jnxt = j + 1;
                if ( j < n-1 ) {
                    if ( *T(j+1,j) != c_zero ) {
                        j2   = j + 1;
                        jnxt = j + 2;
                    }
                }

                if ( j1 == j2 ) {
                    // 1-by-1 diagonal block
                    // Scale if necessary to avoid overflow when forming
                    // the right-hand side.
                    if ( cnorm[j] > vcrit ) {
                        rec = c_one / vmax;
                        len = n;
                        blasf77_dscal( &len, &rec, x(0,0), &ione );
                        vmax = c_one;
                        vcrit = bignum;
                    }

                    len = j-1;
                    *x(j,0) -= magma_cblas_ddot( len, T(1,j), ione, x(1,0), ione );

                    // Solve [ T(j,j)-wr ]**T * x = b
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_false, &ione, &ione, &smin, &c_one,
                            T(j,j), &ldt, &c_one, &c_one, x(j,0), &ldx,
                            &wr, &c_zero, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_false, ione, ione, smin, c_one,
                        T(j,j), ldt, c_one, c_one, x(j,0), ldx,
                        wr, c_zero, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                    }
                    *x(j,0) = *W(0,0);
                    vmax = max( fabs(*x(j,0)), vmax );
                    vcrit = bignum / vmax;
                }
                else {
                    // 2-by-2 diagonal block
                    // Scale if necessary to avoid overflow when forming
                    // the right-hand side.
                    beta = max( cnorm[j], cnorm[j+1] );
                    if ( beta > vcrit ) {
                        rec = c_one / vmax;
                        len = n;
                        blasf77_dscal( &len, &rec, x(0,0), &ione );
                        vmax = c_one;
                        vcrit = bignum;
                    }

                    len = j-1;
                    *x(j,  0) -= magma_cblas_ddot( len, T(1,j),   ione, x(1,0), ione );
                    *x(j+1,0) -= magma_cblas_ddot( len, T(1,j+1), ione, x(1,0), ione );

                    // Solve
                    // [T(j,j)-wr   T(j,j+1)     ]**T * x = scale*( WORK1 )
                    // [T(j+1,j)    T(j+1,j+1)-wr]                ( WORK2 )
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_true, &itwo, &ione, &smin, &c_one,
                            T(j,j), &ldt, &c_one, &c_one, x(j,0), &ldx,
                            &wr, &c_zero, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_true, itwo, ione, smin, c_one,
                        T(j,j), ldt, c_one, c_one, x(j,0), ldx,
                        wr, c_zero, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                    }
                    *x(j,  0) = *W(0,0);
                    *x(j+1,0) = *W(1,0);

                    vmax = max( max( fabs(*x(j,0)), fabs(*x(j+1,0)) ), vmax );
                    vcrit = bignum / vmax;
                }
            }
        }  // end real eigenvalue
        else {
            // ------------------------------------------------------------
            // Complex eigenvalue
            // Initial solve:
            // [ ( T(0,0)  T(0,1) )**T - (wr - i*wi) ]*X = 0.
            // [ ( T(1,0)  T(1,1) )                  ]
            if ( fabs(*T(0,1)) >= fabs(*T(1,0)) ) {
                *x(0,0) = wi / *T(0,1);
                *x(1,1) = c_one;
            }
            else {
                *x(0,0) = c_one;
                *x(1,1) = -wi / *T(1,0);
            }
            *x(1,0) = c_zero;
            *x(0,1) = c_zero;

            // Form right-hand side.
            for( k=0 + 2; k < n; ++k ) {
                *x(k,0) = -(*x(0,0)) * (*T(0,k));
                *x(k,1) = -(*x(1,1)) * (*T(1,k));
            }
            
            // Solve transposed quasi-triangular system:
            // [ T(2:n-1,2:n-1)**T - (wr-i*wi) ]*W = WORK1+i*WORK2
            vmax = c_one;
            vcrit = bignum;

            jnxt = 2;
            for( j=2; j < n; ++j ) {
                if ( j < jnxt ) {
                    continue;
                }
                j1 = j;
                j2 = j;
                jnxt = j + 1;
                if ( j < n-1 ) {
                    if ( *T(j+1,j) != c_zero ) {
                        j2 = j + 1;
                        jnxt = j + 2;
                    }
                }

                if ( j1 == j2 ) {
                    // 1-by-1 diagonal block
                    // Scale if necessary to avoid overflow when
                    // forming the right-hand side elements.
                    if ( cnorm[j] > vcrit ) {
                        rec = c_one / vmax;
                        len = n;
                        blasf77_dscal( &len, &rec, x(0,0), &ione );
                        blasf77_dscal( &len, &rec, x(0,1), &ione );
                        vmax = c_one;
                        vcrit = bignum;
                    }

                    len = j-2;
                    *x(j,0) -= magma_cblas_ddot( len, T(2,j), ione, x(2, 0), ione );
                    *x(j,1) -= magma_cblas_ddot( len, T(2,j), ione, x(2, 1), ione );

                    // Solve [ T(j,j)-(wr-i*wi) ]*(W11+i*W12)= WK+i*WK2
                    tmp = -wi;
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_false, &ione, &itwo, &smin, &c_one,
                            T(j,j), &ldt, &c_one, &c_one, x(j,0), &ldx,
                            &wr, &tmp, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_false, ione, itwo, smin, c_one,
                        T(j,j), ldt, c_one, c_one, x(j,0), ldx,
                        wr, tmp, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                        blasf77_dscal( &len, &scale, x(0,1), &ione );
                    }
                    *x(j,0) = *W(0,0);
                    *x(j,1) = *W(0,1);
                    vmax = max( max( fabs(*x(j,0)), fabs(*x(j,1)) ), vmax );
                    vcrit = bignum / vmax;
                }
                else {
                    // 2-by-2 diagonal block
                    // Scale if necessary to avoid overflow when forming
                    // the right-hand side elements.
                    beta = max( cnorm[j], cnorm[j+1] );
                    if ( beta > vcrit ) {
                        rec = c_one / vmax;
                        len = n;
                        blasf77_dscal( &len, &rec, x(0,0), &ione );
                        blasf77_dscal( &len, &rec, x(0,1), &ione );
                        vmax = c_one;
                        vcrit = bignum;
                    }

                    len = j-2;
                    *x(j,  0) -= magma_cblas_ddot( len, T(2,j),   ione, x(2,0), ione );
                    *x(j,  1) -= magma_cblas_ddot( len, T(2,j),   ione, x(2,1), ione );
                    *x(j+1,0) -= magma_cblas_ddot( len, T(2,j+1), ione, x(2,0), ione );
                    *x(j+1,1) -= magma_cblas_ddot( len, T(2,j+1), ione, x(2,1), ione );

                    // Solve 2-by-2 complex linear equation
                    // [ (T(j,j)   T(j,j+1)  )**T - (wr-i*wi)*i ]*W = scale*B
                    // [ (T(j+1,j) T(j+1,j+1))                  ]
                    tmp = -wi;
                    #if VERSION == 1
                    lapackf77_dlaln2(
                            &c_true, &itwo, &itwo, &smin, &c_one,
                            T(j,j), &ldt, &c_one, &c_one, x(j,0), &ldx,
                            &wr, &tmp, W, &itwo, &scale, &xnorm, &ierr );
                    #else
                    magma_dlaln2(
                        c_true, itwo, itwo, smin, c_one,
                        T(j,j), ldt, c_one, c_one, x(j,0), ldx,
                        wr, tmp, W, itwo, &scale, &xnorm, &ierr );
                    #endif
                    
                    // Scale if necessary
                    if ( scale != 1. ) {
                        len = n;
                        blasf77_dscal( &len, &scale, x(0,0), &ione );
                        blasf77_dscal( &len, &scale, x(0,1), &ione );
                    }
                    *x(j,  0) = *W(0,0);
                    *x(j,  1) = *W(0,1);
                    *x(j+1,0) = *W(1,0);
                    *x(j+1,1) = *W(1,1);
                    vmax = max( max( max( max( fabs(*W(0,0)), fabs(*W(0,1)) ),
                                               fabs(*W(1,0)) ), fabs(*W(1,1)) ), vmax );
                    vcrit = bignum / vmax;
                }
            }
        }  // end complex eigenvalue
    }  // end transposed
    
    return *info;
} /* end dlaqtrsd */
