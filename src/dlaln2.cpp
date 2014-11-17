/*
    Originally translated from lapack dlaln2.f to dlaln2.c using f2c.
    Later cleaned up by hand, particularly to be thread-safe (no static variables).
    
    @author Mark Gates
    @precisions normal d -> s
*/
#include "common_magma.h"

/**
    Purpose
    -------
    DLALN2 solves a system of the form
            (ca A    - w D) X = s B
        or  (ca A**T - w D) X = s B
    with possible scaling ("s") and
    perturbation of A.  (A**T means A-transpose.)

    A is an NA x NA real matrix, ca is a real scalar, D is an NA x NA
    real diagonal matrix, w is a real or complex value, and X and B are
    NA x 1 matrices -- real if w is real, complex if w is complex.  NA
    may be 1 or 2.

    If w is complex, X and B are represented as NA x 2 matrices,
    the first column of each being the real part and the second
    being the imaginary part.

    "s" is a scaling factor (.LE. 1), computed by DLALN2, which is
    so chosen that X can be computed without overflow.  X is further
    scaled if necessary to assure that norm(ca A - w D)*norm(X) is less
    than overflow.

    If both singular values of (ca A - w D) are less than SMIN,
    SMIN*identity will be used instead of (ca A - w D).  If only one
    singular value is less than SMIN, one element of (ca A - w D) will be
    perturbed enough to make the smallest singular value roughly SMIN.
    If both singular values are at least SMIN, (ca A - w D) will not be
    perturbed.  In any case, the perturbation will be at most some small
    multiple of max( SMIN, ulp*norm(ca A - w D) ).  The singular values
    are computed by infinity-norm approximations, and thus will only be
    correct to a factor of 2 or so.

    Note: all input quantities are assumed to be smaller than overflow
    by a reasonable factor.  (See BIGNUM.)

    Arguments
    ----------
    @param[in]
    trans   LOGICAL
            =.TRUE.:  A-transpose will be used.
            =.FALSE.: A will be used (not transposed.)

    @param[in]
    na      INTEGER
            The size of the matrix A.  It may (only) be 1 or 2.

    @param[in]
    nw      INTEGER
            1 if "w" is real, 2 if "w" is complex.  It may only be 1
            or 2.

    @param[in]
    smin    DOUBLE PRECISION
            The desired lower bound on the singular values of A.  This
            should be a safe distance away from underflow or overflow,
            say, between (underflow/machine precision) and  (machine
            precision * overflow ).  (See BIGNUM and ULP.)

    @param[in]
    ca      DOUBLE PRECISION
            The coefficient c, which A is multiplied by.

    @param[in]
    A       DOUBLE PRECISION array, dimension (LDA,NA)
            The NA x NA matrix A.

    @param[in]
    lda     INTEGER
            The leading dimension of A.  It must be at least NA.

    @param[in]
    d1      DOUBLE PRECISION
            The 1,1 element in the diagonal matrix D.

    @param[in]
    d2      DOUBLE PRECISION
            The 2,2 element in the diagonal matrix D.  Not used if NW=1.

    @param[in]
    B       DOUBLE PRECISION array, dimension (LDB,NW)
            The NA x NW matrix B (right-hand side).  If NW=2 ("w" is
            complex), column 1 contains the real part of B and column 2
            contains the imaginary part.

    @param[in]
    ldb     INTEGER
            The leading dimension of B.  It must be at least NA.

    @param[in]
    wr      DOUBLE PRECISION
            The real part of the scalar "w".

    @param[in]
    wi      DOUBLE PRECISION
            The imaginary part of the scalar "w".  Not used if NW=1.

    @param[out]
    X       DOUBLE PRECISION array, dimension (LDX,NW)
            The NA x NW matrix X (unknowns), as computed by DLALN2.
            If NW=2 ("w" is complex), on exit, column 1 will contain
            the real part of X and column 2 will contain the imaginary
            part.

    @param[in]
    ldx     INTEGER
            The leading dimension of X.  It must be at least NA.

    @param[out]
    scale   DOUBLE PRECISION
            The scale factor that B must be multiplied by to insure
            that overflow does not occur when computing X.  Thus,
            (ca A - w D) X  will be SCALE*B, not B (ignoring
            perturbations of A.)  It will be at most 1.

    @param[out]
    xnorm   DOUBLE PRECISION
            The infinity-norm of X, when X is regarded as an NA x NW
            real matrix.

    @param[out]
    info    INTEGER
            An error flag.  It will be set to zero if no error occurs,
            a negative number if an argument is in error, or a positive
            number if  ca A - w D  had to be perturbed.
            The possible values are:
      -     = 0: No error occurred, and (ca A - w D) did not have to be
                   perturbed.
      -     = 1: (ca A - w D) had to be perturbed to make its smallest
                 (or only) singular value greater than SMIN.
            NOTE: In the interests of speed, this routine does not
                  check the inputs for errors.

    @ingroup magma_daux0
    ********************************************************************/
extern "C"
magma_int_t magma_dlaln2(
    magma_int_t trans, magma_int_t na, magma_int_t nw,
    double smin, double ca, const double *A, magma_int_t lda,
    double d1, double d2, const double *B, magma_int_t ldb,
    double wr, double wi, double *X, magma_int_t ldx,
    double *scale, double *xnorm,
    magma_int_t *info)
{
    // normally, we use A(i,j) to be a pointer, A + i + j*lda, but
    // in this function it is more convenient to be an element, A[i + j*lda].
    #define A(i_,j_) (A[ (i_) + (j_)*lda ])
    #define B(i_,j_) (B[ (i_) + (j_)*ldb ])
    #define X(i_,j_) (X[ (i_) + (j_)*ldx ])
    
    /* Initialized data */
    magma_int_t zswap[4] = { false, false, true, true };
    magma_int_t rswap[4] = { false, true, false, true };
    magma_int_t ipivot[16] = { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 };    /* was [4][4] */

    /* System generated locals */
    double d__1, d__2;

    /* Local variables */
    magma_int_t j;
    double bi1, bi2, br1, br2, xi1, xi2, xr1, xr2, ci21, ci22, cr21, cr22,
           li21, csi, ui11, lr21, ui12, ui22;
    double csr, ur11, ur12, ur22;
    double bbnd, cmax, ui11r, ui12s, temp, ur11r, ur12s, u22abs;
    magma_int_t icmax;
    double bnorm, cnorm, smini;
    double bignum, smlnum;

    double equiv_0[4], equiv_1[4];
#define ci  (equiv_0)  /* EQUIVALENCE CI(2,2) and CIV(4) */
#define civ (equiv_0)

#define cr  (equiv_1)  /* EQUIVALENCE CR(2,2) and CRV(4) */
#define crv (equiv_1)
    
    /* Parameter adjustments */
    A -= 1 + lda;
    B -= 1 + ldb;
    X -= 1 + ldx;
    
    /* Compute BIGNUM */
    smlnum = 2. * lapackf77_dlamch("Safe minimum");
    bignum = 1. / smlnum;
    smini = max(smin, smlnum);
    
    /* Don't check for input errors */
    *info = 0;
    
    /* Standard Initializations */
    *scale = 1.;
    
    if (na == 1) {
        /* 1 x 1  (i.e., scalar) system   C X = B */
        if (nw == 1) {
            /* Real 1x1 system. */
            /* C = ca A - w D   */
            csr = ca * A(1,1) - wr * d1;
            cnorm = fabs(csr);
            
            /* If | C | < SMINI, use C = SMINI */
            if (cnorm < smini) {
                csr = smini;
                cnorm = smini;
                *info = 1;
            }
            
            /* Check scaling for  X = B / C */
            bnorm = fabs( B(1,1) );
            if (cnorm < 1. && bnorm > 1.) {
                if (bnorm > bignum * cnorm) {
                    *scale = 1. / bnorm;
                }
            }
            
            /* Compute X */
            X(1,1) = B(1,1) * *scale / csr;
            *xnorm = fabs( X(1,1) );
        }
        else {
            /* Complex 1x1 system (w is complex) */
            /* C = ca A - w D */
            csr = ca * A(1,1) - wr * d1;
            csi = -(wi) * d1;
            cnorm = fabs(csr) + fabs(csi);
            
            /* If | C | < SMINI, use C = SMINI */
            if (cnorm < smini) {
                csr = smini;
                csi = 0.;
                cnorm = smini;
                *info = 1;
            }
            
            /* Check scaling for  X = B / C */
            bnorm = fabs( B(1,1) ) + fabs( B(1,2) );
            if (cnorm < 1. && bnorm > 1.) {
                if (bnorm > bignum * cnorm) {
                    *scale = 1. / bnorm;
                }
            }
            
            /* Compute X */
            d__1 = *scale * B(1,1);
            d__2 = *scale * B(1,2);
            lapackf77_dladiv( &d__1, &d__2, &csr, &csi, &X(1,1), &X(1,2) );
            *xnorm = fabs( X(1,1) ) + fabs( X(1,2) );
        }
    }
    else {
        /* 2x2 System */
        /* Compute the real part of  C = ca A - w D  (or  ca A**T - w D ) */
        cr[0] = ca * A(1,1) - wr * d1;
        cr[3] = ca * A(2,2) - wr * d2;
        if (trans) {
            cr[2] = ca * A(2,1);
            cr[1] = ca * A(1,2);
        }
        else {
            cr[1] = ca * A(2,1);
            cr[2] = ca * A(1,2);
        }
        
        if (nw == 1) {
            /* Real 2x2 system  (w is real) */
            /* Find the largest element in C */
            cmax = 0.;
            icmax = 0;
        
            for (j = 1; j <= 4; ++j) {
                if (fabs( crv[j - 1] ) > cmax) {
                    cmax = fabs( crv[j - 1] );
                    icmax = j;
                }
                /* L10: */
            }
            
            /* If norm(C) < SMINI, use SMINI*identity. */
            if (cmax < smini) {
                bnorm = max( fabs( B(1,1) ), fabs( B(2,1) ) );
                if (smini < 1. && bnorm > 1.) {
                    if (bnorm > bignum * smini) {
                        *scale = 1. / bnorm;
                    }
                }
                temp = *scale / smini;
                X(1,1) = temp * B(1,1);
                X(2,1) = temp * B(2,1);
                *xnorm = temp * bnorm;
                *info = 1;
                return *info;
            }
        
            /* Gaussian elimination with complete pivoting. */
            ur11 = crv[icmax - 1];
            cr21 = crv[ipivot[(icmax << 2) - 3] - 1];
            ur12 = crv[ipivot[(icmax << 2) - 2] - 1];
            cr22 = crv[ipivot[(icmax << 2) - 1] - 1];
            ur11r = 1. / ur11;
            lr21 = ur11r * cr21;
            ur22 = cr22 - ur12 * lr21;
        
            /* If smaller pivot < SMINI, use SMINI */
            if (fabs(ur22) < smini) {
                ur22 = smini;
                *info = 1;
            }
            if (rswap[icmax - 1]) {
                br1 = B(2,1);
                br2 = B(1,1);
            }
            else {
                br1 = B(1,1);
                br2 = B(2,1);
            }
            br2 -= lr21 * br1;
            bbnd = max( fabs( br1 * (ur22 * ur11r) ), fabs(br2) );
            if (bbnd > 1. && fabs(ur22) < 1.) {
                if (bbnd >= bignum * fabs(ur22)) {
                    *scale = 1. / bbnd;
                }
            }
        
            xr2 = br2 * *scale / ur22;
            xr1 = *scale * br1 * ur11r - xr2 * (ur11r * ur12);
            if (zswap[icmax - 1]) {
                X(1,1) = xr2;
                X(2,1) = xr1;
            }
            else {
                X(1,1) = xr1;
                X(2,1) = xr2;
            }
            *xnorm = max( fabs(xr1), fabs(xr2) );
        
            /* Further scaling if  norm(A) norm(X) > overflow */
            if (*xnorm > 1. && cmax > 1.) {
                if (*xnorm > bignum / cmax) {
                    temp = cmax / bignum;
                    X(1,1) = temp * X(1,1);
                    X(2,1) = temp * X(2,1);
                    *xnorm = temp * *xnorm;
                    *scale = temp * *scale;
                }
            }
        }
        else {
            /* Complex 2x2 system  (w is complex) */
            /* Find the largest element in C */
            ci[0] = -(wi) * d1;
            ci[1] = 0.;
            ci[2] = 0.;
            ci[3] = -(wi) * d2;
            cmax = 0.;
            icmax = 0;
        
            for (j = 1; j <= 4; ++j) {
            if (   fabs( crv[j - 1] ) + fabs( civ[j - 1] ) > cmax) {
                cmax = fabs( crv[j - 1] ) + fabs( civ[j - 1] );
                icmax = j;
            }
            /* L20: */
            }
        
            /* If norm(C) < SMINI, use SMINI*identity. */
        
            if (cmax < smini) {
                bnorm = max( fabs( B(1,1) ) + fabs( B(1,2) ),
                             fabs( B(2,1) ) + fabs( B(2,2) ) );
                if (smini < 1. && bnorm > 1.) {
                    if (bnorm > bignum * smini) {
                        *scale = 1. / bnorm;
                    }
                }
                temp = *scale / smini;
                X(1,1) = temp * B(1,1);
                X(2,1) = temp * B(2,1);
                X(1,2) = temp * B(1,2);
                X(2,2) = temp * B(2,2);
                *xnorm = temp * bnorm;
                *info = 1;
                return *info;
            }
        
            /* Gaussian elimination with complete pivoting. */
            ur11 = crv[icmax - 1];
            ui11 = civ[icmax - 1];
            cr21 = crv[ipivot[(icmax << 2) - 3] - 1];
            ci21 = civ[ipivot[(icmax << 2) - 3] - 1];
            ur12 = crv[ipivot[(icmax << 2) - 2] - 1];
            ui12 = civ[ipivot[(icmax << 2) - 2] - 1];
            cr22 = crv[ipivot[(icmax << 2) - 1] - 1];
            ci22 = civ[ipivot[(icmax << 2) - 1] - 1];
            if (icmax == 1 || icmax == 4) {
                /* Code when off-diagonals of pivoted C are real */
                if (fabs(ur11) > fabs(ui11)) {
                    temp = ui11 / ur11;
                    ur11r = 1. / (ur11 * (temp * temp + 1.));
                    ui11r = -temp * ur11r;
                }
                else {
                    temp = ur11 / ui11;
                    ui11r = -1. / (ui11 * (temp * temp + 1.));
                    ur11r = -temp * ui11r;
                }
                lr21  = cr21 * ur11r;
                li21  = cr21 * ui11r;
                ur12s = ur12 * ur11r;
                ui12s = ur12 * ui11r;
                ur22  = cr22 - ur12 * lr21;
                ui22  = ci22 - ur12 * li21;
            }
            else {
                /* Code when diagonals of pivoted C are real */
                ur11r = 1. / ur11;
                ui11r = 0.;
                lr21  = cr21 * ur11r;
                li21  = ci21 * ur11r;
                ur12s = ur12 * ur11r;
                ui12s = ui12 * ur11r;
                ur22  = cr22 - ur12 * lr21 + ui12 * li21;
                ui22  = -ur12 * li21 - ui12 * lr21;
            }
            u22abs = fabs(ur22) + fabs(ui22);
        
            /* If smaller pivot < SMINI, use SMINI */
        
            if (u22abs < smini) {
                ur22 = smini;
                ui22 = 0.;
                *info = 1;
            }
            if (rswap[icmax - 1]) {
                br2 = B(1,1);
                br1 = B(2,1);
                bi2 = B(1,2);
                bi1 = B(2,2);
            }
            else {
                br1 = B(1,1);
                br2 = B(2,1);
                bi1 = B(1,2);
                bi2 = B(2,2);
            }
            br2 = br2 - lr21 * br1 + li21 * bi1;
            bi2 = bi2 - li21 * br1 - lr21 * bi1;
            bbnd = max( (fabs(br1) + fabs(bi1)) * (u22abs * (fabs(ur11r) + fabs(ui11r))),
                        fabs(br2) + fabs(bi2) );
            if (bbnd > 1. && u22abs < 1.) {
                if (bbnd >= bignum * u22abs) {
                    *scale = 1. / bbnd;
                    br1 = *scale * br1;
                    bi1 = *scale * bi1;
                    br2 = *scale * br2;
                    bi2 = *scale * bi2;
                }
            }
            
            lapackf77_dladiv( &br2, &bi2, &ur22, &ui22, &xr2, &xi2 );
            xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
            xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
            if (zswap[icmax - 1]) {
                X(1,1) = xr2;
                X(2,1) = xr1;
                X(1,2) = xi2;
                X(2,2) = xi1;
            }
            else {
                X(1,1) = xr1;
                X(2,1) = xr2;
                X(1,2) = xi1;
                X(2,2) = xi2;
            }
            *xnorm = max( fabs(xr1) + fabs(xi1), fabs(xr2) + fabs(xi2) );
        
            /* Further scaling if  norm(A) norm(X) > overflow */
            if (*xnorm > 1. && cmax > 1.) {
                if (*xnorm > bignum / cmax) {
                    temp = cmax / bignum;
                    X(1,1) = temp * X(1,1);
                    X(2,1) = temp * X(2,1);
                    X(1,2) = temp * X(1,2);
                    X(2,2) = temp * X(2,2);
                    *xnorm = temp * *xnorm;
                    *scale = temp * *scale;
                }
            }
        }
    }

    return *info;
} /* dlaln2_ */

#undef crv
#undef civ
#undef cr
#undef ci
