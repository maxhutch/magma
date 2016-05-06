/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *
 *     @generated from src/core_zlarfy.cpp normal z -> c, Mon May  2 23:30:20 2016
 *
 */
#include "magma_internal.h"
#include "magma_bulge.h"

/***************************************************************************//**
 *
 * @ingroup magma_caux3
 *
 *  magma_clarfy applies an elementary reflector, or Householder matrix, H,
 *  to a n-by-n Hermitian matrix C, from both the left and the right.
 *
 *  H is represented in the form
 *
 *     H = I - tau * v * v'
 *
 *  where  tau  is a scalar and  v  is a vector.
 *
 *  If tau is zero, then H is taken to be the unit matrix.
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The number of rows and columns of the matrix C.  n >= 0.
 *
 * @param[in,out] A
 *          COMPLEX array, dimension (lda, n)
 *          On entry, the Hermetian matrix A.
 *          On exit, A is overwritten by H * A * H'.
 *
 * @param[in] lda
 *         The leading dimension of the array A.  lda >= max(1,n).
 *
 * @param[in] V
 *          The vector V that contains the Householder reflectors.
 *
 * @param[in] TAU
 *          The value tau.
 *
 * @param[out] work
 *          Workspace.
 *
 ******************************************************************************/
extern "C" void
magma_clarfy(
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *V, const magmaFloatComplex *TAU,
    magmaFloatComplex *work)
{
    /*
    work (workspace) float complex array, dimension n
    */

    static magma_int_t ione = 1;
    static magmaFloatComplex c_zero   =  MAGMA_C_ZERO;
    static magmaFloatComplex c_neg_one=  MAGMA_C_NEG_ONE;
    static magmaFloatComplex c_half   =  MAGMA_C_HALF;
    magmaFloatComplex dtmp;

    /* X = AVtau */
    blasf77_chemv("L",&n, TAU, A, &lda, V, &ione, &c_zero, work, &ione);

    /* compute dtmp= X'*V */
    dtmp = magma_cblas_cdotc(n, work, ione, V, ione);
    /*
    dtmp = c_zero;
    for (magma_int_t j = 0; j < n; j++)
        dtmp = dtmp + MAGMA_C_CONJ(work[j]) * V[j];
    */

    /* compute 1/2 X'*V*t = 1/2*dtmp*tau  */
    dtmp = -dtmp * c_half * (*TAU);

    /* compute W=X-1/2VX'Vt = X - dtmp*V */
    blasf77_caxpy(&n, &dtmp, V, &ione, work, &ione);

    /* performs the symmetric rank 2 operation A := alpha*x*y' + alpha*y*x' + A */
    blasf77_cher2("L", &n, &c_neg_one, work, &ione, V, &ione, A, &lda);
}
