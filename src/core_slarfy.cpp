/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *
 *     @generated from src/core_zlarfy.cpp normal z -> s, Mon May  2 23:30:20 2016
 *
 */
#include "magma_internal.h"
#include "magma_bulge.h"

/***************************************************************************//**
 *
 * @ingroup magma_saux3
 *
 *  magma_slarfy applies an elementary reflector, or Householder matrix, H,
 *  to a n-by-n symmetric matrix C, from both the left and the right.
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
 *          REAL array, dimension (lda, n)
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
magma_slarfy(
    magma_int_t n,
    float *A, magma_int_t lda,
    const float *V, const float *TAU,
    float *work)
{
    /*
    work (workspace) float real array, dimension n
    */

    static magma_int_t ione = 1;
    static float c_zero   =  MAGMA_S_ZERO;
    static float c_neg_one=  MAGMA_S_NEG_ONE;
    static float c_half   =  MAGMA_S_HALF;
    float dtmp;

    /* X = AVtau */
    blasf77_ssymv("L",&n, TAU, A, &lda, V, &ione, &c_zero, work, &ione);

    /* compute dtmp= X'*V */
    dtmp = magma_cblas_sdot(n, work, ione, V, ione);
    /*
    dtmp = c_zero;
    for (magma_int_t j = 0; j < n; j++)
        dtmp = dtmp + MAGMA_S_CONJ(work[j]) * V[j];
    */

    /* compute 1/2 X'*V*t = 1/2*dtmp*tau  */
    dtmp = -dtmp * c_half * (*TAU);

    /* compute W=X-1/2VX'Vt = X - dtmp*V */
    blasf77_saxpy(&n, &dtmp, V, &ione, work, &ione);

    /* performs the symmetric rank 2 operation A := alpha*x*y' + alpha*y*x' + A */
    blasf77_ssyr2("L", &n, &c_neg_one, work, &ione, V, &ione, A, &lda);
}
