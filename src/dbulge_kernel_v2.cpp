/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @generated from zbulge_kernel_v2.cpp normal z -> d, Fri Jan 30 19:00:17 2015
 *
 */

#include "common_magma.h"
#include "magma_bulge.h"

#define PRECISION_d

inline static void
magma_dlarfxsym_v2(
    magma_int_t n,
    double *A, magma_int_t lda,
    double *V, double *TAU,
    double *work);
///////////////////////////////////////////////////////////

inline static void
magma_dlarfxsym_v2(
    magma_int_t n,
    double *A, magma_int_t lda,
    double *V, double *TAU,
    double *work)
{
/*
    WORK (workspace) double real array, dimension N
*/

    magma_int_t ione = 1;
    double dtmp;
    double c_zero   =  MAGMA_D_ZERO;
    double c_neg_one=  MAGMA_D_NEG_ONE;
    double c_half   =  MAGMA_D_HALF;

    /* X = AVtau */
    blasf77_dsymv("L",&n, TAU, A, &lda, V, &ione, &c_zero, work, &ione);

    /* compute dtmp= X'*V */
    dtmp = magma_cblas_ddot(n, work, ione, V, ione);

    /* compute 1/2 X'*V*t = 1/2*dtmp*tau  */
    dtmp = -dtmp * c_half * (*TAU);

    /* compute W=X-1/2VX'Vt = X - dtmp*V */
    blasf77_daxpy(&n, &dtmp, V, &ione, work, &ione);

    /* performs the symmetric rank 2 operation A := alpha*x*y' + alpha*y*x' + A */
    blasf77_dsyr2("L", &n, &c_neg_one, work, &ione, V, &ione, A, &lda);
}

///////////////////////////////////////////////////////////
//                  TYPE 1-BAND Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(i,j)   &(A[((i)-(j)) + lda*((j)-1)])
#define V(i)     &(V[(i)])
#define TAU(i)   &(TAU[(i)])
extern "C" void
magma_dtrdtype1cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *V, magma_int_t ldv,
    double *TAU,
    magma_int_t st, magma_int_t ed,
    magma_int_t sweep, magma_int_t Vblksiz,
    double *work)
{
/*
    WORK (workspace) double real array, dimension N
*/

    magma_int_t ione = 1;
    magma_int_t vpos, taupos, len, len2;

    double c_one    =  MAGMA_D_ONE;

    magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, st-1, ldv, &vpos, &taupos);
    //printf("voici vpos %d taupos %d  tpos %d  blkid %d \n", vpos, taupos, tpos, blkid);

    len     = ed-st+1;
    *V(vpos)  = c_one;

    len2 = len-1;
    blasf77_dcopy( &len2, A(st+1, st-1), &ione, V(vpos+1), &ione );
    //memcpy(V(vpos+1), A(st+1, st-1), (len-1)*sizeof(double));
    memset(A(st+1, st-1), 0, (len-1)*sizeof(double));

    /* Eliminate the col  at st-1 */
    lapackf77_dlarfg( &len, A(st, st-1), V(vpos+1), &ione, TAU(taupos) );

    /* apply left and right on A(st:ed,st:ed)*/
    magma_dlarfxsym_v2(len, A(st,st), lda-1, V(vpos), TAU(taupos), work);
}
#undef A
#undef V
#undef TAU

///////////////////////////////////////////////////////////
//                  TYPE 1-LPK Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(i,j)   &(A[((i)-(j)) + lda*((j)-1)])
#define V(i)     &(V[(i)])
#define TAU(i)   &(TAU[(i)])
extern "C" void
magma_dtrdtype2cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *V, magma_int_t ldv,
    double *TAU,
    magma_int_t st, magma_int_t ed,
    magma_int_t sweep, magma_int_t Vblksiz,
    double *work)
{
    /*
     WORK (workspace) double real array, dimension NB
    */

    magma_int_t ione = 1;
    magma_int_t vpos, taupos;

    double conjtmp;

    double c_one = MAGMA_D_ONE;

    magma_int_t ldx = lda-1;
    magma_int_t len = ed - st + 1;
    magma_int_t lem = min(ed+nb, n) - ed;
    magma_int_t lem2;
    
    if (lem > 0) {
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, st-1, ldv, &vpos, &taupos);
        /* apply remaining right coming from the top block */
        lapackf77_dlarfx("R", &lem, &len, V(vpos), TAU(taupos), A(ed+1, st), &ldx, work);
    }
    if (lem > 1) {
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, ed, ldv, &vpos, &taupos);

        /* remove the first column of the created bulge */
        *V(vpos)  = c_one;
        //memcpy(V(vpos+1), A(ed+2, st), (lem-1)*sizeof(double));
        lem2 = lem-1;
        blasf77_dcopy( &lem2, A(ed+2, st), &ione, V(vpos+1), &ione );
        memset(A(ed+2, st),0,(lem-1)*sizeof(double));

        /* Eliminate the col at st */
        lapackf77_dlarfg( &lem, A(ed+1, st), V(vpos+1), &ione, TAU(taupos) );

        /* apply left on A(J1:J2,st+1:ed) */
        len = len-1; /* because we start at col st+1 instead of st. col st is the col that has been removed; */
        conjtmp = MAGMA_D_CNJG(*TAU(taupos));
        lapackf77_dlarfx("L", &lem, &len, V(vpos),  &conjtmp, A(ed+1, st+1), &ldx, work);
    }
}
#undef A
#undef V
#undef TAU

///////////////////////////////////////////////////////////
//                  TYPE 1-LPK Householder
///////////////////////////////////////////////////////////
//// add -1 because of C
#define A(i,j)   &(A[((i)-(j)) + lda*((j)-1)])
#define V(i)     &(V[(i)])
#define TAU(i)   &(TAU[(i)])
extern "C" void
magma_dtrdtype3cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *V, magma_int_t ldv,
    double *TAU,
    magma_int_t st, magma_int_t ed,
    magma_int_t sweep, magma_int_t Vblksiz,
    double *work)
{
    /*
     WORK (workspace) double real array, dimension N
     */

    magma_int_t vpos, taupos;

    magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep-1, st-1, ldv, &vpos, &taupos);

    magma_int_t len = ed-st+1;

    /* apply left and right on A(st:ed,st:ed)*/
    magma_dlarfxsym_v2(len, A(st,st), lda-1, V(vpos), TAU(taupos), work);
}
#undef A
#undef V
#undef TAU
///////////////////////////////////////////////////////////
