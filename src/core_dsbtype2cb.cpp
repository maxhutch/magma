/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar

       @generated from src/core_zhbtype2cb.cpp normal z -> d, Mon May  2 23:30:20 2016

*/
#include "magma_internal.h"


#define A(m,n)   (A + lda * (n) + ((m)-(n)))
#define V(m)     (V + (m))
#define TAU(m)   (TAU + (m))

/***************************************************************************//**
 *
 * @ingroup magma_double
 *
 *  magma_dsbtype2cb is a kernel that will operate on a region (triangle) of data
 *  bounded by st and ed. This kernel apply the right update remaining from the
 *  type1 and this later will create a bulge so it eliminate the first column of
 *  the created bulge and do the corresponding Left update.
 *
 *  All detail are available on technical report or SC11 paper.
 *  Azzam Haidar, Hatem Ltaief, and Jack Dongarra. 2011.
 *  Parallel reduction to condensed forms for symmetric eigenvalue problems
 *  using aggregated fine-grained and memory-aware kernels. In Proceedings
 *  of 2011 International Conference for High Performance Computing,
 *  Networking, Storage and Analysis (SC '11). ACM, New York, NY, USA,
 *  Article 8, 11 pages.
 *  http://doi.acm.org/10.1145/2063384.2063394
 *
 *******************************************************************************
 *
 * @param[in] n
 *          The order of the matrix A.
 *
 * @param[in] nb
 *          The size of the band.
 *
 * @param[in, out] A
 *          A pointer to the matrix A of size (2*nb+1)-by-n.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A. lda >= max(1,2*nb+1)
 *
 * @param[in, out] V
 *          double array, dimension 2*n if eigenvalue only
 *          requested or (LDV*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors of the previous type 1 are used here
 *          to continue update then new one are generated to eliminate the
 *          bulge and stored in this array.
 *
 * @param[in, out] TAU
 *          double array, dimension (n).
 *          The scalar factors of the Householder reflectors of the previous
 *          type 1 are used here to continue update then new one are generated
 *          to eliminate the bulge and stored in this array.
 *
 * @param[in] st
 *          A pointer to the start index where this kernel will operate.
 *
 * @param[in] ed
 *          A pointer to the end index where this kernel will operate.
 *
 * @param[in] sweep
 *          The sweep number that is eliminated. it serve to calculate the
 *          pointer to the position where to store the Vs and Ts.
 *
 * @param[in] Vblksiz
 *          constant which correspond to the blocking used when applying the Vs.
 *          it serve to calculate the pointer to the position where to store the
 *          Vs and Ts.
 *
 * @param[in] wantz
 *          constant which indicate if Eigenvalue are requested or both
 *          Eigenvalue/Eigenvectors.
 *
 * @param[in] work
 *          Workspace of size nb.
 *
 *******************************************************************************
 *
 * @return
 *          \retval MAGMA_SUCCESS successful exit
 *          \retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/

/***************************************************************************
 *          TYPE 2-BAND Lower-columnwise-Householder
 ***************************************************************************/
extern "C" void
magma_dsbtype2cb(magma_int_t n, magma_int_t nb,
                double *A, magma_int_t lda,
                double *V, magma_int_t ldv,
                double *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep,
                magma_int_t Vblksiz, magma_int_t wantz,
                double *work)
{
    double ctmp;
    magma_int_t J1, J2, len, lem, ldx;
    magma_int_t vpos, taupos;
    //magma_int_t blkid, tpos;
    magma_int_t ione = 1;
    const double c_one    =  MAGMA_D_ONE;

    if ( wantz == 0 ) {
        vpos   = (sweep%2)*n + st;
        taupos = (sweep%2)*n + st;
    } else {
        //findVTpos(n, nb, Vblksiz, sweep, st, &vpos, &taupos, &tpos, &blkid);
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep, st, ldv, &vpos, &taupos);
    }

    ldx = lda-1;
    J1  = ed+1;
    J2  = min(ed+nb,n-1);
    len = ed-st+1;
    lem = J2-J1+1;

    if ( lem > 0 ) {
        /* Apply remaining right commming from the top block */
        lapackf77_dlarfx("R", &lem, &len, V(vpos), TAU(taupos), A(J1, st), &ldx, work);
    }

    if ( lem > 1 ) {
        if ( wantz == 0 ) {
            vpos   = (sweep%2)*n + J1;
            taupos = (sweep%2)*n + J1;
        } else {
            magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep, J1, ldv, &vpos, &taupos);
            //findVTpos(n,nb,Vblksiz,sweep,J1, &vpos, &taupos, &tpos, &blkid);
        }

        /* Remove the first column of the created bulge */
        *V(vpos)  = c_one;
        
        //magma_int_t lem2=lem-1;
        //blasf77_dcopy( &lem2, A(ed+2, st), &ione, V(vpos+1), &ione );
        memcpy(V(vpos+1), A(J1+1, st), (lem-1)*sizeof(double));
        memset(A(J1+1, st), 0, (lem-1)*sizeof(double));

        /* Eliminate the col at st */
        lapackf77_dlarfg( &lem, A(J1, st), V(vpos+1), &ione, TAU(taupos) );

        /*
         * Apply left on A(J1:J2,st+1:ed)
         * We decrease len because we start at col st+1 instead of st.
         * col st is the col that has been revomved;
         */
        len = len-1;
        ctmp = MAGMA_D_CONJ(*TAU(taupos));
        lapackf77_dlarfx("L", &lem, &len, V(vpos),  &ctmp, A(J1, st+1), &ldx, work);
    }
    return;
}
/***************************************************************************/
#undef A
#undef V
#undef TAU
