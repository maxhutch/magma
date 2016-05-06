/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar

       @generated from src/core_zhbtype3cb.cpp normal z -> s, Mon May  2 23:30:20 2016

*/
#include "magma_internal.h"


#define A(m,n)   (A + lda * (n) + ((m)-(n)))
#define V(m)     (V + (m))
#define TAU(m)   (TAU + (m))

/***************************************************************************//**
 *
 * @ingroup magma_float
 *
 *  magma_ssbtype3cb is a kernel that will operate on a region (triangle) of data
 *  bounded by st and ed. This kernel apply a left+right update on the symmetric
 *  triangle.  Note that this kernel is very similar to type1 but does not do an
 *  elimination.
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
 * @param[in] V
 *          float array, dimension 2*n if eigenvalue only
 *          requested or (ldv*blkcnt*Vblksiz) if Eigenvectors requested
 *          The Householder reflectors are stored in this array.
 *
 * @param[in] TAU
 *          float array, dimension (n).
 *          The scalar factors of the Householder reflectors are stored
 *          in this array.
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

/***************************************************************************//**
 *          TYPE 3-BAND Lower-columnwise-Householder
 ***************************************************************************/
extern "C" void
magma_ssbtype3cb(magma_int_t n, magma_int_t nb,
                float *A, magma_int_t lda,
                float *V, magma_int_t ldv, 
                float *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                float *work)
{
    magma_int_t len;
    magma_int_t vpos, taupos;
    //magma_int_t blkid, tpos;


    if ( wantz == 0 ) {
        vpos   = (sweep%2)*n + st;
        taupos = (sweep%2)*n + st;
    } else {
        magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep, st, ldv, &vpos, &taupos);
        //findVTpos(n, nb, Vblksiz, sweep, st, &vpos, &taupos, &tpos, &blkid);
    }

    len = ed-st+1;
    /* Apply left and right on A(st:ed,st:ed)*/
    magma_slarfy(len, A(st,st), lda-1, V(vpos), TAU(taupos), work);
    return;
}
/***************************************************************************/
#undef A
#undef V
#undef TAU
