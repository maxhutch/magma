/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @precisions normal z -> s d c
 *
 */

#include "common_magma.h"
#include "magma_zbulgeinc.h"


////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
#define dE(m,n)  (dE+(m) + LDE*(n))
#define dV(m)    (dV+(m))
#define dT(m)    (dT+(m))
#define E(m,n)   &(E[(m) + LDE*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])

extern "C" void magma_zbulge_applyQ(
    magma_int_t WANTZ, magma_side_t SIDE, magma_int_t NE, magma_int_t N, magma_int_t NB,
    magma_int_t Vblksiz, magmaDoubleComplex *E, magma_int_t LDE,
    magmaDoubleComplex *V, magmaDoubleComplex *TAU, magmaDoubleComplex *T,
    magma_int_t *INFO, magmaDoubleComplex *dV, magmaDoubleComplex *dT,
    magmaDoubleComplex *dE, magma_int_t copytype )
{
    //%===========================
    //%   local variables
    //%===========================
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    
    magma_int_t LDT, LDV, firstcolj;
    magma_int_t bg, nbGblk, rownbm, k, m, n;
    magma_int_t st, ed, fst, vlen, vnb, colj, len;
    magma_int_t blkid, vpos, taupos, tpos;
    //magmaDoubleComplex *WORK;
    magma_int_t LWORK;
    magma_int_t  cur_blksiz, avai_blksiz, ncolinvolvd;
    magma_int_t  nbgr, colst, coled, versionL, versionR;
    magma_int_t blkcnt=-1;

    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    *INFO=0;
    versionL = 113;
    versionR = 92;
    LDT      = Vblksiz;
    LDV      = NB+Vblksiz-1;
    //blklen = LDV*Vblksiz;
    nbGblk   = plasma_ceildiv((N-1), Vblksiz);
    //magma_zmalloc_cpu( &WORK, LWORK );

    /* find the size of the matrix T V*/
    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);
    /* Copy E & V & T to the GPU in dE and dV and dT
     * depending on copytype:
     * 1: mean copy only V
     * 2: mean copy V and T
     * 3: mean copy V, T and E
     * */
    if (copytype > 0) magma_zsetmatrix( LDV, blkcnt*Vblksiz, V, LDV, dV, LDV );
    if (copytype > 1) magma_zsetmatrix( LDT, blkcnt*Vblksiz, T, LDT, dT, LDT );
    if (copytype > 2) magma_zsetmatrix( N, NE, E, N, dE, N );
    magmaDoubleComplex *dwork;
    //ldwork  = NE;
    LWORK   = 2*N*max(Vblksiz, 64);
    if (MAGMA_SUCCESS != magma_zmalloc( &dwork, LWORK )) {
        printf ("!!!!  magma_zbulge_applyQ magma_alloc failed for: dwork\n" );
        exit(-1);
    }

    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

    /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update.
     *         So we use the reverse order from small q to large one,
     *         so from q_n to q_1 so Left update to Identity.
     *         Use versionL 113 because in 114 we need to update the whole matrix and not in icreasing order.
     * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
     * */
    if (WANTZ == 1) {
        versionL=113;
        SIDE = MagmaLeft;
        //set the matrix to Identity here to avoid copying it from the CPU
        magmablas_zlaset( MagmaFull, N, N, c_zero, c_one, dE, N );
    }
    


    printf("  APPLY Q_v115 GPU with  N %d   NB %d   Vblksiz %d SIDE %c versionL %d versionR %d WANTZ %d \n",
           (int) N, (int) NB, (int) Vblksiz, SIDE, (int) versionL, (int) versionR, (int) WANTZ);


#if defined(USESTREAM)
    magma_int_t N2=N/2;
    magma_int_t N1=N-N2;
    printf("using stream\n");
    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
#endif
    

    if (SIDE == MagmaLeft) {
        if (versionL == 113) {
            for (bg = nbGblk; bg > 0; bg--) {
                firstcolj = (bg-1)*Vblksiz + 1;
                if (bg == nbGblk)
                    rownbm = plasma_ceildiv((N-(firstcolj)), NB);  // last blk has size=1 used for complex to handle A(N,N-1)
                else
                    rownbm = plasma_ceildiv((N-(firstcolj+1)), NB);
                
                for (m = rownbm; m > 0; m--) {
                    vlen = 0;
                    vnb  = 0;
                    colj = (bg-1)*Vblksiz; // for k=0; I compute the fst and then can remove it from the loop
                    fst  = (rownbm -m)*NB+colj +1;
                    for (k=0; k < Vblksiz; k++) {
                        colj = (bg-1)*Vblksiz + k;
                        st   = (rownbm -m)*NB+colj +1;
                        ed   = min(st+NB-1, N-1);
                        if (st > ed) break;
                        if ((st == ed) && (colj != N-2)) break;
                        vlen=ed-fst+1;
                        vnb=k+1;
                    }
                    colst     = (bg-1)*Vblksiz;
                    findVTpos(N, NB, Vblksiz, colst, fst, &vpos, &taupos, &tpos, &blkid);
                    printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n", (int) bg, (int) m, (int) vlen, (int) vnb, (int) colst+1, (int) vpos+1, (int) taupos+1);
                    if ((vlen > 0) && (vnb > 0)) {
                        if (WANTZ == 1) {
                            len =  N-colst;
                            magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, vlen, len, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,colst), LDE, dwork, len);
                        } else {
                            magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, vlen, NE, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,0), LDE, dwork, NE);
                        }
                    }
                }
            }
        } else if (versionL == 114) {
            rownbm = plasma_ceildiv((N-1), NB);
            for (m = rownbm; m > 0; m--) {
                ncolinvolvd = min(N-1, m*NB);
                avai_blksiz=min(Vblksiz, ncolinvolvd);
                nbgr = plasma_ceildiv(ncolinvolvd, avai_blksiz);
                for (n = nbgr; n > 0; n--) {
                    vlen = 0;
                    vnb  = 0;
                    cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
                    colst = (n-1)*avai_blksiz;
                    coled = colst + cur_blksiz -1;
                    fst   = (rownbm -m)*NB+colst +1;
                    for (colj=colst; colj <= coled; colj++) {
                        st = (rownbm -m)*NB+colj +1;
                        ed = min(st+NB-1, N-1);
                        if (st > ed) break;
                        if ((st == ed) && (colj != N-2)) break;
                        vlen=ed-fst+1;
                        vnb=vnb+1;
                    }
                    findVTpos(N, NB, Vblksiz, colst, fst, &vpos, &taupos, &tpos, &blkid);
                    //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n", bg, m, vlen, vnb, colst+1, vpos+1, taupos+1);
                    if ((vlen > 0) && (vnb > 0)) {
                        #if defined(USESTREAM)
                        magmablasSetKernelStream(stream[0]);
                        magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, vlen, N1, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,0), LDE, dwork, N1);
                        magmablasSetKernelStream(stream[1]);
                        magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, vlen, N2, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,N1), LDE, &dwork[N1*Vblksiz], N2);
                        #else
                        magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, vlen, NE, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(fst,0), LDE, dwork, NE);
                        #endif
                    }
                }
            }
        }
    } else if (SIDE == MagmaRight) {
        if (versionR == 91) {
            for (bg =1; bg <= nbGblk; bg++) {
                firstcolj = (bg-1)*Vblksiz + 1;
                rownbm    = plasma_ceildiv((N-(firstcolj+1)), NB);
                if (bg == nbGblk) rownbm    = plasma_ceildiv((N-(firstcolj)), NB);  // last blk has size=1 used for complex to handle A(N,N-1)
                for (m = 1; m <= rownbm; m++) {
                    vlen = 0;
                    vnb  = 0;
                    // for k=0; I compute the fst and then can remove it from the loop
                    colj = (bg-1)*Vblksiz;
                    fst  = (rownbm -m)*NB+colj +1;
                    for (k=0; k < Vblksiz; k++) {
                        colj = (bg-1)*Vblksiz + k;
                        st   = (rownbm -m)*NB+colj +1;
                        ed   = min(st+NB-1, N-1);
                        if (st > ed) break;
                        if ((st == ed) && (colj != N-2)) break;
                        vlen=ed-fst+1;
                        vnb=k+1;
                    }
                    colj     = (bg-1)*Vblksiz;
                    findVTpos(N, NB, Vblksiz, colj, fst, &vpos, &taupos, &tpos, &blkid);
                    //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n", bg, m, vlen, vnb, colj, vpos, taupos);
                    if ((vlen > 0) && (vnb > 0)) {
                        #if defined(USESTREAM)
                        magmablasSetKernelStream(stream[0]);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, N1, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(0, fst), LDE, dwork, N1);
                        magmablasSetKernelStream(stream[1]);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, N2, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(N1, fst), LDE, &dwork[N1*Vblksiz], N2);
                        #else
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(0, fst), LDE, dwork, NE);
                        #endif
                    }
                }
            }
        } else if (versionR == 92) {
            rownbm = plasma_ceildiv((N-1), NB);
            for (m = 1; m <= rownbm; m++) {
                ncolinvolvd = min(N-1, m*NB);
                avai_blksiz=min(Vblksiz, ncolinvolvd);
                nbgr = plasma_ceildiv(ncolinvolvd, avai_blksiz);
                for (n = 1; n <= nbgr; n++) {
                    vlen = 0;
                    vnb  = 0;
                    cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
                    colst = (n-1)*avai_blksiz;
                    coled = colst + cur_blksiz -1;
                    fst   = (rownbm -m)*NB+colst +1;
                    for (colj=colst; colj <= coled; colj++) {
                        st = (rownbm -m)*NB+colj +1;
                        ed = min(st+NB-1, N-1);
                        if (st > ed) break;
                        if ((st == ed) && (colj != N-2)) break;
                        vlen=ed-fst+1;
                        vnb=vnb+1;
                    }
                    findVTpos(N, NB, Vblksiz, colst, fst, &vpos, &taupos, &tpos, &blkid);
                    if ((vlen > 0) && (vnb > 0)) {
                        #if defined(USESTREAM)
                        magmablasSetKernelStream(stream[0]);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, N1, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(0, fst), LDE, dwork, N1);
                        magmablasSetKernelStream(stream[1]);
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, N2, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(N1, fst), LDE, &dwork[N1*Vblksiz], N2);
                        #else
                        magma_zlarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, vlen, vnb, dV(vpos), LDV, dT(tpos), LDT, dE(0, fst), LDE, dwork, NE);
                        #endif
                    }
                }
            }
        }
    } else {
            printf("ERROR SIDE %d\n", SIDE);
    }

#if defined(USESTREAM)
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
#endif
    magmablasSetKernelStream( orig_stream );
}

#undef E
#undef V
#undef TAU
#undef T
#undef dE
#undef dV
#undef dT
////////////////////////////////////////////////////////////////////////////////////////////////////
