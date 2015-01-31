/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *     @author Raffaele Solca
 *
 *     @generated from zbulge_applyQ_v2.cpp normal z -> c, Fri Jan 30 19:00:18 2015
 *
 */

#include "common_magma.h"
#include "magma_bulge.h"
#include "magma_cbulgeinc.h"


////////////////////////////////////////////////////////////////////////////////////////////////////

#define dE(i,j)  (dE+(i) + ldde*(j))
#define V(j)     (V+(j))
#define T(j)     (T+(j))
/***************************************************************************
 *  Parallel apply Q2 from bulgechasing symetric matrices - static scheduling
 *  Lower case is treated
 **/
    /*
     * side == magmaLeft:
     *     meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so
     *     traverse Vs in reverse order (forward) from q_n to q_1 Also
     *     E is splitten by block of col over cores because each apply
     *     consist in a block of row (horizontal block)
     */
    /*
     * side == magmaRight:
     *     meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so
     *     traverse Vs in normal order (forward) from q_1 to q_n Also
     *     E is splitten by block of row over core because each apply
     *     consist in a block of col (vertical block)
     */
/***************************************************************************/
extern "C" magma_int_t
magma_cbulge_applyQ_v2(
    magma_side_t side,
    magma_int_t NE, magma_int_t N,
    magma_int_t NB, magma_int_t Vblksiz,
    magmaFloatComplex_ptr dE, magma_int_t ldde,
    magmaFloatComplex *V, magma_int_t ldv,
    magmaFloatComplex *T, magma_int_t ldt,
    magma_int_t *info)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t Vm, Vn, mt, nt;
    magma_int_t myrow, mycol, blkj, blki;
    magma_int_t blkid,vpos,tpos;
    magma_int_t firstrow, nbcolinvolvd;
    magma_int_t versionL  = 113;
    magma_int_t versionR  = 92;
    magma_int_t Vchunksiz = 10;
    *info=0;

    /* Quick return */
    if ( NE == 0 ) {
        return *info;
    }
    if ( N == 0 ) {
        return *info;
    }
    if ( NB == 0 ) {
        return *info;
    }
    /* ==========================================
     * some infos for developer
     * Initialisation and checking nb of cores
     * ==========================================*/
    /* we have 2 algo for left (113 114) and 2 algo for right (91 92)
     * which correspond to versionL versionR.
     * They are very similar (detail explained in tech report and matlab code)
     * however version 114 and 92 improve locality.
     * while version 113 is used in case WNATZ=1 (construct Q2) which allow
     * the construction to be done in an optimized way taking into
     * consideration that the matrix is Identity so making less flops.
     *
    */

    // Initialize streaming and events
    magma_device_sync();
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );

    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    magma_event_t myevent[2];
    cudaEventCreateWithFlags(&myevent[0],cudaEventDisableTiming);
    cudaEventCreateWithFlags(&myevent[1],cudaEventDisableTiming);



    // Azzam 21/11/2012
    // NOTE THAT dwork was of size 2*NE*Vblksiz+...
    // but I am thinking why not modifing it to NE*Vblksiz+...
    // BUT NO because the 2* is used because of making 2 streams working and so
    // they might be using dwork in parallel
    magmaFloatComplex *dwork, *dwork0, *dwork1, *dwvt0, *dwvt1;
    magmaFloatComplex *dT0, *dV0, *dT1, *dV1;
    magma_int_t lddv = ldv;
    magma_int_t lddt = ldt;
    magma_int_t lddw = 0;
    magma_int_t lddwork  = ((NE+31)/32)*32;
    magma_int_t dwVTsiz  = lddv*Vblksiz; // lddv*lddv + lddv*lddwork; (v2) // lddv*Vblksiz; (v1,v3)
    magma_int_t dworksiz = lddwork*Vblksiz;  // lddv*Vblksiz; (v2)   // NE*Vblksiz=lddwork*Vblksiz; (v1,v3)

    if (MAGMA_SUCCESS != magma_cmalloc( &dwork, 2*dworksiz + 2*dwVTsiz +  2*Vchunksiz* (Vblksiz* (lddv+lddt)) )) {
       printf ("!!!!  magma_cbulge_applyQ magma_alloc failed for: dwork\n" );
       exit(-1);
    }
    dwork0 = dwork;               // size = dworksiz;
    dwork1 = dwork0 + dworksiz;   // size = dworksiz;
    dwvt0  = dwork + 2*dworksiz;  // size = dwVTsiz;
    dwvt1  = dwvt0 + dwVTsiz;     // size = dwVTsiz;
    dV0    = dwork + 2*dworksiz + 2*dwVTsiz;
    dT0    = dV0 + Vchunksiz*Vblksiz*lddv;
    dV1    = dT0 + Vchunksiz*Vblksiz*lddt;
    dT1    = dV1 + Vchunksiz*Vblksiz*lddv;


    // make overlapped copy
    magma_int_t ncpy = 0;
    magma_int_t copyed=0, copyst=0;
    magma_int_t blkcnt,nothing, mysiz, flip, vld,tld, locpos;
    findVTsiz(N, NB, Vblksiz, &blkcnt, &nothing);

    flip = 0;

    // performance loss if the reflector are applied to a big number of eigenvectors (~10000)
    // => apply the reflectors to blocks of eigenvectors.
    //magma_int_t nr_bl = magma_ceildiv(NE,10000);        //nr of blocks
    magma_int_t sz_bl = NE; //magma_ceildiv(NE,nr_bl*64)*64; //maximum size of blocks (to have blocks of around the same size and multiple of 64)
    magma_int_t ib;                                      //size of current block


    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

    #ifdef ENABLE_DEBUG
    printf("  APPLY Q_v22 GPU with  N %d, NE %d,  NB %d, Vblksiz %d, versionL %d versionR %d  SIDE %c \n",
           N, NE, NB, Vblksiz, versionL, versionR, side);
    #endif

    /*
     * MagmamaLeft
     */
    if (side == MagmaLeft) {
        /*
         * Version 113:
         * loop over the block_col (nt) and for each find the
         * number of tiles (mt) in this block_col. then loop over mt, find
         * the size of the V's(Vm,Vn) and apply it to the corresponding
         * portion of E.
         */
        if ( versionL == 113 ) {
            nt  = magma_ceildiv((N-1),Vblksiz);
            for (blkj=nt-1; blkj >= 0; blkj--) {
                /* the index of the first row on the top of block (blkj) */
                firstrow = blkj * Vblksiz + 1;
                /*find the number of tile for this block */
                if ( blkj == nt-1 )
                    mt = magma_ceildiv( N -  firstrow,    NB);
                else
                    mt = magma_ceildiv( N - (firstrow+1), NB);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blki=mt; blki > 0; blki--) {
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow     = firstrow + (mt-blki)*NB;
                    mycol     = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }
                    /*calculate the pointer to the Vs and the Ts.
                     * Note that Vs and Ts have special storage done
                     * by the bulgechasing function*/
                    //printf("voici blkj %d blki %d  Vm %d  Vn %d mycol %d vpos %d \n",blkj,blki,Vm, Vn,mycol,vpos);
                    magma_bulge_findpos113(N, NB, Vblksiz, mycol, myrow, &blkid);
               
                    // COPY Vchunksiz Vs and Vchunksiz Ts to GPU and store it in dV0/dV1 and dT0/dT1
                    if (ncpy == 0) {
                        // flip = 1 for this.
                        copyst = 0;                             // meaning that copy will start copying from blkid =copyst
                        copyed = min(copyst+Vchunksiz, blkcnt); // meaning that copy will end copying at blkid =copyed-1==> next copy had to start at copyed
                        mysiz  = copyed-copyst;                 // the size of the chunk to be copied
                        if (mysiz > 0) {
                            ncpy = 1;
                            flip = 1;
                            vpos = copyst*Vblksiz*ldv;
                            tpos = copyst*Vblksiz*ldt;
                            vld  = mysiz * ldv;
                            tld  = mysiz * ldt;
                            magmablasSetKernelStream(stream[1]);
                            magma_csetmatrix_async(vld, Vblksiz, V(vpos), vld, dV1, vld, stream[1]);
                            magma_csetmatrix_async(tld, Vblksiz, T(tpos), tld, dT1, tld, stream[1]);
                            //printf("doing the first copy   of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV1 dT1\n",mysiz,copyst,copyed,vpos,tpos);
                        }
                    }
                   
                    if (blkid == copyst) {
                        flip   = ncpy % 2;
                        copyst = copyed;                             // meaning that copy will start copying from blkid =copyst
                        copyed = min(copyst+Vchunksiz, blkcnt); // meaning that copy will end copying at blkid =copyed-1==> next copy had to start at copyed
                        mysiz  = copyed-copyst;                 // the size of the chunk to be copied
                        //printf(" get to copy blkid %d blkid+(2*Vchunksiz) %d copyst %d copyed %d\n",blkid,blkid+(Vchunksiz),copyst,copyed);
                        if (mysiz > 0) {
                            ncpy = ncpy + 1;
                            vpos = copyst*Vblksiz*ldv;
                            tpos = copyst*Vblksiz*ldt;
                            vld  = mysiz * ldv;
                            tld  = mysiz * ldt;
                            if (flip == 0) { // now I am working on dV0 so copy the next and put it on dV1
                                //printf("doing overlapping copy of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV1 dT1\n",mysiz,copyst,copyed,vpos,tpos);
                                magmablasSetKernelStream(stream[1]);
                                magma_csetmatrix_async(vld, Vblksiz, V(vpos), vld, dV1, vld, stream[1]);
                                magma_csetmatrix_async(tld, Vblksiz, T(tpos), tld, dT1, tld, stream[1]);
                            } else { // now I am working on dV1 so copy the next and put it on dV0
                                //printf("doing overlapping copy of mysiz %2d copyst %2d copyed %2d vpos %8d tpos %8d into dV0 dT0\n",mysiz,copyst,copyed,vpos,tpos);
                                magmablasSetKernelStream(stream[0]);
                                magma_csetmatrix_async(vld, Vblksiz, V(vpos), vld, dV0, vld, stream[0]);
                                magma_csetmatrix_async(tld, Vblksiz, T(tpos), tld, dT0, tld, stream[0]);
                            }
                        }
                    }

                    if ((Vm > 0) && (Vn > 0)) {
                        locpos = blkid%Vchunksiz;
                        magma_int_t lcvpos   = locpos*Vblksiz*lddv;
                        magma_int_t lctpos   = locpos*Vblksiz*lddt;
                        //printf("voici blkj %d blki %d  Vm %d  Vn %d mycol %d locvpos %5d loctpos %5d  blkid %2d  using data in dV%1d dT%1d \n",blkj,blki,Vm, Vn,mycol,lcvpos,lctpos, blkid,flip,flip);
                        if (flip == 0) {
                            magmablasSetKernelStream(stream[0]);
                            magma_queue_wait_event( stream[0], myevent[1] );
                            for (magma_int_t i=0; i < NE; i += sz_bl) {
                                ib = min(sz_bl, NE-i);
                                lddw = min(lddwork,sz_bl);
                                //magma_clarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0+lcvpos, lddv, dT0+lctpos, lddt, dE(myrow,i), ldde, dwork0, lddw);
                                magma_clarfb_gpu_gemm( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0+lcvpos, lddv, dT0+lctpos, lddt, dE(myrow,i), ldde, dwork0, lddw, dwvt0, lddv);
                            }
                            magma_event_record( myevent[0], stream[0] );
                        } else {
                            magmablasSetKernelStream(stream[1]);
                            magma_queue_wait_event( stream[1], myevent[0] );
                            for (magma_int_t i=0; i < NE; i += sz_bl) {
                                ib = min(sz_bl, NE-i);
                                lddw = min(lddwork,sz_bl);
                                //magma_clarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV1+lcvpos, lddv, dT1+lctpos, lddt, dE(myrow,i), ldde, dwork1, lddw);
                                magma_clarfb_gpu_gemm( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV1+lcvpos, lddv, dT1+lctpos, lddt, dE(myrow,i), ldde, dwork1, lddw, dwvt1, lddv);
                            }
                            magma_event_record( myevent[1], stream[1] );
                        }
                    }  // end for (Vm &Vn) > 0
                } // end for blki
            } // end for blkj
        } // end if version=113
        /*
         * Version 114:
         * loop over the block_row (mt) and for each find diagonally the
         * number of tiles (nt) in this block_row. then loop over nt, find
         * the size of the V's(Vm,Vn) and apply it to the corresponding
         * portion of E.
         */
        else {
            mt    = magma_ceildiv((N-1),NB);
            for (blki = mt; blki > 0; blki--) {
                /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                nbcolinvolvd = min(N-1, blki*NB);
                /*find the number of tile for this block (diagonal row of tiles) */
                nt = magma_ceildiv(nbcolinvolvd,Vblksiz);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blkj = nt-1; blkj >= 0; blkj--) {
                    /* the index of the first row of the first col meaning
                     * the block on the top left (blki) */
                    firstrow = (mt-blki)*NB+1;
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow    = firstrow + blkj*Vblksiz;
                    mycol    = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }

                    if ((Vm > 0) && (Vn > 0)) {
                    /*calculate the pointer to the Vs and the Ts.
                     * Note that Vs and Ts have special storage done
                     * by the bulgechasing function*/
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_csetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, NULL);
                        magma_csetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, NULL);
                        //printf("voici blki %d  rownbm %d mycol %d  coled %d  blkid %d vpos %d  tpos %d\n", blki, rownbm, mycol, coled, blkid, vpos, tpos);
                        for (magma_int_t i=0; i < NE; i += sz_bl) {
                            ib = min(sz_bl, NE-i);
                            magma_clarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise, Vm, ib, Vn, dV0, lddv, dT0, lddt, dE(myrow,i), ldde, dwork, NE);
                        }
                    } // end for (Vm &Vn) > 0
                } // end for blkj
            } // end for blki
        } // end version 114
    } // end LEFT
    /*
     * MagmaRight
     */
    else {
        /*
         * Version 91:
         */
        if ( versionR == 91 ) {
            nt  = magma_ceildiv((N-1),Vblksiz);
            for (blkj=0; blkj < nt; blkj++) {
                /* the index of the first myrow on the top of block (blkj) */
                firstrow = blkj * Vblksiz + 1;
                /*find the number of tile for this block */
                if ( blkj == nt-1 )
                    mt = magma_ceildiv( N -  firstrow,    NB);
                else
                    mt = magma_ceildiv( N - (firstrow+1), NB);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blki=1; blki <= mt; blki++) {
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow  = firstrow + (mt-blki)*NB;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( (blkj == nt-1) && (blki == mt) ) {
                        Vn = min( Vblksiz, Vm );
                    } else {
                        Vn = min( Vblksiz, Vm-1 );
                    }
                    mycol     = blkj*Vblksiz;
                    if ((Vm > 0) && (Vn > 0)) {
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_csetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, NULL);
                        magma_csetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, NULL);
                        magma_clarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, Vm, Vn, dV0, lddv, dT0, lddt, dE(0, myrow), ldde, dwork, NE);
                    } // end for (Vm &Vn) > 0
                } // end for blki
            } // end fo blkj
        } // end of version 91
        /*
         * Version 92:
         */
        else {
            mt    = magma_ceildiv((N-1),NB);
            for (blki = 1; blki <= mt; blki++) {
                /* nbcolinvolvd = number of column corresponding to this block_row (blki) */
                nbcolinvolvd = min(N-1, blki*NB);
                /*find the number of tile for this block (diagonal row of tiles) */
                nt = magma_ceildiv(nbcolinvolvd,Vblksiz);
                /*loop over the tiles find the size of the Vs and apply it */
                for (blkj = 0; blkj < nt; blkj++) {
                    /* the index of the first row of the first col meaning
                     * the block on the top left (blki) */
                    firstrow = (mt-blki)*NB+1;
                    /*calculate the size of each losange of Vs= (Vm,Vn)*/
                    myrow    = firstrow + blkj*Vblksiz;
                    mycol    = blkj*Vblksiz;
                    Vm = min( NB+Vblksiz-1, N-myrow);
                    if ( ( blkj == nt-1 ) && ( blki == mt ) ) {
                        Vn = min (Vblksiz, Vm);
                    } else {
                        Vn = min (Vblksiz, Vm-1);
                    }
                    if ((Vm > 0) && (Vn > 0)) {
                        /*calculate the pointer to the Vs and the Ts.
                         * Note that Vs and Ts have special storage done
                         * by the bulgechasing function*/
                        magma_bulge_findVTpos(N, NB, Vblksiz, mycol, myrow, ldv, ldt, &vpos, &tpos);
                        magma_csetmatrix_async(Vm, Vn, V(vpos), ldv, dV0, lddv, NULL);
                        magma_csetmatrix_async(Vn,  Vn, T(tpos), ldt, dT0, lddt, NULL);
                        magma_clarfb_gpu( MagmaRight, MagmaNoTrans, MagmaForward, MagmaColumnwise, NE, Vm, Vn, dV0, lddv, dT0, lddt, dE(0, myrow), ldde, dwork, NE);
                    } // end for (Vm &Vn) > 0
                } //end for blkj
            } // end for blki
        } //end of version 92
    } // end RIGHT


    magma_device_sync();
    magmablasSetKernelStream( orig_stream );
    magma_event_destroy( myevent[0] );
    magma_event_destroy( myevent[1] );
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_free(dwork);


    return *info;
}
#undef V
#undef T
#undef dE
////////////////////////////////////////////////////////////////////////////////////////////////////
