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
 */

#include "common_magma.h"
#include "magma_bulge.h"

#define applyQver 113


extern "C" {
    /////////////////////////////////////////
    magma_int_t magma_sbulge_get_lq2(magma_int_t n, magma_int_t threads)
    {
        magma_int_t nb = magma_get_sbulge_nb(n, threads);
        magma_int_t Vblksiz = magma_sbulge_get_Vblksiz(n, nb, threads);
        magma_int_t ldv = nb + Vblksiz;
        magma_int_t ldt = Vblksiz;
        return magma_bulge_get_blkcnt(n, nb, Vblksiz) * Vblksiz * (ldt + ldv + 1);
    }

    magma_int_t magma_dbulge_get_lq2(magma_int_t n, magma_int_t threads)
    {
        magma_int_t nb = magma_get_dbulge_nb(n, threads);
        magma_int_t Vblksiz = magma_dbulge_get_Vblksiz(n, nb, threads);
        magma_int_t ldv = nb + Vblksiz;
        magma_int_t ldt = Vblksiz;
        return magma_bulge_get_blkcnt(n, nb, Vblksiz) * Vblksiz * (ldt + ldv + 1);
    }

    magma_int_t magma_cbulge_get_lq2(magma_int_t n, magma_int_t threads)
    {
        magma_int_t nb = magma_get_cbulge_nb(n, threads);
        magma_int_t Vblksiz = magma_cbulge_get_Vblksiz(n, nb, threads);
        magma_int_t ldv = nb + Vblksiz;
        magma_int_t ldt = Vblksiz;
        return magma_bulge_get_blkcnt(n, nb, Vblksiz) * Vblksiz * (ldt + ldv + 1);
    }

    magma_int_t magma_zbulge_get_lq2(magma_int_t n, magma_int_t threads)
    {
        magma_int_t nb = magma_get_zbulge_nb(n, threads);
        magma_int_t Vblksiz = magma_zbulge_get_Vblksiz(n, nb, threads);
        magma_int_t ldv = nb + Vblksiz;
        magma_int_t ldt = Vblksiz;
        return magma_bulge_get_blkcnt(n, nb, Vblksiz) * Vblksiz * (ldt + ldv + 1);
    }

    //////////////////////////////////////////////////
    // Auxiliary functions for 2-stage eigensolvers //
    //////////////////////////////////////////////////

    void cmp_vals(int n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2)
    {
        int i;
        double curv, maxv, sumv;

        maxv = 0.0;
        sumv = 0.0;
        for (i = 0; i < n; ++i) {

            curv = fabs( wr1[i] - wr2[i]);
            sumv += curv;
            if (maxv < curv) maxv = curv;
        }

        *nrmI = maxv;
        *nrm1 = sumv;
        *nrm2 = sqrt( sumv );
    }

    void magma_bulge_findpos113(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *myblkid)
    {
        magma_int_t prevGblkid, mastersweep;
        magma_int_t locblknb   = 0;
        magma_int_t prevblkcnt = 0;
        magma_int_t myblknb    = 0;
        magma_int_t lstGblkid  = magma_ceildiv((n-1),Vblksiz)-1;//nbGblk-1;
        magma_int_t myGblkid   = sweep/Vblksiz;

        // go forward and for each Gblk(a column of blocks) before my Gblk compute its number of blk.

        //===================================================
        for (prevGblkid = lstGblkid; prevGblkid > myGblkid; prevGblkid--)
        {
            mastersweep  = prevGblkid * Vblksiz;
            if(prevGblkid==lstGblkid)
                locblknb = magma_ceildiv((n-(mastersweep+1)),nb);
            else
                locblknb = magma_ceildiv((n-(mastersweep+2)),nb);
            prevblkcnt   = prevblkcnt + locblknb;
        }
        //===================================================
        /*
        // for best performance, the if condiiton inside the loop
        // is only for prevblkid==lastblkid so I can unroll this
        // out of the loop and so remove the if condition.
        //===================================================
        // for prevGblkid==lstGblkid
        mastersweep  = lstGblkid * Vblksiz;
        locblknb     = magma_ceildiv((n-(mastersweep+1)),nb);
        prevblkcnt   = prevblkcnt + locblknb;
        // the remaining of the loop
        for (prevGblkid = lstGblkid-1; prevGblkid > myGblkid; prevGblkid--)
        {
            mastersweep  = prevGblkid * Vblksiz;
            locblknb     = magma_ceildiv((n-(mastersweep+2)),nb);
            prevblkcnt   = prevblkcnt + locblknb;
        }
        //===================================================
        */
        myblknb = magma_ceildiv((st-sweep),nb);
        *myblkid    = prevblkcnt + myblknb -1;
        //printf("voici sweep %d  lstGblkid %d  myGblkid %d  prevcnt %d  myblknb %d  myblkid %d\n", sweep, lstGblkid,myGblkid,prevblkcnt,myblknb,*myblkid );
    }



    void magma_bulge_findpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *myblkid)
    {
        magma_int_t locblknb, prevblkcnt, prevGblkid;
        magma_int_t myblknb, nbprevGblk, mastersweep;

        locblknb = 0;
        prevblkcnt   = 0;
        myblknb  = 0;

        nbprevGblk = sweep/Vblksiz;
        for (prevGblkid = 0; prevGblkid < nbprevGblk; prevGblkid++)
        {
            mastersweep  = prevGblkid * Vblksiz;
            locblknb = magma_ceildiv((n-(mastersweep+2)),nb);
            prevblkcnt   = prevblkcnt + locblknb;
        }
        myblknb = magma_ceildiv((st-sweep),nb);
        *myblkid    = prevblkcnt + myblknb -1;
    }

    void magma_bulge_findVTAUpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv,
                                 magma_int_t *Vpos, magma_int_t *TAUpos)
    {
        magma_int_t myblkid;
        magma_int_t locj = sweep%Vblksiz;

        if(applyQver==113)
            magma_bulge_findpos113(n, nb, Vblksiz, sweep, st, &myblkid);
        else
            magma_bulge_findpos(n, nb, Vblksiz, sweep, st, &myblkid);

        *Vpos   = myblkid*Vblksiz*ldv + locj*ldv + locj;
        *TAUpos = myblkid*Vblksiz + locj;
    }

    void magma_bulge_findVTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t ldt,
                               magma_int_t *Vpos, magma_int_t *Tpos)
    {
        magma_int_t myblkid;
        magma_int_t locj = sweep%Vblksiz;

        if(applyQver==113)
            magma_bulge_findpos113(n, nb, Vblksiz, sweep, st, &myblkid);
        else
            magma_bulge_findpos(n, nb, Vblksiz, sweep, st, &myblkid);


        *Vpos   = myblkid*Vblksiz*ldv + locj*ldv + locj;
        *Tpos   = myblkid*Vblksiz*ldt + locj*ldt + locj;
    }

    void magma_bulge_findVTAUTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t ldt,
                               magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *blkid)
    {
        magma_int_t myblkid;
        magma_int_t locj = sweep%Vblksiz;

        if(applyQver==113)
            magma_bulge_findpos113(n, nb, Vblksiz, sweep, st, &myblkid);
        else
            magma_bulge_findpos(n, nb, Vblksiz, sweep, st, &myblkid);


        *Vpos   = myblkid*Vblksiz*ldv + locj*ldv + locj;
        *TAUpos = myblkid*Vblksiz     + locj;
        *Tpos   = myblkid*Vblksiz*ldt + locj*ldt + locj;
        *blkid  = myblkid;
    }

    magma_int_t magma_bulge_get_blkcnt(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz)
    {
        magma_int_t colblk, nbcolblk;
        magma_int_t myblknb, mastersweep;

        magma_int_t blkcnt = 0;
        nbcolblk = magma_ceildiv((n-1),Vblksiz);
        for (colblk = 0; colblk<nbcolblk; colblk++)
        {
            mastersweep = colblk * Vblksiz;
            if(colblk == (nbcolblk-1))
                myblknb = magma_ceildiv((n-(mastersweep+1)),nb);
            else
                myblknb = magma_ceildiv((n-(mastersweep+2)),nb);
            blkcnt      = blkcnt + myblknb;
            //printf("voici  nbcolblk %d    master sweep %d     blkcnt %d \n",nbcolblk, mastersweep,*blkcnt);
        }
        return blkcnt;
    }

    ///////////////////
    // Old functions //
    ///////////////////

    magma_int_t plasma_ceildiv(magma_int_t a, magma_int_t b)
    {
        return magma_ceildiv(a,b);
    }

    void findVTpos(magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *myblkid)
    {
        // to be able to use and compare with the old reduction function.
        // route the old function to the new ones because the changes are done on the new function.
        magma_int_t ldv = NB + Vblksiz;
        magma_int_t ldt = Vblksiz;
        magma_bulge_findVTAUTpos(N,  NB, Vblksiz,  sweep,  st,  ldv, ldt,
                               Vpos, TAUpos, Tpos, myblkid);
        return;


        magma_int_t locblknb, prevblkcnt, prevGblkid;
        magma_int_t myblknb, nbprevGblk, mastersweep;
        magma_int_t blkid, locj, LDV;
        locblknb = 0;
        prevblkcnt   = 0;
        myblknb  = 0;

        nbprevGblk = sweep/Vblksiz;
        for (prevGblkid = 0; prevGblkid < nbprevGblk; prevGblkid++)
        {
            mastersweep  = prevGblkid * Vblksiz;
            locblknb = plasma_ceildiv((N-(mastersweep+2)),NB);
            prevblkcnt   = prevblkcnt + locblknb;
        }
        myblknb = plasma_ceildiv((st-sweep),NB);
        blkid       = prevblkcnt + myblknb -1;
        locj        = sweep%Vblksiz;
        LDV         = NB + Vblksiz;

        *myblkid= blkid;
        *Vpos   = blkid*Vblksiz*LDV  + locj*LDV + locj;
        *TAUpos = blkid*Vblksiz + locj;
        *Tpos   = blkid*Vblksiz*Vblksiz + locj*Vblksiz + locj;
        //printf("voici  blkid  %d  locj %d  vpos %d tpos %d \n",blkid,locj,*Vpos,*Tpos);
    }



    void findVTsiz(magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, magma_int_t *blkcnt, magma_int_t *LDV)
    {
        magma_int_t colblk, nbcolblk;
        magma_int_t myblknb, mastersweep;

        *blkcnt   = 0;
        nbcolblk = plasma_ceildiv((N-1),Vblksiz);
        for (colblk = 0; colblk<nbcolblk; colblk++)
        {
            mastersweep = colblk * Vblksiz;
            if(colblk == (nbcolblk-1))
                myblknb = magma_ceildiv((N-(mastersweep+1)),NB);
            else
                myblknb = magma_ceildiv((N-(mastersweep+2)),NB);

            *blkcnt      = *blkcnt + myblknb;
            //printf("voici  nbcolblk %d    master sweep %d     blkcnt %d \n",nbcolblk, mastersweep,*blkcnt);
        }
        *LDV= NB+Vblksiz;
    }


}



