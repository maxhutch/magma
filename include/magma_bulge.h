/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016
*/

#ifndef MAGMA_BULGE_H
#define MAGMA_BULGE_H

#include "magma_types.h"
#include "magma_zbulge.h"
#include "magma_cbulge.h"
#include "magma_dbulge.h"
#include "magma_sbulge.h"

#ifdef __cplusplus
extern "C" {
#endif
    
    magma_int_t magma_yield();
    magma_int_t magma_bulge_getlwstg1(magma_int_t n, magma_int_t nb, magma_int_t *lda2);

    void cmp_vals(magma_int_t n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);

    void magma_bulge_findVTAUpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv,
                                 magma_int_t *Vpos, magma_int_t *TAUpos);

    void magma_bulge_findVTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t ldt,
                               magma_int_t *Vpos, magma_int_t *Tpos);

    void magma_bulge_findVTAUTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t ldt,
                                  magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *blkid);

    void magma_bulge_findpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *myblkid);
    void magma_bulge_findpos113(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *myblkid);

    magma_int_t magma_bulge_get_blkcnt(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz);

    void findVTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *myblkid);

#ifdef __cplusplus
}
#endif

#endif  // MAGMA_BULGE_H
