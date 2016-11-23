/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @precisions normal z -> s d c
*/

#ifndef MAGMA_ZBULGEINC_H
#define MAGMA_ZBULGEINC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


// =============================================================================
// Configuration

// maximum contexts
#define MAX_THREADS_BLG         256

void findVTpos(
    magma_int_t N, magma_int_t NB, magma_int_t Vblksiz,
    magma_int_t sweep, magma_int_t st,
    magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos,
    magma_int_t *myblkid);

void findVTsiz(
    magma_int_t N, magma_int_t NB, magma_int_t Vblksiz,
    magma_int_t *blkcnt, magma_int_t *LDV);

struct gbstrct_blg {
    magmaDoubleComplex *dQ1;
    magmaDoubleComplex *dT1;
    magmaDoubleComplex *dT2;
    magmaDoubleComplex *dV2;
    magmaDoubleComplex *dE;
    magmaDoubleComplex *T;
    magmaDoubleComplex *A;
    magmaDoubleComplex *V;
    magmaDoubleComplex *TAU;
    magmaDoubleComplex *E;
    magmaDoubleComplex *E_CPU;
    int cores_num;
    int locores_num;
    int overlapQ1;
    int usemulticpu;
    int NB;
    int NBTILES;
    int N;
    int NE;
    int N_CPU;
    int N_GPU;
    int LDA;
    int LDE;
    int BAND;
    int grsiz;
    int Vblksiz;
    int WANTZ;
    magma_side_t SIDE;
    real_Double_t *timeblg;
    real_Double_t *timeaplQ;
    volatile int *ss_prog;
};

// declare globals here; defined in zhetrd_bhe2trc.cpp
extern struct gbstrct_blg core_in_all;


#ifdef __cplusplus
}
#endif

#endif /* MAGMA_ZBULGEINC_H */
