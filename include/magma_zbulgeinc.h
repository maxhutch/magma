/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
*/

#ifndef MAGMA_ZBULGEINC_H
#define MAGMA_ZBULGEINC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


/***************************************************************************//**
 *  Configuration
 **/

 // maximum contexts
#define MAX_THREADS_BLG         256

void findVTpos(magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *myblkid);
void findVTsiz(magma_int_t N, magma_int_t NB, magma_int_t Vblksiz, magma_int_t *blkcnt, magma_int_t *LDV);
magma_int_t plasma_ceildiv(magma_int_t a, magma_int_t b);


/*
extern volatile magma_int_t barrier_in[MAX_THREADS_BLG];
extern volatile magma_int_t barrier_out[MAX_THREADS_BLG];
extern volatile magma_int_t *ss_prog;
*/

 /***************************************************************************//**
 *  Static scheduler
 **/
/*
#define ssched_init(nbtiles) \
{ \
        volatile int   prog_ol[2*nbtiles+10];\
                 int   iamdone[MAX_THREADS_BLG]; \
                 int   thread_num[MAX_THREADS_BLG];\
        pthread_t      thread_id[MAX_THREADS_BLG];\
        pthread_attr_t thread_attr;\
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////



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
} ;

// declare globals here; defined in zhetrd_bhe2trc.cpp
extern struct gbstrct_blg core_in_all;





////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_EVENTSBLG 163840
//#define MAX_EVENTSBLG 1048576

// If we're not using GNU C, elide __attribute__
#ifndef __GNUC__
#define __attribute__(x)  /*NOTHING*/
#endif

// declare globals here; defined in zhetrd_bhe2trc.cpp
extern int           event_numblg        [MAX_THREADS_BLG]                 __attribute__ ((aligned (128)));
extern real_Double_t event_start_timeblg [MAX_THREADS_BLG]                 __attribute__ ((aligned (128)));
extern real_Double_t event_end_timeblg   [MAX_THREADS_BLG]                 __attribute__ ((aligned (128)));
extern real_Double_t event_logblg        [MAX_THREADS_BLG][MAX_EVENTSBLG]  __attribute__ ((aligned (128)));
extern int           log_eventsblg;

#ifndef __GNUC__
#undef  __attribute__
#endif

#define core_event_startblg(my_core_id)\
    event_start_timeblg[my_core_id] = magma_wtime();

#define core_event_endblg(my_core_id)\
    event_end_timeblg[my_core_id] = magma_wtime();

#define core_log_eventblg(event, my_core_id)\
    event_logblg[my_core_id][event_numblg[my_core_id]+0] = my_core_id;\
    event_logblg[my_core_id][event_numblg[my_core_id]+1] = event_start_timeblg[my_core_id];\
    event_logblg[my_core_id][event_numblg[my_core_id]+2] = event_end_timeblg[my_core_id];\
    event_logblg[my_core_id][event_numblg[my_core_id]+3] = (event);\
    event_numblg[my_core_id] += (log_eventsblg << 2);\
    event_numblg[my_core_id] &= (MAX_EVENTSBLG-1);

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_ZBULGEINC_H */
