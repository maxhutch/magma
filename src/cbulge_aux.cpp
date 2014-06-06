/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @generated c Tue Dec 17 13:18:36 2013
 *
 */

#include "common_magma.h"


//////////////////////////////////////////////////////////////
//          CSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_cstedc_withZ(char JOBZ, magma_int_t N, float *D, float * E, magmaFloatComplex *Z, magma_int_t LDZ)
{
    magmaFloatComplex *WORK;
    float *RWORK;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK, LRWORK;
    magma_int_t INFO;
    
    // use log() as log2() is not defined everywhere (e.g., Windows)
    const float log_2 = 0.6931471805599453;
    if (JOBZ=='V') {
        LWORK  = N*N;
        LRWORK = 1 + 3*N + 3*N*((magma_int_t)(log( (float)N )/log_2) + 1) + 4*N*N + 256*N;
        LIWORK = 6 + 6*N + 6*N*((magma_int_t)(log( (float)N )/log_2) + 1) + 256*N;
    } else if (JOBZ=='I') {
        LWORK  = N;
        LRWORK = 2*N*N + 4*N + 1 + 256*N;
        LIWORK = 256*N;
    } else if (JOBZ=='N') {
        LWORK  = N;
        LRWORK = 256*N + 1;
        LIWORK = 256*N;
    } else {
        printf("ERROR JOBZ %c\n",JOBZ);
        exit(-1);
    }
    
    RWORK  = (float*) malloc( LRWORK*sizeof( float) );
    WORK   = (magmaFloatComplex*) malloc( LWORK*sizeof( magmaFloatComplex) );
    IWORK  = (magma_int_t*) malloc( LIWORK*sizeof( magma_int_t) );
    
    lapackf77_cstedc(&JOBZ, &N, D, E, Z, &LDZ, WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);
    
    if (INFO!=0) {
        printf("=================================================\n");
        printf("CSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
        //assert(INFO==0);
    }
    
    magma_free_cpu( IWORK );
    magma_free_cpu( WORK );
    magma_free_cpu( RWORK );
}
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//          CSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_cstedx_withZ(magma_int_t N, magma_int_t NE, float *D, float * E, magmaFloatComplex *Z, magma_int_t LDZ)
{
    float *RWORK;
    float *dwork;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK, LRWORK;
    magma_int_t INFO;
    
    LWORK  = N;
    LRWORK = 2*N*N + 4*N + 1 + 256*N;
    LIWORK = 256*N;
    
    RWORK = (float*) malloc( LRWORK*sizeof(float) );
    IWORK = (magma_int_t*) malloc( LIWORK*sizeof(magma_int_t) );
    
    if (MAGMA_SUCCESS != magma_smalloc( &dwork, 3*N*(N/2 + 1) )) {
        printf("=================================================\n");
        printf("CSTEDC ERROR OCCURED IN CUDAMALLOC\n");
        printf("=================================================\n");
        return;
    }
    printf("using magma_cstedx\n");

#ifdef ENABLE_TIMER
    magma_timestr_t start, end;
    start = get_current_time();
#endif

    char job = 'I';

    if (NE==N)
        job = 'A';

    magma_cstedx(job, N, 0.,0., 1, NE, D, E, Z, LDZ, RWORK, LRWORK, IWORK, LIWORK, dwork, &INFO);

    if (INFO!=0) {
        printf("=================================================\n");
        printf("CSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
        //assert(INFO==0);
    }

#ifdef ENABLE_TIMER
    end = get_current_time();
    printf("time zstevx = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    magma_free( dwork );
    magma_free_cpu( IWORK );
    magma_free_cpu( RWORK );
}
