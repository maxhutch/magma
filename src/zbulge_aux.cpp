/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @precisions normal z -> c
 *
 */

#include "common_magma.h"


//////////////////////////////////////////////////////////////
//          ZSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_zstedc_withZ(char JOBZ, magma_int_t N, double *D, double * E, magmaDoubleComplex *Z, magma_int_t LDZ)
{
    magmaDoubleComplex *WORK;
    double *RWORK;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK, LRWORK;
    magma_int_t INFO;
    
    // use log() as log2() is not defined everywhere (e.g., Windows)
    const double log_2 = 0.6931471805599453;
    if (JOBZ=='V') {
        LWORK  = N*N;
        LRWORK = 1 + 3*N + 3*N*((magma_int_t)(log( (double)N )/log_2) + 1) + 4*N*N + 256*N;
        LIWORK = 6 + 6*N + 6*N*((magma_int_t)(log( (double)N )/log_2) + 1) + 256*N;
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
    
    RWORK  = (double*) malloc( LRWORK*sizeof( double) );
    WORK   = (magmaDoubleComplex*) malloc( LWORK*sizeof( magmaDoubleComplex) );
    IWORK  = (magma_int_t*) malloc( LIWORK*sizeof( magma_int_t) );
    
    lapackf77_zstedc(&JOBZ, &N, D, E, Z, &LDZ, WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);
    
    if (INFO!=0) {
        printf("=================================================\n");
        printf("ZSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
        //assert(INFO==0);
    }
    
    magma_free_cpu( IWORK );
    magma_free_cpu( WORK );
    magma_free_cpu( RWORK );
}
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//          ZSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_zstedx_withZ(magma_int_t N, magma_int_t NE, double *D, double * E, magmaDoubleComplex *Z, magma_int_t LDZ)
{
    double *RWORK;
    double *dwork;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK, LRWORK;
    magma_int_t INFO;
    
    LWORK  = N;
    LRWORK = 2*N*N + 4*N + 1 + 256*N;
    LIWORK = 256*N;
    
    RWORK = (double*) malloc( LRWORK*sizeof(double) );
    IWORK = (magma_int_t*) malloc( LIWORK*sizeof(magma_int_t) );
    
    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, 3*N*(N/2 + 1) )) {
        printf("=================================================\n");
        printf("ZSTEDC ERROR OCCURED IN CUDAMALLOC\n");
        printf("=================================================\n");
        return;
    }
    printf("using magma_zstedx\n");

#ifdef ENABLE_TIMER
    magma_timestr_t start, end;
    start = get_current_time();
#endif

    char job = 'I';

    if (NE==N)
        job = 'A';

    magma_zstedx(job, N, 0.,0., 1, NE, D, E, Z, LDZ, RWORK, LRWORK, IWORK, LIWORK, dwork, &INFO);

    if (INFO!=0) {
        printf("=================================================\n");
        printf("ZSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
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
