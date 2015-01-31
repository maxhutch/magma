/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @generated from zbulge_aux.cpp normal z -> c, Fri Jan 30 19:00:18 2015
 *
 */

#include "common_magma.h"
#include "magma_timer.h"


//////////////////////////////////////////////////////////////
//          CSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_cstedc_withZ(magma_vec_t JOBZ, magma_int_t N, float *D, float * E, magmaFloatComplex *Z, magma_int_t LDZ)
{
    magmaFloatComplex *WORK;
    float *RWORK;
    magma_int_t *IWORK;
    magma_int_t LWORK, LIWORK, LRWORK;
    magma_int_t INFO;
    
    // use log() as log2() is not defined everywhere (e.g., Windows)
    const float log_2 = 0.6931471805599453;
    if (JOBZ == MagmaVec) {
        LWORK  = N*N;
        LRWORK = 1 + 3*N + 3*N*((magma_int_t)(log( (float)N )/log_2) + 1) + 4*N*N + 256*N;
        LIWORK = 6 + 6*N + 6*N*((magma_int_t)(log( (float)N )/log_2) + 1) + 256*N;
    } else if (JOBZ == MagmaIVec) {
        LWORK  = N;
        LRWORK = 2*N*N + 4*N + 1 + 256*N;
        LIWORK = 256*N;
    } else if (JOBZ == MagmaNoVec) {
        LWORK  = N;
        LRWORK = 256*N + 1;
        LIWORK = 256*N;
    } else {
        printf("ERROR JOBZ %c\n", JOBZ);
        exit(-1);
    }
    
    magma_smalloc_cpu( &RWORK, LRWORK );
    magma_cmalloc_cpu( &WORK,  LWORK  );
    magma_imalloc_cpu( &IWORK, LIWORK );
    
    lapackf77_cstedc( lapack_vec_const(JOBZ), &N, D, E, Z, &LDZ, WORK, &LWORK, RWORK, &LRWORK, IWORK, &LIWORK, &INFO);
    
    if (INFO != 0) {
        printf("=================================================\n");
        printf("CSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
        //assert(INFO == 0);
    }
    
    magma_free_cpu( IWORK );
    magma_free_cpu( WORK  );
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
    magma_int_t LIWORK, LRWORK;
    magma_int_t INFO;
    
    //LWORK  = N;
    LRWORK = 2*N*N + 4*N + 1 + 256*N;
    LIWORK = 256*N;
    
    magma_smalloc_cpu( &RWORK, LRWORK );
    magma_imalloc_cpu( &IWORK, LIWORK );
    
    if (MAGMA_SUCCESS != magma_smalloc( &dwork, 3*N*(N/2 + 1) )) {
        printf("=================================================\n");
        printf("CSTEDC ERROR OCCURED IN CUDAMALLOC\n");
        printf("=================================================\n");
        return;
    }
    printf("using magma_cstedx\n");

    magma_timer_t time=0;
    timer_start( time );

    magma_range_t job = MagmaRangeI;
    if (NE == N)
        job = MagmaRangeAll;

    magma_cstedx(job, N, 0., 0., 1, NE, D, E, Z, LDZ, RWORK, LRWORK, IWORK, LIWORK, dwork, &INFO);

    if (INFO != 0) {
        printf("=================================================\n");
        printf("CSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
        //assert(INFO == 0);
    }

    timer_stop( time );
    timer_printf( "time zstevx = %6.2f\n", time );

    magma_free( dwork );
    magma_free_cpu( IWORK );
    magma_free_cpu( RWORK );
}
