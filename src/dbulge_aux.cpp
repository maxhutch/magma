/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 *     @author Azzam Haidar
 *     @author Stan Tomov
 *
 *     @precisions normal d -> s
 *
 */

#include "common_magma.h"
#include "magma_dbulgeinc.h"
 

//////////////////////////////////////////////////////////////
//          DSTEDC          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_dstedc_withZ(char JOBZ, magma_int_t N, double *D, double * E, double *Z, magma_int_t LDZ) {
  double *WORK;
  magma_int_t *IWORK;
  magma_int_t LWORK, LIWORK;
  magma_int_t INFO;
   
  if(JOBZ=='V'){
        LWORK  = 1 + 3*N + 3*N*((magma_int_t)log2(N)+1) + 4*N*N+ 256*N; 
        LIWORK =  6 + 6*N + 6*N*((magma_int_t)log2(N)+1) + 256*N;
  }else if(JOBZ=='I'){
        LWORK  = 2*N*N+256*N+1; 
          LIWORK = 256*N;
  }else if(JOBZ=='N'){
        LWORK  = 256*N+1; 
          LIWORK = 256*N;  
  }else{
          printf("ERROR JOBZ %c\n",JOBZ);
          exit(-1);
  }

  WORK = (double*) malloc( LWORK*sizeof( double) );
  IWORK = (magma_int_t*) malloc( LIWORK*sizeof( magma_int_t) );

  lapackf77_dstedc(&JOBZ, &N, D, E, Z, &LDZ, WORK,&LWORK,IWORK,&LIWORK,&INFO);

  if(INFO!=0){
        printf("=================================================\n");
        printf("DSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
          //assert(INFO==0);
  }


  magma_free_cpu( IWORK );
  magma_free_cpu( WORK );
}
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//          DSTEDX          Divide and Conquer for tridiag
//////////////////////////////////////////////////////////////
extern "C" void  magma_dstedx_withZ(magma_int_t N, magma_int_t NE, double *D, double * E, double *Z, magma_int_t LDZ) {
  double *WORK;
  double *dwork;
  magma_int_t *IWORK;
  magma_int_t LWORK, LIWORK;
  magma_int_t INFO;
   
  LWORK  = N*N+4*N+1; 
  LIWORK = 3 + 5*N;

  WORK = (double*) malloc( LWORK*sizeof( double) );
  IWORK = (magma_int_t*) malloc( LIWORK*sizeof( magma_int_t) );

  if (MAGMA_SUCCESS != magma_dmalloc( &dwork, 3*N*(N/2 + 1) )) {
     printf("=================================================\n");
     printf("DSTEDC ERROR OCCURED IN CUDAMALLOC\n");
     printf("=================================================\n");
     return;
  }
  printf("using magma_dstedx\n");

 
#ifdef ENABLE_TIMER 
    magma_timestr_t start, end;
    start = get_current_time();
#endif

  char job = 'I';

  if(NE==N)
    job = 'A';

  magma_dstedx('I', N, 0., 0., 1, NE, D, E, Z, LDZ, WORK,LWORK,IWORK,LIWORK,dwork,&INFO);

  if(INFO!=0){
        printf("=================================================\n");
        printf("DSTEDC ERROR OCCURED. HERE IS INFO %d \n ", (int) INFO);
        printf("=================================================\n");
          //assert(INFO==0);
  }

#ifdef ENABLE_TIMER    
    end = get_current_time();
    printf("time dstedx = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

  magma_free( dwork );
  magma_free_cpu( IWORK );
  magma_free_cpu( WORK );
}
//////////////////////////////////////////////////////////////
