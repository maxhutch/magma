/* 
    -- MAGMA (version 1.4.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 

 
*/ 
#ifndef MAGMA_ASYNC_H
#define MAGMA_ASYNC_H

#if (dbglevel >=1)
#include "ca_dbg_tools.h" /*Enable tracing tools and debugging tools*/
#endif

/* Context and arguments */
#include "magma_async_args.h"

/* initialisation and controls */
#include "magma_async_controls.h"

/* Scheduling takes place here */
#include "schedule.h"

/* Fill block of memory in async_dmemset.cpp*/
void magma_dmemset_async(double *ptr, double value, int n, magma_int_t chunck, int P);

/*LU factorization in async_dgetrf_rec_gpu.cpp*/

extern "C" magma_int_t magma_dgetrf_async_gpu(
magma_int_t m, magma_int_t n, 
double *dA, magma_int_t dA_LD,
magma_int_t *ipiv, magma_int_t *info
);

extern "C" magma_int_t magma_dgetrf_async_work_gpu(
magma_int_t m, magma_int_t n,  
double *dA, magma_int_t dA_LD, 
magma_int_t *ipiv, magma_int_t *info,
/*Workspace on the cpu side*/
double *AWORK, magma_int_t AWORK_LD, magma_int_t AWORK_n
);

#endif
