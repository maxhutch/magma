/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Azzam Haidar
*/

#ifndef MAGMA_THREADSETTING_H
#define MAGMA_THREADSETTING_H

#ifdef __cplusplus
extern "C" {
#endif
/***************************************************************************//**
 *  Internal routines
 **/
void magma_setlapack_multithreads(magma_int_t numthreads);
void magma_setlapack_sequential();
void magma_setlapack_numthreads(magma_int_t numthreads);
magma_int_t magma_getlapack_numthreads();
magma_int_t magma_get_numthreads();
/***************************************************************************/
#ifdef __cplusplus
}
#endif

#endif  // MAGMA_THREADSETTING_H
