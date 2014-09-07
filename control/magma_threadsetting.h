/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

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
void magma_set_lapack_numthreads(magma_int_t numthreads);
magma_int_t magma_get_lapack_numthreads();
magma_int_t magma_get_parallel_numthreads();
/***************************************************************************/
#ifdef __cplusplus
}
#endif

#endif  // MAGMA_THREADSETTING_H
