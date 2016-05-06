/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==== Definition of blocking sizes for Nvidia cards
#ifdef HAVE_CUBLAS

#define ZPOTRF_SWITCH 160
#define CPOTRF_SWITCH 224
#define DPOTRF_SWITCH 384
#define SPOTRF_SWITCH 432

/* ////////////////////////////////////////////////////////////////////////////
   -- Return crossover for potrf based on m
*/
void magma_get_zpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= ZPOTRF_SWITCH)
    {
        *nb    = ZPOTRF_SWITCH;
        *recnb = ZPOTRF_SWITCH;
        return;
    }
    *nb    = 64;
    *recnb = 32;
    return;
}
void magma_get_cpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= CPOTRF_SWITCH)
    {
        *nb    = CPOTRF_SWITCH;
        *recnb = CPOTRF_SWITCH;
        return;
    }
    
    if (n <= 256)
    {
        *nb    = 256;
        *recnb = 256;
    }
    else {
        *nb    = 128;
        *recnb =  32;
    }
    return;
}

void magma_get_dpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= DPOTRF_SWITCH)
    {
        *nb    = DPOTRF_SWITCH;
        *recnb = DPOTRF_SWITCH;
        return;
    }
    if (n <= 384)
    {
        *nb    = 384;
        *recnb = 384;
    }
    else {
        *nb    = 128;
        *recnb =  32;
    }
    return;
}

void magma_get_spotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    if (n <= SPOTRF_SWITCH)
    {
        *nb    = SPOTRF_SWITCH;
        *recnb = SPOTRF_SWITCH;
        return;
    }
    if (n <= 464)
    {
        *nb    = 512;
        *recnb = 512;
    }
    else {
        *nb    = 256;
        *recnb =  64;
    }
    return;
}
/* ////////////////////////////////////////////////////////////////////////////
   -- Return crossover for potrf based on m
*/
void magma_get_zgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 64;
    *recnb = 32;
    return;
}
void magma_get_cgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 128;
    *recnb =  32;
    return;
}

void magma_get_dgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 128;
    *recnb =  32;
    return;
}

void magma_get_sgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb)
{
    *nb    = 128;
    *recnb =  32;
    return;
}




magma_int_t magma_get_zgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

magma_int_t magma_get_cgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

magma_int_t magma_get_dgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

magma_int_t magma_get_sgeqrf_batched_nb(magma_int_t m)
{
    return 32;
}

/* get the cross over switch between the _lg or the kernel directly*/
magma_int_t magma_get_zpotrf_batched_crossover()
{
    return ZPOTRF_SWITCH;
}
magma_int_t magma_get_cpotrf_batched_crossover()
{
    return CPOTRF_SWITCH;
}
magma_int_t magma_get_dpotrf_batched_crossover()
{
    return DPOTRF_SWITCH;
}
magma_int_t magma_get_spotrf_batched_crossover()
{
    return SPOTRF_SWITCH;
}

#endif  // HAVE_CUBLAS

#ifdef __cplusplus
} // extern "C"
#endif
