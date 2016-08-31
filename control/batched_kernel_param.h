/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016
*/

#ifndef BATCHED_KERNEL_PARAM_H
#define BATCHED_KERNEL_PARAM_H

#define MAX_NTHREADS        1024     // 1024 is max threads for 2.x cards
#define MAX_SHARED_ALLOWED    44

#define zamax            256
#define DOTC_MAX_BS      512     // 512 is max threads for 1.x cards


#define POTRF_NB         128     // blocking in main algorithm 128 if using recursive panel or 32 if using standard panel
#define POTF2_NB           8     // blocking size in panel factorization
#define POTF2_TILE_SIZE   32
#define MAX_POTF2_SM     128
#define VERSION20


#define BATRF_NB         128
#define BATRF_RECNB       32
#define BATF2_NB           8
#define BASWP_WIDTH        4
#define SWP_WIDTH          4

#define BAQRF_NB         32

#define BATRI_NB         128        // ztrsm_nb should be >= BATRF_NB
#define TRI_NB           128        // ztrsm_nb should match the NB in BATRF_NB
#define TRI_BLOCK_SIZE    16

// TRSM tuning parameters 
#define STRTRI_BATCHED_NB         (64)
#define STRTRI_BATCHED_BLOCK_SIZE (16)
#define DTRTRI_BATCHED_NB         (64)
#define DTRTRI_BATCHED_BLOCK_SIZE (16)
#define CTRTRI_BATCHED_NB         (32)
#define CTRTRI_BATCHED_BLOCK_SIZE (16)
#define ZTRTRI_BATCHED_NB         (128)
#define ZTRTRI_BATCHED_BLOCK_SIZE (16)

#endif        //  #ifndef BATCHED_KERNEL_PARAM_H
