/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       [zcds]gemm_fermi.cu        defines the CPU driver.
       [zcds]gemm_fermi_kernels.h defines the block sizes for each precision.
       gemm_stencil_defs.h        defines types and functions for precision-independent code.
       gemm_stencil.cu            defines the GPU kernel. It gets included
                                  multiple times, once for each transpose version.
*/
#include "common_magma.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

//#define COMPLEX
#define DOUBLE
#define TEXTURE_1D

#include "gemm_stencil_defs.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// A x B
// size of work for a thread block
#define BLK_M_nn  64
#define BLK_N_nn  64

#define BLK_K  16

// size of thread block for calculating C (innermost loop)
#define DIM_X  16
#define DIM_Y  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  16
#define DIM_YB  16

#define version trans_nn
#include "gemm_stencil.cu"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


///////////////////////////////////////////////////////////////////////////////////////////////////
// A x B^T
// size of work for a thread block
#define BLK_M_nt  64
#define BLK_N_nt  64

#define BLK_M_nc  64
#define BLK_N_nc  64

#define BLK_K  16

// size of thread block for calculating C (innermost loop)
//#define DIM_X  16
//#define DIM_Y  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  16
#define DIM_YB  16

#define version trans_nt
#include "gemm_stencil.cu"

#define version trans_nc
#include "gemm_stencil.cu"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


///////////////////////////////////////////////////////////////////////////////////////////////////
// A^T x B^T
// size of work for a thread block
#define BLK_M_tt  64
#define BLK_N_tt  64

#define BLK_M_tc  64
#define BLK_N_tc  64

#define BLK_M_ct  64
#define BLK_N_ct  64

#define BLK_M_cc  64
#define BLK_N_cc  64

#define BLK_K  16

// size of thread block for calculating C (innermost loop)
//#define DIM_X  16
//#define DIM_Y  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  16
#define DIM_YB  16

#define version trans_tt
#include "gemm_stencil.cu"

#define version trans_tc
#include "gemm_stencil.cu"

#define version trans_ct
#include "gemm_stencil.cu"

#define version trans_cc
#include "gemm_stencil.cu"

//#undef BLK_M
//#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB


///////////////////////////////////////////////////////////////////////////////////////////////////
// A^T x B
// size of work for a thread block
#define BLK_M_tn  64
#define BLK_N_tn  64

#define BLK_M_cn  64
#define BLK_N_cn  64

#define BLK_K  16

// size of thread block for calculating C (innermost loop)
//#define DIM_X  16
//#define DIM_Y  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  16
#define DIM_YA  16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  16
#define DIM_YB  16

#define version trans_tn
#include "gemm_stencil.cu"

#define version trans_cn
#include "gemm_stencil.cu"
