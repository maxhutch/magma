/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions normal z

*/

#include "common_magma.h"
#include "commonblas_d.h"
#include <assert.h>

#define magmablas_zgemm_fermi magmablas_zgemm

#include "gemm_stencil_defs.cu"

///////////////////////////////////////////////////////////////////////////////////////////////////

#define trans_nn 1
#define        trans_nt 2
#define        trans_nc 3

#define        trans_tn 4
#define        trans_tt 5
#define        trans_tc 6

#define        trans_cn 7
#define        trans_ct 8
#define        trans_cc 9

///////////////////////////////////////////////////////////////////////////////////////////////////

  #ifdef TEXTURE_1D

    #ifdef COMPLEX
      #ifdef DOUBLE
        texture<int4, 1, cudaReadModeElementType> tex_ref_A;
        texture<int4, 1, cudaReadModeElementType> tex_ref_B;
      #else
        texture<float2, 1, cudaReadModeElementType> tex_ref_A;
        texture<float2, 1, cudaReadModeElementType> tex_ref_B;
      #endif
    #else
      #ifdef DOUBLE
        texture<int2, 1, cudaReadModeElementType> tex_ref_A;
        texture<int2, 1, cudaReadModeElementType> tex_ref_B;
      #else
        texture<float, 1, cudaReadModeElementType> tex_ref_A;
        texture<float, 1, cudaReadModeElementType> tex_ref_B;
      #endif
    #endif

  #endif

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// size of work for a thread block
#define BLK_M 24
#define BLK_N 16 
#define BLK_K  8

// size of thread block for calculating C (innermost loop)
#define DIM_X  8
#define DIM_Y  8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 8
#define DIM_YA 8
  
// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#define version trans_nn
#include "gemm_stencil.cu"
 
#undef BLK_M
#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB

///////////////////////////////////////////////////////////////////////////////////////////////////
 
///////////////////////////////////////////////////////////////////////////////////////////////////
// size of work for a thread block
#define BLK_M 16
#define BLK_N 24
#define BLK_K 8
  
// size of thread block for calculating C (innermost loop)
//#define DIM_X 8
//#define DIM_Y 8
   
// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 8
#define DIM_YA 8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#define version trans_nt
#include "gemm_stencil.cu"

#define        version        trans_nc
#include "gemm_stencil.cu"

#undef BLK_M
#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// size of work for a thread block
#define BLK_M 16
#define BLK_N 24
#define BLK_K 8

// size of thread block for calculating C (innermost loop)
//#define DIM_X 8
//#define DIM_Y 8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 4
#define DIM_YA 16

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#define version trans_tt
#include "gemm_stencil.cu"

#define version trans_tc
#include "gemm_stencil.cu"

#define version trans_ct
#include "gemm_stencil.cu"

#define        version        trans_cc
#include "gemm_stencil.cu"

#undef BLK_M
#undef BLK_N
#undef BLK_K

#undef DIM_XA
#undef DIM_YA

#undef DIM_XB
#undef DIM_YB
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
// size of work for a thread block
#define BLK_M 24
#define BLK_N 16
#define BLK_K  8

// size of thread block for calculating C (innermost loop)
//#define DIM_X 8
//#define DIM_Y 8

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA 8
#define DIM_YA 8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB 8
#define DIM_YB 8

#define version trans_tn
#include "gemm_stencil.cu"

#define        version        trans_cn
#include "gemm_stencil.cu"

///////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" void
magmablas_zgemm_fermi( char TRANSA, char TRANSB, magma_int_t m, magma_int_t n, magma_int_t k,
                       magmaDoubleComplex alpha, const magmaDoubleComplex *d_A, magma_int_t lda,
                                              const magmaDoubleComplex *d_B, magma_int_t ldb,
                       magmaDoubleComplex beta,        magmaDoubleComplex *d_C, magma_int_t ldc )
{
/*  -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

   Purpose
   =======
   ZGEMM  performs one of the matrix-matrix operations

      C := alpha*op( A )*op( B ) + beta*C,

   where  op( X ) is one of

      op( X ) = X   or   op( X ) = X',

   alpha and beta are scalars, and A, B and C are matrices, with op( A )
   an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

   Parameters
   ==========
   TRANSA - CHARACTER*1.
            On entry, TRANSA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
               TRANSA = 'N' or 'n',  op( A ) = A.
               TRANSA = 'T' or 't',  op( A ) = A'.
               TRANSA = 'C' or 'c',  op( A ) = A'.
            Unchanged on exit.

   TRANSB - CHARACTER*1.
            On entry, TRANSB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
               TRANSB = 'N' or 'n',  op( B ) = B.
               TRANSB = 'T' or 't',  op( B ) = B'.
               TRANSB = 'C' or 'c',  op( B ) = B'.
            Unchanged on exit.

   M      - INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( d_A )  and of the  matrix d_C.  M  must  be at least  zero.
            Unchanged on exit.

   N      - INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( d_B ) and the number of columns of the matrix d_C. N must be
            at least zero.
            Unchanged on exit.

   K      - INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( d_A ) and the number of rows of the matrix op( d_B ). K must
            be at least  zero.
            Unchanged on exit.

   ALPHA  - COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.
            Unchanged on exit.

   d_A    - COMPLEX_16 array of DIMENSION ( LDA, ka ), where ka is
            k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
            Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
            part of the array d_A must contain the matrix d_A, otherwise
            the leading  k by m  part of the array d_A must contain  the
            matrix d_A.
            Unchanged on exit.

   LDA    - INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. When  TRANSA = 'N' or 'n' then
            LDA must be at least  max( 1, m ), otherwise  LDA must be at
            least  max( 1, k ).
            Unchanged on exit.

   d_B    - COMPLEX_16 array of DIMENSION ( LDB, kb ), where kb is
            n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
            Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
            part of the array d_B must contain the matrix d_B, otherwise
            the leading  n by k  part of the array d_B must contain  the
            matrix d_B.
            Unchanged on exit.
 
   LDB    - INTEGER.
            On entry, LDB specifies the first dimension of d_B as declared
            in the calling (sub) program. When  TRANSB = 'N' or 'n' then
            LDB must be at least  max( 1, k ), otherwise  LDB must be at
            least  max( 1, n ).
            Unchanged on exit.

   BETA   - COMPLEX_16.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then d_C need not be set on input.
            Unchanged on exit.

   d_C    - COMPLEX_16 array of DIMENSION ( LDC, n ).
            Before entry, the leading  m by n  part of the array  d_C must
            contain the matrix  d_C,  except when  beta  is zero, in which
            case d_C need not be set on entry.
            On exit, the array  d_C  is overwritten by the  m by n  matrix
            ( alpha*op( d_A )*op( d_B ) + beta*d_C ).

   LDC    - INTEGER.
            On entry, LDC specifies the first dimension of d_C as declared
            in  the  calling  (sub)  program.   LDC  must  be  at  least
            max( 1, m ).
            Unchanged on exit.
   =====================================================================    */

    if (m<=0 || n<=0 || k<=0)
       return;

    size_t offsetA = 0;
    size_t offsetB = 0;

    int TransA = 2, TransB = 2;
    if (TRANSA == 'T' ||  TRANSA == 't')
       TransA = 1;
    else
       if (TRANSA == 'N' ||  TRANSA == 'n')
          TransA = 0;
                    
    if (TRANSB == 'T' ||  TRANSB == 't')
       TransB = 1;
    else
       if (TRANSB == 'N' ||  TRANSB == 'n')
          TransB = 0;

    size_t sizeA = (size_t) lda * (size_t) (!TransA ? k : m);
    size_t sizeB = (size_t) ldb * (size_t) (!TransB ? n : k);

    size_t CUBLAS_MAX_1DBUF_SIZE = ((1 << 27) - 512);
    if (sizeA>=CUBLAS_MAX_1DBUF_SIZE ||
        sizeB>=CUBLAS_MAX_1DBUF_SIZE )
    {
        cublasZgemm(TRANSA, TRANSB, m, n, k, alpha,
                     d_A, lda, d_B, ldb,
                     beta, d_C, ldc);
        return;
    }

    #ifdef TEXTURE_1D
        // Set textures parameters
        tex_ref_A.normalized = false;
        tex_ref_A.filterMode = cudaFilterModePoint;
        tex_ref_A.addressMode[0] = cudaAddressModeClamp;

        tex_ref_B.normalized = false;
        tex_ref_B.filterMode = cudaFilterModePoint;
        tex_ref_B.addressMode[0] = cudaAddressModeClamp;

        // Bind A and B to texture references
        assert(cudaBindTexture(&offsetA, tex_ref_A, d_A, sizeA*sizeof(magmaDoubleComplex)) 
                   == cudaSuccess);
        assert(cudaBindTexture(&offsetB, tex_ref_B, d_B, sizeB*sizeof(magmaDoubleComplex))
                   == cudaSuccess);
    #endif

    // Set up grids
    dim3 dimBlock(DIM_X, DIM_Y);

    offsetA = offsetA/sizeof(d_A[0]);
    offsetB = offsetB/sizeof(d_B[0]);
 
    if (TransA==0 && TransB ==0){
       //dim3 dimGrid(m/BLK_M_nn + (m%BLK_M_nn != 0),
       //             n/BLK_N_nn + (n%BLK_N_nn != 0));
       dim3 dimGrid(m/24 + (m%24 != 0),  n/16 + (n%16 != 0));
       fermi_gemm_kernel_nn<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else 
    if (TransA==0 && TransB ==1){
       //dim3 dimGrid(m/BLK_M_nt + (m%BLK_M_nt != 0),
       //             n/BLK_N_nt + (n%BLK_N_nt != 0));
       dim3 dimGrid(m/16 + (m%16 != 0),         n/24 + (n%24 != 0));
       fermi_gemm_kernel_nt<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else
    if (TransA==0 && TransB ==2){
       //dim3 dimGrid(m/BLK_M_nc + (m%BLK_M_nc != 0),
       //             n/BLK_N_nc + (n%BLK_N_nc != 0));
       dim3 dimGrid(m/16 + (m%16 != 0),  n/24 + (n%24 != 0));
       fermi_gemm_kernel_nc<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else
    
    if (TransA==1 && TransB ==0){
       //dim3 dimGrid(m/BLK_M_tn + (m%BLK_M_tn != 0),
       //             n/BLK_N_tn + (n%BLK_N_tn != 0));
       dim3 dimGrid(m/24 + (m%24 != 0),         n/16 + (n%16 != 0));
       fermi_gemm_kernel_tn<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else
    if (TransA==1 && TransB ==1){
       //dim3 dimGrid(m/BLK_M_tt + (m%BLK_M_tt != 0),
       //             n/BLK_N_tt + (n%BLK_N_tt != 0));
       dim3 dimGrid(m/16 + (m%16 != 0), n/24 + (n%24 != 0));
       fermi_gemm_kernel_tt<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else
    if (TransA==1 && TransB ==2){
       //dim3 dimGrid(m/BLK_M_tc + (m%BLK_M_tc != 0),
       //             n/BLK_N_tc + (n%BLK_N_tc != 0));
       dim3 dimGrid(m/16 + (m%16 != 0), n/24 + (n%24 != 0));
       fermi_gemm_kernel_tc<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else

    if (TransA==2 && TransB ==0){
       //dim3 dimGrid(m/BLK_M_cn + (m%BLK_M_cn != 0),
       //             n/BLK_N_cn + (n%BLK_N_cn != 0));
       dim3 dimGrid(m/24 + (m%24 != 0),  n/16 + (n%16 != 0));
       fermi_gemm_kernel_cn<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else
    if (TransA==2 && TransB ==1){
       //dim3 dimGrid(m/BLK_M_ct + (m%BLK_M_ct != 0),
       //             n/BLK_N_ct + (n%BLK_N_ct != 0));
       dim3 dimGrid(m/16 + (m%16 != 0), n/24 + (n%24 != 0));
       fermi_gemm_kernel_ct<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    } else
    if (TransA==2 && TransB ==2){
       //dim3 dimGrid(m/BLK_M_cc + (m%BLK_M_cc != 0),
       //             n/BLK_N_cc + (n%BLK_N_cc != 0));
       dim3 dimGrid(m/16 + (m%16 != 0), n/24 + (n%24 != 0));
       fermi_gemm_kernel_cc<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, k, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta,
                                                   (int)offsetA, (int)offsetB);
    }

    cudaUnbindTexture ( tex_ref_A ) ;
    cudaUnbindTexture ( tex_ref_B ) ;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
