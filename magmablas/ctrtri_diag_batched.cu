/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/ztrtri_diag_batched.cu normal z -> c, Mon May  2 23:30:42 2016

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       File named ctrtri_diag.cu to avoid name conflict with src/ctrtri.o
       in the library. The actual kernels are in ctrtri_lower.cu and ctrtri_upper.cu
*/

#include "magma_internal.h"
#include "ctrtri.cuh"


/**
    Purpose
    -------
    CTRTRI_DIAG inverts the NB x NB diagonal blocks of a triangular matrix.
    This routine is used in ctrsm.
    
    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    n       INTEGER.
            On entry, n specifies the order of the matrix A. N >= 0.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount). 
             Each is a COMPLEX array A of dimension ( ldda, n )
             The triangular matrix A.
    \n
             If UPLO = MagmaUpper, the leading N-by-N upper triangular part of A
             contains the upper triangular matrix, and the strictly lower
             triangular part of A is not referenced.
    \n
            If UPLO = MagmaLower, the leading N-by-N lower triangular part of A
            contains the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.
    \n
            If DIAG = MagmaUnit, the diagonal elements of A are also not referenced
            and are assumed to be 1.

    @param[in]
    ldda    INTEGER.
            The leading dimension of each array A.  LDDA >= max(1,N).

    @param[out]
    dinvA_array Array of pointers, dimension (batchCount). 
                Each is a COMPLEX array dinvA of dimension (NB, ceil(n/NB)*NB),
                where NB = 128.
                On exit, contains inverses of the NB-by-NB diagonal blocks of A.

    @param[in]
    resetozero INTEGER
               If not zero, each array dinvA will be reset to all zeros 
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magmablas_ctrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex const * const *dA_array, magma_int_t ldda,
    magmaFloatComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if (uplo != MagmaLower && uplo != MagmaUpper)
        info = -1;
    else if (diag != MagmaNonUnit && diag != MagmaUnit)
        info = -2;
    else if (n < 0)
        info = -3;
    else if (ldda < n)
        info = -5;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info
    }
    
    int nblocks = magma_ceildiv( n, IB );

    if ( resetozero ) {
        magmablas_claset_batched(MagmaFull, magma_roundup( n, NB ), NB, MAGMA_C_ZERO, MAGMA_C_ZERO, dinvA_array, magma_roundup( n, NB ), batchCount, queue);
    }
    // if someone want to use cudamemset he need to set the whole vectors 
    // of initial size otherwise it is a bug and thus need to have dinvA_length 
    // in input parameter and has been tested and was slower.
    //was not the largest size computed by the high API getrf_batched then it is bug and need to use magmablas_claset_batched


    if ( uplo == MagmaLower ) {
        // invert diagonal IB x IB inner blocks
        dim3 diaggrid( nblocks, 1, batchCount );  // emulate 3D grid
        ctrtri_diag_lower_kernel_batched
            <<< diaggrid, IB, 0, queue->cuda_stream() >>>
            ( diag, n, dA_array, ldda, dinvA_array );

        // build up NB x NB blocks (assuming IB=16 here):
        // use   16 x 16  blocks to build  32 x 32  blocks,  1 x (1 x npages) grid,  4 x 4 threads;
        // then  32 x 32  blocks to build  64 x 64  blocks,  1 x (2 x npages) grid,  8 x 4 threads;
        // then  64 x 64  blocks to build 128 x 128 blocks,  1 x (4 x npages) grid, 16 x 4 threads;
        // then 128 x 128 blocks to build 256 x 256 blocks,  2 x (8 x npages) grid, 16 x 4 threads.
        for( int jb=IB; jb < NB; jb *= 2 ) {
            int kb = jb*2;
            int npages = magma_ceildiv( n, kb );
            dim3 threads( (jb <= 32 ? jb/4 : 16), 4 );
            dim3 grid( jb/(threads.x*threads.y), npages*(jb/16), batchCount );  // emulate 3D grid: NX * (NY*npages), for CUDA ARCH 1.x
            
            //printf( "n %d, jb %d, grid %d x %d (%d x %d)\n", n, jb, grid.x, grid.y, grid.y / npages, npages );
            switch (jb) {
                case 16:
                    triple_cgemm16_part1_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm16_part2_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 32:
                    triple_cgemm32_part1_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm32_part2_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 64:
                    triple_cgemm64_part1_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm64_part2_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                default:
                    triple_cgemm_above64_part1_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm_above64_part2_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm_above64_part3_lower_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
            }
            if ( kb >= n ) break;
        }
    }
    else {
        dim3 diaggrid( nblocks, 1, batchCount );  // emulate 3D grid
        ctrtri_diag_upper_kernel_batched
            <<< diaggrid, IB, 0, queue->cuda_stream() >>>
            ( diag, n, dA_array, ldda, dinvA_array );

        // update the inverse up to the size of IB
        for( int jb=IB; jb < NB; jb *= 2 ) {
            int kb = jb*2;
            int npages = magma_ceildiv( n, kb );
            dim3 threads( (jb <= 32 ? jb/4 : 16), 4 );
            dim3 grid( jb/(threads.x*threads.y), npages*(jb/16), batchCount );  // emulate 3D grid: NX * (NY*npages), for CUDA ARCH 1.x
            
            switch (jb) {
                case 16:
                    triple_cgemm16_part1_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm16_part2_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 32:
                    triple_cgemm32_part1_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm32_part2_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 64:
                    triple_cgemm64_part1_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm64_part2_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                default:
                    triple_cgemm_above64_part1_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm_above64_part2_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_cgemm_above64_part3_upper_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
            }
            if ( kb >= n ) break;
        }
    }
}
