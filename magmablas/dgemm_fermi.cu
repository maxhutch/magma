/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgemm_fermi.cu normal z -> d, Fri Jan 30 19:00:10 2015

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates

       [zcds]gemm_fermi.cu          defines the CPU driver.
       [zcds]gemm_fermi_kernels.h   defines the block sizes for each precision.
       gemm_stencil_defs.h          defines types and functions for precision-independent code.
       
       These files are included multiple times, once for each transpose version.
       gemm_stencil.cuh             defines the GPU kernel (device function).
       gemm_kernel.cuh              defines the GPU kernel (global function).
       
       The batched version uses gemm_kernel_batched.cuh instead of gemm_kernel.cuh.
*/
#include "common_magma.h"
#include "commonblas_d.h"

#define PRECISION_d

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "dgemm_fermi_kernels.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/**
    Purpose
    -------
    DGEMM performs one of the matrix-matrix operations
    
        C = alpha*op( A )*op( B ) + beta*C,
    
    where op( X ) is one of
    
        op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
    
    alpha and beta are scalars, and A, B and C are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    @param[in]
    transA  CHARACTER*1.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = 'N':  op( A ) = A.
      -     = 'T':  op( A ) = A**T.
      -     = 'C':  op( A ) = A**H.
    
    @param[in]
    transB  CHARACTER*1.
            On entry, transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = 'N':  op( B ) = B.
      -     = 'T':  op( B ) = B**T.
      -     = 'C':  op( B ) = B**H.
    
    @param[in]
    m       INTEGER.
            On entry,  M  specifies  the number  of rows  of the  matrix
            op( dA )  and of the  matrix dC.  M  must  be at least  zero.
    
    @param[in]
    n       INTEGER.
            On entry,  N  specifies the number  of columns of the matrix
            op( dB ) and the number of columns of the matrix dC. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry,  K  specifies  the number of columns of the matrix
            op( dA ) and the number of rows of the matrix op( dB ). K must
            be at least  zero.
    
    @param[in]
    alpha   DOUBLE_PRECISION
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA      DOUBLE_PRECISION array of DIMENSION ( LDA, ka ), where ka is
            k  when  transA = MagmaNoTrans,  and is  m  otherwise.
            Before entry with  transA = MagmaNoTrans,  the leading  m by k
            part of the array dA must contain the matrix dA, otherwise
            the leading  k by m  part of the array dA must contain  the
            matrix dA.
    
    @param[in]
    ldda    INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. When  transA = MagmaNoTrans then
            LDA must be at least  max( 1, m ), otherwise  LDA must be at
            least  max( 1, k ).
    
    @param[in]
    dB      DOUBLE_PRECISION array of DIMENSION ( LDB, kb ), where kb is
            n  when  transB = MagmaNoTrans,  and is  k  otherwise.
            Before entry with  transB = MagmaNoTrans,  the leading  k by n
            part of the array dB must contain the matrix dB, otherwise
            the leading  n by k  part of the array dB must contain  the
            matrix dB.
    
    @param[in]
    lddb    INTEGER.
            On entry, LDB specifies the first dimension of dB as declared
            in the calling (sub) program. When  transB = MagmaNoTrans then
            LDB must be at least  max( 1, k ), otherwise  LDB must be at
            least  max( 1, n ).
    
    @param[in]
    beta    DOUBLE_PRECISION.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then dC need not be set on input.
    
    @param[in,out]
    dC      DOUBLE_PRECISION array of DIMENSION ( LDC, n ).
            Before entry, the leading  m by n  part of the array  dC must
            contain the matrix  dC,  except when  beta  is zero, in which
            case dC need not be set on entry.
            On exit, the array  dC  is overwritten by the  m by n  matrix
            ( alpha*op( dA )*op( dB ) + beta*dC ).
    
    @param[in]
    lddc    INTEGER.
            On entry, LDC specifies the first dimension of dC as declared
            in  the  calling  (sub)  program.   LDC  must  be  at  least
            max( 1, m ).

    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magmablas_dgemm(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // call CUDA ARCH 1.x version
        // magmablas for [sd] precisions, cublas for [zc] precisions.
        #if defined(PRECISION_z) || defined(PRECISION_c)
        magma_dgemm(
            transA, transB,
            m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc );
        #else
        magmablas_dgemm_tesla(
            transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc );
        #endif
        return;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( m <= 0 || n <= 0 || k <= 0 )
        return;
    
    size_t offsetA = 0;
    size_t offsetB = 0;

    int TransA = 2, TransB = 2;
    if      ( transA == MagmaTrans )
        TransA = 1;
    else if ( transA == MagmaNoTrans )
        TransA = 0;
                    
    if      ( transB == MagmaTrans )
        TransB = 1;
    else if ( transB == MagmaNoTrans )
        TransB = 0;

    magma_int_t Am = ( ! TransA ? m : k);
    magma_int_t An = (!TransA ? k : m);
    magma_int_t Bm = ( ! TransB ? k : n);
    magma_int_t Bn = (!TransB ? n : k);
    size_t sizeA = (size_t) ldda * (An - 1) + Am;
    size_t sizeB = (size_t) lddb * (Bn - 1) + Bm;

    size_t CUBLAS_MAX_1DBUF_SIZE = ((1 << 27) - 512);
    if ( sizeA >= CUBLAS_MAX_1DBUF_SIZE ||
         sizeB >= CUBLAS_MAX_1DBUF_SIZE )
    {
        magma_dgemm( transA, transB, m, n, k, alpha,
                     dA, ldda, dB, lddb,
                     beta, dC, lddc );
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
        cudaError_t err;
        err = cudaBindTexture(&offsetA, tex_ref_A, dA, sizeA*sizeof(double));
        if ( err != cudaSuccess ) {
            fprintf( stderr, "cannot bind A to texture: %s (%d)\n", cudaGetErrorString(err), err );
            return;
        }
        err = cudaBindTexture(&offsetB, tex_ref_B, dB, sizeB*sizeof(double));
        if ( err != cudaSuccess ) {
            fprintf( stderr, "cannot bind B to texture: %s (%d)\n", cudaGetErrorString(err), err );
            cudaUnbindTexture( tex_ref_A );
            return;
        }
    #endif

    // Set up grids
    dim3 dimBlock(DIM_X, DIM_Y);

    offsetA = offsetA/sizeof(dA[0]);
    offsetB = offsetB/sizeof(dB[0]);
 
    if ( TransA == 0 && TransB == 0 ) {
        dim3 dimGrid( (m - 1)/BLK_M_nn + 1,
                      (n - 1)/BLK_N_nn + 1 );
        dgemm_kernel_fermi_nn<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 0 && TransB == 1 ) {
        dim3 dimGrid( (m - 1)/BLK_M_nt + 1,
                      (n - 1)/BLK_N_nt + 1 );
        dgemm_kernel_fermi_nt<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 0 && TransB == 2 ) {
        dim3 dimGrid( (m - 1)/BLK_M_nc + 1,
                      (n - 1)/BLK_N_nc + 1 );
        dgemm_kernel_fermi_nc<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 1 && TransB == 0 ) {
        dim3 dimGrid( (m - 1)/BLK_M_tn + 1,
                      (n - 1)/BLK_N_tn + 1 );
        dgemm_kernel_fermi_tn<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 1 && TransB == 1 ) {
        dim3 dimGrid( (m - 1)/BLK_M_tt + 1,
                      (n - 1)/BLK_N_tt + 1 );
        dgemm_kernel_fermi_tt<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 1 && TransB == 2 ) {
        dim3 dimGrid( (m - 1)/BLK_M_tc + 1,
                      (n - 1)/BLK_N_tc + 1 );
        dgemm_kernel_fermi_tc<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 2 && TransB == 0 ) {
        dim3 dimGrid( (m - 1)/BLK_M_cn + 1,
                      (n - 1)/BLK_N_cn + 1 );
        dgemm_kernel_fermi_cn<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 2 && TransB == 1 ) {
        dim3 dimGrid( (m - 1)/BLK_M_ct + 1,
                      (n - 1)/BLK_N_ct + 1 );
        dgemm_kernel_fermi_ct<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }
    else if ( TransA == 2 && TransB == 2 ) {
        dim3 dimGrid( (m - 1)/BLK_M_cc + 1,
                      (n - 1)/BLK_N_cc + 1 );
        dgemm_kernel_fermi_cc<<< dimGrid, dimBlock, 0, magma_stream >>>(
            m, n, k, dA, ldda, dB, lddb, dC, lddc, alpha, beta,
            (int)offsetA, (int)offsetB );
    }

    #ifdef TEXTURE_1D
        cudaUnbindTexture( tex_ref_A );
        cudaUnbindTexture( tex_ref_B );
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
