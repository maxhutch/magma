/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar

       [zcds]gemm_fermi.cu          defines the CPU driver.
       [zcds]gemm_fermi_kernels.h   defines the block sizes for each precision.
       gemm_stencil_defs.h          defines types and functions for precision-independent code.
       
       These files are included multiple times, once for each transpose version.
       herk_stencil.cuh             defines the GPU kernel (device function).
       herk_kernel_batched.cuh              defines the GPU kernel (global function).
       
       The batched version uses herk_kernel_batched.cuh instead of herk_kernel.cuh.
*/
#include "common_magma.h"
#include "commonblas_z.h"

#define PRECISION_z

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "zgemm_fermi_kernels_batched_k32.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    ZHERK performs one of the hermitian rank k operations

    C := alpha*A*A**H + beta*C,

    or

    C := alpha*A**H*A + beta*C,

    where alpha and beta are real scalars, C is an n by n hermitian
    matrix and A is an n by k matrix in the first case and a k by n
    matrix in the second case.
    
    Parameters
    ----------

    @param[in]
    uplo    CHARACTER*1.
           On entry, uplo specifies whether the upper or lower
           triangular part of the array C is to be referenced as
           follows:

           uplo = 'U' or 'u' Only the upper triangular part of C
           is to be referenced.

           uplo = 'L' or 'l' Only the lower triangular part of C
           is to be referenced.
    
    @param[in]
    trans   CHARACTER*1.
            On entry, trans specifies the operation to be performed as
            follows:

            trans = 'N' or 'n' C := alpha*A*A**H + beta*C.

            trans = 'C' or 'c' C := alpha*A**H*A + beta*C.

    @param[in]
    n       INTEGER.
            On entry,  specifies the order of the matrix C. N must be
            at least zero.
    
    @param[in]
    k       INTEGER.
            On entry with trans = 'N' or 'n', k specifies the number
            of columns of the matrix A, and on entry with
            trans = 'C' or 'c', k specifies the number of rows of the
            matrix A. K must be at least zero.

    @param[in]
    alpha   DOUBLE PRECISION
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA      COMPLEX_16 array of DIMENSION ( ldda, ka ), where ka is
            k  when  trans = MagmaNoTrans,  and is  n  otherwise.
            Before entry with  trans = MagmaNoTrans,  the leading  m by k
            part of the array dA must contain the matrix dA, otherwise
            the leading  k by m  part of the array dA must contain  the
            matrix dA.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A as declared
            in the calling (sub) program. When  trans = MagmaNoTrans then
            ldda must be at least  max( 1, n ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    beta    DOUBLE PRECISION.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then dC need not be set on input.
    
    @param[in,out]
    dC      COMPLEX_16 array of DIMENSION ( lddc, n ).
            Before entry with uplo = 'U' or 'u', the leading n by n
            upper triangular part of the array C must contain the upper
            triangular part of the hermitian matrix and the strictly
            lower triangular part of C is not referenced. On exit, the
            upper triangular part of the array C is overwritten by the
            upper triangular part of the updated matrix.
            Before entry with uplo = 'L' or 'l', the leading n by n
            lower triangular part of the array C must contain the lower
            triangular part of the hermitian matrix and the strictly
            upper triangular part of C is not referenced. On exit, the
            lower triangular part of the array C is overwritten by the
            lower triangular part of the updated matrix.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero, and on exit they
            are set to zero.

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of dC as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magmablas_zherk_batched_k32(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue )
{
    magmaDoubleComplex cbeta  = MAGMA_Z_MAKE( beta, 0. );
    magmaDoubleComplex calpha = MAGMA_Z_MAKE( alpha, 0. );

    magma_int_t info = 0;
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        printf("not supported \n"); // TODO call cublas
        return;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( n <= 0 || k <= 0 )
        return;
    
    size_t offsetA = 0;
    int TransA = 0, TransB = 0, uploA = 0;

    if      ( uplo == MagmaLower )
        uploA = 1;
    else if ( uplo == MagmaUpper )
        uploA = 2;

    if      ( trans == MagmaNoTrans )
        #if defined(PRECISION_z) || defined(PRECISION_c)     
        TransB = 2;
        #else
        TransB = 1;
        #endif
    else if ( trans == MagmaTrans || trans == MagmaConjTrans)
        #if defined(PRECISION_z) || defined(PRECISION_c)     
        TransA = 2;
        #else
        TransA = 1;
        #endif


    #ifdef TEXTURE_1D
        size_t sizeA = (size_t) ldda * (size_t) (!TransA ? k : n);

        size_t CUBLAS_MAX_1DBUF_SIZE = ((1 << 27) - 512);
        if ( sizeA >= CUBLAS_MAX_1DBUF_SIZE )
        {
            printf("not supported \n"); // TODO call cublas
            return;
        }
        // Set textures parameters
        tex_ref_A.normalized = false;
        tex_ref_A.filterMode = cudaFilterModePoint;
        tex_ref_A.addressMode[0] = cudaAddressModeClamp;

        // Bind A and B to texture references
        cudaError_t err;
        err = cudaBindTexture(&offsetA, tex_ref_A, dA_array[0], sizeA*sizeof(magmaDoubleComplex));
        if ( err != cudaSuccess ) {
            fprintf( stderr, "cannot bind A to texture: %s (%d)\n", cudaGetErrorString(err), err );
            return;
        }
    #endif

    // Set up grids
    dim3 dimBlock(DIM_X, DIM_Y);

    offsetA = offsetA/sizeof(magmaDoubleComplex);
 
    if ( TransA == 0 && TransB == 1 ) {
        dim3 dimGrid( (n - 1)/BLK_M_nt + 1,
                      (n - 1)/BLK_N_nt + 1 ,
                      batchCount );
        magmablas_z_herk_kernel_fermi_nt_batched<<< dimGrid, dimBlock, 0, queue >>>(
            uploA, n, k, dA_array, ldda, dA_array, ldda, dC_array, lddc, calpha, cbeta,
            (int)offsetA, (int)offsetA );
    }
    else if ( TransA == 0 && TransB == 2 ) {
        dim3 dimGrid( (n - 1)/BLK_M_nc + 1,
                      (n - 1)/BLK_N_nc + 1 ,
                      batchCount );
         magmablas_z_herk_kernel_fermi_nc_batched<<< dimGrid, dimBlock, 0, queue >>>(
            uploA, n, k, dA_array, ldda, dA_array, ldda, dC_array, lddc, calpha, cbeta,
            (int)offsetA, (int)offsetA );
    }
    else if ( TransA == 1 && TransB == 0 ) {
        dim3 dimGrid( (n - 1)/BLK_M_tn + 1,
                      (n - 1)/BLK_N_tn + 1 ,
                      batchCount );
         magmablas_z_herk_kernel_fermi_tn_batched<<< dimGrid, dimBlock, 0, queue >>>(
            uploA, n, k, dA_array, ldda, dA_array, ldda, dC_array, lddc, calpha, cbeta,
            (int)offsetA, (int)offsetA );
    }
    else if ( TransA == 2 && TransB == 0 ) {
        dim3 dimGrid( (n - 1)/BLK_M_cn + 1,
                      (n - 1)/BLK_N_cn + 1 ,
                      batchCount );
         magmablas_z_herk_kernel_fermi_cn_batched<<< dimGrid, dimBlock, 0, queue >>>(
            uploA, n, k, dA_array, ldda, dA_array, ldda, dC_array, lddc, calpha, cbeta,
            (int)offsetA, (int)offsetA );
    }

    #ifdef TEXTURE_1D
        cudaUnbindTexture( tex_ref_A );
    #endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
