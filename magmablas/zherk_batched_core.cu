/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @precisions normal z

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/

#include "common_magma.h"
#define PRECISION_z

#include "herk_template_kernel_batched.cuh"
#include "gemm_config/zgemm_param_nn.h"
#include "gemm_config/zgemm_param_nt.h"
#include "gemm_config/zgemm_param_tn.h"
#include "gemm_config/zgemm_param_tt.h"
#define version(s,v) s ## _V_ ## v

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
    dA_array      Array of pointers, dimension (batchCount). 
             Each is a COMPLEX_16 array A of DIMENSION ( ldda, ka ), where ka is
             k  when  trans = MagmaNoTrans,  and is  n  otherwise.
             Before entry with  trans = MagmaNoTrans,  the leading  m by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by m  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of each array A as declared
            in the calling (sub) program. When  trans = MagmaNoTrans then
            ldda must be at least  max( 1, n ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    beta    DOUBLE PRECISION.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.
    
    @param[in,out]
    dC_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array C of DIMENSION ( lddc, n ).
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
            On entry, lddc specifies the first dimension of each array C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, m ).
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zblas3
    ********************************************************************/
void
magmablas_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magmaDoubleComplex cbeta  = MAGMA_Z_MAKE( beta, 0. );
    magmaDoubleComplex calpha = MAGMA_Z_MAKE( alpha, 0. );

    magma_int_t info = 0;
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    #if defined(PRECISION_c) || defined(PRECISION_z) 
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
    #else 
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
    #endif
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

    // we have two shapes only (nc or cn)
    magma_int_t shape = 0;
    if      (trans == MagmaNoTrans)   { shape = 0; } // nc
    else                              { shape = 1; } // cn
        
    //TODO: probably the texture init code should be placed here

    size_t offsetA = 0;
    size_t offsetB = 0;
    offsetA = offsetA/sizeof(magmaDoubleComplex);
    offsetB = offsetB/sizeof(magmaDoubleComplex);
    
    switch(shape)
    {
        case 0: // nc
            {
                if (k <= 8)
                {
                    // version 58
                    herk_template_batched_nt<magmaDoubleComplex, version(NT,58), 0, 1>
                    (uplo, n, k, dA_array, ldda, dC_array, lddc, calpha, cbeta, offsetA, offsetB, batchCount, queue);
                }
                else
                {
                    // version 29
                    herk_template_batched_nt<magmaDoubleComplex, version(NT,29), 0, 1>
                    (uplo, n, k, dA_array, ldda, dC_array, lddc, calpha, cbeta, offsetA, offsetB, batchCount, queue);
                }
            }
            break;
        case 1: // cn
            {
                // version 72
                herk_template_batched_tn<magmaDoubleComplex, version(TN,72), 1, 0>
                (uplo, n, k, dA_array, ldda, dC_array, lddc, calpha, cbeta, offsetA, offsetB, batchCount, queue);
            }
            break;
        default:; // propose something
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
