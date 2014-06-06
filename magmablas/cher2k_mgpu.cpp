/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:45 2013
       @author Mark Gates
       @author Azzam Haidar 
*/
#include "common_magma.h"

/*
    Purpose
    =======
    CHER2K performs one of the Hermitian rank 2k operations

       C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,

    or

       C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

    where alpha and beta are scalars with beta real, C is an n by n
    Hermitian matrix and A and B are n by k matrices in the first case
    and k by n matrices in the second case.

    Arguments
    ==========

    UPLO     (input) CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array C is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of C
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of C
                                    is to be referenced.

             **** current only Lower case is implemented.

    TRANS    (input) CHARACTER*1.
             On entry, TRANS specifies the operation to be performed as
             follows:

                TRANS = 'N' or 'n'
                  C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C.

                TRANS = 'C' or 'c'
                  C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C.

             **** current only NoTrans case is implemented.

    N        (input) INTEGER.
             On entry, N specifies the order of the matrix C. N must be
             at least zero.

    K        (input) INTEGER.
             On entry with TRANS = 'N' or 'n', K specifies the number
             of columns of the matrices A and B, and on entry with
             TRANS = 'C' or 'c', K specifies the number of rows of the
             matrices A and B. K must be at least zero.

    ALPHA    (input) COMPLEX.
             On entry, ALPHA specifies the scalar alpha.

    dA       (input) COMPLEX array of DIMENSION ( LDA, ka ), where ka is
             k when TRANS = 'N' or 'n', and is n otherwise.
             Before entry with TRANS = 'N' or 'n', the leading n by k
             part of the array A must contain the matrix A, otherwise
             the leading k by n part of the array A must contain the
             matrix A.
             
             [TODO: describe distribution: duplicated on all GPUs.]

    LDA      (input) INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When TRANS = 'N' or 'n'
             then LDA must be at least max( 1, n ), otherwise LDA must
             be at least max( 1, k ).

    AOFFSET  (input) INTEGER
             Row offset to start sub-matrix of dA. Uses dA(aoffset:aoffset+n, :).
             0 <= aoffset < lda.
             
    dB       (input) COMPLEX array of DIMENSION ( LDB, kb ), where kb is
             k when TRANS = 'N' or 'n', and is n otherwise.
             Before entry with TRANS = 'N' or 'n', the leading n by k
             part of the array B must contain the matrix B, otherwise
             the leading k by n part of the array B must contain the
             matrix B.
             
             [TODO: describe distribution: duplicated on all GPUs.]

    LDB      (input) INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When TRANS = 'N' or 'n'
             then LDB must be at least max( 1, n ), otherwise LDB must
             be at least max( 1, k ).

    BOFFSET  (input) INTEGER
             Row offset to start sub-matrix of dB. Uses dB(boffset:boffset+n, :).
             0 <= boffset < ldb.
             
    BETA     (input) REAL.
             On entry, BETA specifies the scalar beta.

    dC       (input/output) COMPLEX array of DIMENSION ( LDC, n ).
             Before entry with UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array C must contain the upper
             triangular part of the Hermitian matrix and the strictly
             lower triangular part of C is not referenced. On exit, the
             upper triangular part of the array C is overwritten by the
             upper triangular part of the updated matrix.

             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array C must contain the lower
             triangular part of the Hermitian matrix and the strictly
             upper triangular part of C is not referenced. On exit, the
             lower triangular part of the array C is overwritten by the
             lower triangular part of the updated matrix.

             Note that the imaginary parts of the diagonal elements need
             not be set, they are assumed to be zero, and on exit they
             are set to zero. [TODO: verify]
             
             [TODO: describe distribution: 1D column block-cyclic across GPUs.]

    LDC      (input) INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in the calling (sub) program. LDC must be at least max( 1, n ).

    COFFSET  (input) INTEGER.
             Row and column offset to start sub-matrix of dC.
             Uses dC(coffset:coffset+n, coffset:coffset+n).
             0 <= coffset < ldc.

    NGPU     (input) INTEGER.
             Number of GPUs over which matrix C is distributed.

    NB       (input) INTEGER.
             Block size used for distribution of C.

    STREAMS  (input) array of CUDA streams, of dimension NGPU by 20.
             Streams to use for running multiple GEMMs in parallel.
             Only up to NSTREAM streams are used on each GPU.

    NSTREAM  (input) INTEGER.
             Number of streams to use on each device
             
*/

extern "C"
void magmablas_cher2k_mgpu2(
    char uplo, char trans, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha, magmaFloatComplex *dA[], magma_int_t lda, magma_int_t aoffset,
                           magmaFloatComplex *dB[], magma_int_t ldb, magma_int_t boffset,
    float beta,           magmaFloatComplex *dC[], magma_int_t ldc, magma_int_t coffset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t streams[][20], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*lda + (aoffset) )
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*ldb + (boffset) )
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*ldc)
    
    /* Check arguments */
    magma_int_t info = 0;
    if ( ! (uplo == 'l' || uplo == 'L')) {
        info = -1;  // 'u' not yet handled
    } else if ( ! (trans == 'n' || trans == 'N')) {
        info = -2;  // 'c' not yet handled
    } else if ( n < 0 ) {
        info = -3;
    } else if ( k < 0 ) {
        info = -4;
    } else if ( ((trans == 'n' || trans == 'N') && lda < max(1,n)) ||
                ((trans == 'c' || trans == 'C') && lda < max(1,k)) ) {
        info = -7;
    } else if ( aoffset < 0 || aoffset > lda ) {
        info = -8;
    } else if ( ((trans == 'n' || trans == 'N') && ldb < max(1,n)) ||
                ((trans == 'c' || trans == 'C') && ldb < max(1,k)) ) {
        info = -10;
    } else if ( boffset < 0 || boffset > ldb ) {
        info = -11;
    } else if ( ldc < max(1,n) ) {
        info = -13;
    } else if ( coffset < 0 || coffset > ldc ) {
        info = -14;
    } else if ( ngpu <= 0 ) {
        info = -15;
    } else if ( nb <= 0 ) {
        info = -16;
    } else if ( nstream <= 0 ) {
        info = -18;
    }
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    const magmaFloatComplex c_one = MAGMA_C_ONE;
    magmaFloatComplex cbeta = MAGMA_C_MAKE( beta, 0. );
    
    magma_int_t ib, ioff, iblock, idev, di, s;
    
    magma_device_t cdev;
    magma_queue_t cqueue;
    magma_getdevice( &cdev );
    magmablasGetKernelStream( &cqueue );
    
    // loop over all blocks
    // Faster to have two loops: first loop does C_hat = alpha*A*B' + beta*C
    // blockoffset is offset within first block; for subsequent blocks it is 0
    magma_int_t blockoffset = coffset % nb;
    for( magma_int_t i = 0; i < n; i += ib ) {
        ib     = min( nb-blockoffset, n-i );  // block size
        ioff   = i + coffset;                 // global index in parent matrix
        iblock = (ioff / nb) / ngpu;          // local block id
        idev   = (ioff / nb) % ngpu;          // device with this block
        di     = iblock*nb + blockoffset;     // local index in parent matrix
        
        magma_setdevice( idev );
        s = iblock % nstream;
        magmablasSetKernelStream( streams[ idev ][ s ] );
        
        // C[i:n,i] = alpha * A[i:n,0] * B[i,0]' + beta*C[i:n,i]
        //printf( "cgemm  n=%4d, ib=%4d, k=%4d, i=%4d\n", n-i, ib, k, i );
        magma_cgemm( MagmaNoTrans, MagmaConjTrans, n-i, ib, k,
                     alpha, dA(idev,i,0), lda,
                            dB(idev,i,0), ldb,
                     cbeta, dC(idev,ioff,di), ldc );
        blockoffset = 0;
    }
    
    // second loop does C = conjf(alpha)*B*A' + C_hat
    alpha = MAGMA_C_CNJG( alpha );
    blockoffset = coffset % nb;
    for( magma_int_t i = 0; i < n; i += ib ) {
        ib     = min( nb-blockoffset, n-i );  // block size
        ioff   = i + coffset;                 // global index in parent matrix
        iblock = (ioff / nb) / ngpu;          // local block id
        idev   = (ioff / nb) % ngpu;          // device with this block
        di     = iblock*nb + blockoffset;     // local index in parent matrix
        
        magma_setdevice( idev );
        s = iblock % nstream;
        magmablasSetKernelStream( streams[ idev ][ s ] );
        
        // C[i:n,i] += conjf(alpha) * B[i:n,0] * A[i,0]'
        //printf( "cgemm  n=%4d, ib=%4d, k=%4d, i=%4d\n", n-i, ib, k, i );
        magma_cgemm( MagmaNoTrans, MagmaConjTrans, n-i, ib, k,
                     alpha, dB(idev,i,0), ldb,
                            dA(idev,i,0), lda,
                     c_one, dC(idev,ioff,di), ldc );
        blockoffset = 0;
    }
    
    magma_setdevice( cdev );
    magmablasSetKernelStream( cqueue );
}
