/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Mark Gates
       @author Azzam Haidar 
*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZHER2K performs one of the Hermitian rank 2k operations

       C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,

    or

       C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

    where alpha and beta are scalars with beta real, C is an n by n
    Hermitian matrix and A and B are n by k matrices in the first case
    and k by n matrices in the second case.
    
    This version assumes C has been symmetrized, so both upper and lower are
    stored, and it maintains the symmetry, doing twice the operations.

    Arguments
    ----------

    @param[in]
    uplo     magma_uplo_t.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array C is to be referenced as
             follows:
      -     = MagmaUpper:  Only the upper triangular part of C is to be referenced.
      -     = MagmaLower:  Only the lower triangular part of C is to be referenced.

             **** current only Lower case is implemented.

    @param[in]
    trans    magma_trans_t.
             On entry, TRANS specifies the operation to be performed as
             follows:
      -     = MagmaNoTrans:     C := alpha*A*B**H + conj( alpha )*B*A**H + beta*C.
      -     = Magma_ConjTrans:  C := alpha*A**H*B + conj( alpha )*B**H*A + beta*C.

             **** current only NoTrans case is implemented.

    @param[in]
    n        INTEGER.
             On entry, N specifies the order of the matrix C. N must be
             at least zero.

    @param[in]
    k        INTEGER.
             On entry with TRANS = MagmaNoTrans, K specifies the number
             of columns of the matrices A and B, and on entry with
             TRANS = Magma_ConjTrans, K specifies the number of rows of the
             matrices A and B. K must be at least zero.

    @param[in]
    alpha    COMPLEX*16.
             On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA       COMPLEX*16 array of DIMENSION ( LDA, ka ), where ka is
             k when TRANS = MagmaNoTrans, and is n otherwise.
             Before entry with TRANS = MagmaNoTrans, the leading n by k
             part of the array A must contain the matrix A, otherwise
             the leading k by n part of the array A must contain the
             matrix A.
             
             [TODO: describe distribution: duplicated on all GPUs.]

    @param[in]
    lda      INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. When TRANS = MagmaNoTrans
             then LDA must be at least max( 1, n ), otherwise LDA must
             be at least max( 1, k ).

    @param[in]
    aoffset  INTEGER
             Row offset to start sub-matrix of dA. Uses dA(aoffset:aoffset+n, :).
             0 <= aoffset < lda.

    @param[in]
    dB       COMPLEX*16 array of DIMENSION ( LDB, kb ), where kb is
             k when TRANS = MagmaNoTrans, and is n otherwise.
             Before entry with TRANS = MagmaNoTrans, the leading n by k
             part of the array B must contain the matrix B, otherwise
             the leading k by n part of the array B must contain the
             matrix B.
             
             [TODO: describe distribution: duplicated on all GPUs.]

    @param[in]
    ldb      INTEGER.
             On entry, LDB specifies the first dimension of B as declared
             in the calling (sub) program. When TRANS = MagmaNoTrans
             then LDB must be at least max( 1, n ), otherwise LDB must
             be at least max( 1, k ).

    @param[in]
    boffset  INTEGER
             Row offset to start sub-matrix of dB. Uses dB(boffset:boffset+n, :).
             0 <= boffset < ldb.

    @param[in]
    beta     DOUBLE PRECISION.
             On entry, BETA specifies the scalar beta.

    @param[in,out]
    dC       COMPLEX*16 array of DIMENSION ( LDC, n ).
             Before entry with UPLO = MagmaUpper, the leading n by n
             upper triangular part of the array C must contain the upper
             triangular part of the Hermitian matrix and the strictly
             lower triangular part of C is not referenced. On exit, the
             upper triangular part of the array C is overwritten by the
             upper triangular part of the updated matrix.
    \n
             Before entry with UPLO = MagmaLower, the leading n by n
             lower triangular part of the array C must contain the lower
             triangular part of the Hermitian matrix and the strictly
             upper triangular part of C is not referenced. On exit, the
             lower triangular part of the array C is overwritten by the
             lower triangular part of the updated matrix.
    \n
             Note that the imaginary parts of the diagonal elements need
             not be set, they are assumed to be zero, and on exit they
             are set to zero. [TODO: verify]
             
             [TODO: describe distribution: 1D column block-cyclic across GPUs.]

    @param[in]
    ldc      INTEGER.
             On entry, LDC specifies the first dimension of C as declared
             in the calling (sub) program. LDC must be at least max( 1, n ).

    @param[in]
    coffset  INTEGER.
             Row and column offset to start sub-matrix of dC.
             Uses dC(coffset:coffset+n, coffset:coffset+n).
             0 <= coffset < ldc.

    @param[in]
    ngpu     INTEGER.
             Number of GPUs over which matrix C is distributed.

    @param[in]
    nb       INTEGER.
             Block size used for distribution of C.

    @param[in]
    streams  array of CUDA streams, of dimension NGPU by 20.
             Streams to use for running multiple GEMMs in parallel.
             Only up to NSTREAM streams are used on each GPU.

    @param[in]
    nstream  INTEGER.
             Number of streams to use on each device

    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_zher2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha, magmaDoubleComplex *dA[], magma_int_t lda, magma_int_t aoffset,
                           magmaDoubleComplex *dB[], magma_int_t ldb, magma_int_t boffset,
    double beta,           magmaDoubleComplex *dC[], magma_int_t ldc, magma_int_t coffset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t streams[][20], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*lda + (aoffset) )
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*ldb + (boffset) )
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*ldc)
    
    /* Check arguments */
    magma_int_t info = 0;
    if ( uplo != MagmaLower ) {
        info = -1;  // 'u' not yet handled
    } else if ( trans != MagmaNoTrans ) {
        info = -2;  // 'c' not yet handled
    } else if ( n < 0 ) {
        info = -3;
    } else if ( k < 0 ) {
        info = -4;
    } else if ( ((trans == MagmaNoTrans)    && lda < max(1,n)) ||
                ((trans == Magma_ConjTrans) && lda < max(1,k)) ) {
        info = -7;
    } else if ( aoffset < 0 || aoffset > lda ) {
        info = -8;
    } else if ( ((trans == MagmaNoTrans)    && ldb < max(1,n)) ||
                ((trans == Magma_ConjTrans) && ldb < max(1,k)) ) {
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
    
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex cbeta = MAGMA_Z_MAKE( beta, 0. );
    
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
        //printf( "zgemm  n=%4d, ib=%4d, k=%4d, i=%4d\n", n-i, ib, k, i );
        magma_zgemm( MagmaNoTrans, Magma_ConjTrans, n, ib, k,
                     alpha, dA(idev,0,0), lda,
                            dB(idev,i,0), ldb,
                     cbeta, dC(idev,coffset,di), ldc );
        blockoffset = 0;
    }
    
    // second loop does C = conj(alpha)*B*A' + C_hat
    alpha = MAGMA_Z_CNJG( alpha );
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
        
        // C[i:n,i] += conj(alpha) * B[i:n,0] * A[i,0]'
        //printf( "zgemm  n=%4d, ib=%4d, k=%4d, i=%4d\n", n-i, ib, k, i );
        magma_zgemm( MagmaNoTrans, Magma_ConjTrans, n, ib, k,
                     alpha, dB(idev,0,0), ldb,
                            dA(idev,i,0), lda,
                     c_one, dC(idev,coffset,di), ldc );
        blockoffset = 0;
    }
    
    magma_setdevice( cdev );
    magmablasSetKernelStream( cqueue );
}
