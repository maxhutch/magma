/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zcgetrs_gpu.cpp mixed zc -> ds, Mon May  2 23:30:02 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    DSGETRS solves a system of linear equations
       A * X = B,  A**T * X = B,  or  A**H * X = B
    with a general N-by-N matrix A using the LU factorization computed
    by MAGMA_SGETRF_GPU. B and X are in DOUBLE PRECISION, and A is in SINGLE PRECISION.
    This routine is used in the mixed precision iterative solver
    magma_dsgesv.

    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
            Specifies the form of the system of equations:
      -     = MagmaNoTrans:    A * X = B     (No transpose)
      -     = MagmaTrans:      A**T * X = B  (Transpose)
      -     = MagmaConjTrans:  A**H * X = B  (Conjugate transpose)

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA      SINGLE PRECISION array on the GPU, dimension (LDDA,N)
            The factors L and U from the factorization A = P*L*U
            as computed by CGETRF_GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).

    @param[in]
    dipiv   INTEGER array on the GPU, dimension (N)
            The pivot indices; for 1 <= i <= N, after permuting, row i of the
            matrix was moved to row dIPIV(i).
            Note this is different than IPIV from DGETRF, where interchanges
            are applied one-after-another.

    @param[in]
    dB      DOUBLE PRECISION array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the arrays X and B.  LDDB >= max(1,N).

    @param[out]
    dX      DOUBLE PRECISION array on the GPU, dimension (LDDX, NRHS)
            On exit, the solution matrix dX.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX, LDDX >= max(1,N).

    @param
    dSX     (workspace) SINGLE PRECISION array on the GPU used as workspace,
            dimension (N, NRHS)

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_dgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dsgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr  dA, magma_int_t ldda,
    magmaInt_ptr        dipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magmaDouble_ptr dX, magma_int_t lddx,
    magmaFloat_ptr dSX,
    magma_int_t *info)
{
    /* Constants */
    float c_one = MAGMA_S_ONE;
    
    /* Local variables */
    bool notran = (trans == MagmaNoTrans);
    magma_int_t inc;
    magma_int_t lddsx = n;

    *info = 0;
    if ( (! notran) &&
         (trans != MagmaTrans) &&
         (trans != MagmaConjTrans) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldda < n) {
        *info = -5;
    } else if (lddb < n) {
        *info = -8;
    } else if (lddx < n) {
        *info = -10;
    }
    // I think this is resolved, but it is unclear what the issue ever was.
    //else if (lddx != lddb) { /* TODO: remove it when dslaswp will have the correct interface */
    //    *info = -10;
    //}
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }
    
    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    if (notran) {
        inc = 1;
        
        /* Get X by row applying interchanges to B and cast to single */
        /*
         * TODO: clean dslaswp interface to have interface closer to zlaswp
         */
        magmablas_dslaswp( nrhs, dB, lddb, dSX, lddsx,
                           n, dipiv, inc, queue );
        
        /* Solve L*X = B, overwriting B with SX. */
        magma_strsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                     n, nrhs, c_one, dA, ldda, dSX, lddsx, queue );
        
        /* Solve U*X = B, overwriting B with X. */
        magma_strsm( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                     n, nrhs, c_one, dA, ldda, dSX, lddsx, queue );
        
        magmablas_slag2d( n, nrhs, dSX, lddsx, dX, lddx, queue, info );
    }
    else {
        inc = -1;
        
        /* Cast the DOUBLE PRECISION RHS to SINGLE PRECISION */
        magmablas_dlag2s( n, nrhs, dB, lddb, dSX, lddsx, queue, info );
        
        /* Solve A**T * X = B, or A**H * X = B */
        magma_strsm( MagmaLeft, MagmaUpper, trans, MagmaNonUnit,
                     n, nrhs, c_one, dA, ldda, dSX, lddsx, queue );
        
        magma_strsm( MagmaLeft, MagmaLower, trans, MagmaUnit,
                     n, nrhs, c_one, dA, ldda, dSX, lddsx, queue );
        
        magmablas_dslaswp( nrhs, dX, lddx, dSX, lddsx,
                           n, dipiv, inc, queue );
    }
    
    magma_queue_destroy( queue );

    return *info;
} /* magma_dsgetrs */
