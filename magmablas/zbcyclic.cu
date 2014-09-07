/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Stan Tomov
       @author Mark Gates
       @precisions normal z -> s d c
*/
#include "common_magma.h"

#define PRECISION_z


//===========================================================================
// Set a matrix from CPU to multi-GPUs in 1D column block cyclic distribution.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_zsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex       *dA[], magma_int_t ldda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < m )
        info = -4;
    else if ( ldda < m )
        info = -6;
    else if ( num_gpus < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t j, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_zsetmatrix_async( m, jb,
                                hA + j*lda, lda,
                                dA[dev] + j/(nb*num_gpus)*nb*ldda, ldda, NULL );
    }

    magma_setdevice( cdevice );
}


//===========================================================================
// Get a matrix with 1D column block cyclic distribution from multi-GPUs to the CPU.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_zgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex  *dA[], magma_int_t ldda,
    magmaDoubleComplex  *hA,   magma_int_t lda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lda < m )
        info = -6;
    else if ( num_gpus < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t j, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_zgetmatrix_async( m, jb,
                                dA[dev] + j/(nb*num_gpus)*nb*ldda, ldda,
                                hA + j*lda, lda, NULL );
    }

    magma_setdevice( cdevice );
}


//===========================================================================
// Set a matrix from CPU to multi-GPUs in 1D row block cyclic distribution.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_zsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex       *dA[], magma_int_t ldda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( lda < m )
        info = -4;
    else if ( ldda < m )
        info = -6;
    else if ( num_gpus < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t i, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_zsetmatrix_async( jb, n,
                                hA + i, lda,
                                dA[dev] + i/(nb*num_gpus)*nb, ldda, NULL );
    }

    magma_setdevice( cdevice );
}


//===========================================================================
// Get a matrix with 1D row block cyclic distribution from multi-GPUs to the CPU.
// The dA arrays are pointers to the matrix data for the corresponding GPUs.
//===========================================================================
extern "C" void
magma_zgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex  *dA[], magma_int_t ldda,
    magmaDoubleComplex  *hA,   magma_int_t lda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lda < m )
        info = -6;
    else if ( num_gpus < 1 )
        info = -7;
    else if ( nb < 1 )
        info = -8;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t i, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_zgetmatrix_async( jb, n,
                                dA[dev] + i/(nb*num_gpus)*nb, ldda,
                                hA + i, lda, NULL );
    }

    magma_setdevice( cdevice );
}
