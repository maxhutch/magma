/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Stan Tomov
       @author Mark Gates
       @generated d Tue Dec 17 13:18:45 2013
*/
#include "common_magma.h"
#include "commonblas.h"

#define PRECISION_d

//===========================================================================
// Set a matrix from CPU to multi-GPUs in 1D column block cyclic distribution.
// The dA arrays are pointers to the matrix data on the corresponding GPUs.
//===========================================================================
extern "C" void
magma_dsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA,   magma_int_t lda,
    double       *dA[], magma_int_t ldda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t j, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_dsetmatrix_async( m, jb,
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
magma_dgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    double  *dA[], magma_int_t ldda,
    double  *hA,   magma_int_t lda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t j, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( j = 0; j < n; j += nb ) {
        dev = (j/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, n-j);
        magma_dgetmatrix_async( m, jb,
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
magma_dsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA,   magma_int_t lda,
    double       *dA[], magma_int_t ldda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t i, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_dsetmatrix_async( jb, n,
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
magma_dgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    double  *dA[], magma_int_t ldda,
    double  *hA,   magma_int_t lda,
    magma_int_t num_gpus, magma_int_t nb )
{
    magma_int_t i, dev, jb;
    magma_device_t cdevice;

    magma_getdevice( &cdevice );

    for( i = 0; i < m; i += nb ) {
        dev = (i/nb) % num_gpus;
        magma_setdevice( dev );
        jb = min(nb, m-i);
        magma_dgetmatrix_async( jb, n,
                                dA[dev] + i/(nb*num_gpus)*nb, ldda,
                                hA + i, lda, NULL );
    }

    magma_setdevice( cdevice );
}
