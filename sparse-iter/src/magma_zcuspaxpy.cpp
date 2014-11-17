/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    This is an interface to the cuSPARSE routine csrgeam computing the sum
    of two sparse matrices stored in csr format:

        C = alpha * A + beta * B


    Arguments
    ---------

    @param[in]
    alpha       magmaDoubleComplex*
                scalar

    @param[in]
    A           magma_z_sparse_matrix
                input matrix 

    @param[in]
    beta        magmaDoubleComplex*
                scalar

    @param[in]
    B           magma_z_sparse_matrix
                input matrix 

    @param[out]
    AB          magma_z_sparse_matrix*
                output matrix AB = alpha * A + beta * B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcuspaxpy(
    magmaDoubleComplex *alpha, magma_z_sparse_matrix A, 
    magmaDoubleComplex *beta, magma_z_sparse_matrix B, 
    magma_z_sparse_matrix *AB,
    magma_queue_t queue )
{
    if (    A.memory_location == Magma_DEV 
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) ) {

            magma_z_sparse_matrix C;
            C.num_rows = A.num_rows;
            C.num_cols = A.num_cols;
            C.storage_type = A.storage_type;
            C.memory_location = A.memory_location;
            magma_int_t stat_dev = 0;
            C.val = NULL;
            C.col = NULL;
            C.row = NULL;
            C.rowidx = NULL;
            C.blockinfo = NULL;
            C.diag = NULL;
            C.dval = NULL;
            C.dcol = NULL;
            C.drow = NULL;
            C.drowidx = NULL;
            C.ddiag = NULL;

            // CUSPARSE context //
            cusparseHandle_t handle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&handle);
            cusparseSetStream( handle, queue );
             if (cusparseStatus != 0)    printf("error in Handle.\n");

            cusparseMatDescr_t descrA;
            cusparseMatDescr_t descrB;
            cusparseMatDescr_t descrC;
            cusparseStatus = cusparseCreateMatDescr(&descrA);
            cusparseStatus = cusparseCreateMatDescr(&descrB);
            cusparseStatus = cusparseCreateMatDescr(&descrC);
             if (cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
             if (cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
             if (cusparseStatus != 0)    printf("error in IndexBase.\n");

            // multiply A and B on the device
            magma_int_t baseC;
            // nnzTotalDevHostPtr points to host memory
            magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &C.nnz;
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
            stat_dev += magma_index_malloc( &C.drow, (A.num_rows + 1) );

            cusparseXcsrgeamNnz(handle,A.num_rows, A.num_cols,
                        descrA, A.nnz, A.drow, A.dcol,
                        descrB, B.nnz, B.drow, B.dcol,
                        descrC, C.row, nnzTotalDevHostPtr);

            if (NULL != nnzTotalDevHostPtr) {
                C.nnz = *nnzTotalDevHostPtr;
            } else {
                // workaround as nnz and base C are magma_int_t 
                magma_index_t base_t, nnz_t; 
                magma_index_getvector( 1, C.drow+C.num_rows, 1, &nnz_t, 1 );
                magma_index_getvector( 1, C.drow,   1, &base_t,    1 );
                C.nnz = (magma_int_t) nnz_t;
                baseC = (magma_int_t) base_t;
                C.nnz -= baseC;
            }
            stat_dev += magma_index_malloc( &C.dcol, C.nnz );
            stat_dev += magma_zmalloc( &C.dval, C.nnz );
            if( stat_dev != 0 ){
                magma_z_mfree( &C, queue );
                return MAGMA_ERR_DEVICE_ALLOC;
            }

            cusparseZcsrgeam(handle, A.num_rows, A.num_cols,
                    alpha,
                    descrA, A.nnz,
                    A.dval, A.drow, A.dcol,
                    beta,
                    descrB, B.nnz,
                    B.dval, B.drow, B.dcol,
                    descrC,
                    C.dval, C.drow, C.dcol);


            cusparseDestroyMatDescr( descrA );
            cusparseDestroyMatDescr( descrB );
            cusparseDestroyMatDescr( descrC );
            cusparseDestroy( handle );
            // end CUSPARSE context //

            magma_z_mtransfer( C, AB, Magma_DEV, Magma_DEV, queue );
            magma_z_mfree( &C, queue );

        return MAGMA_SUCCESS; 
    }
    else {

        printf("error: CSRSPAXPY only supported on device and CSR format.\n");

        return MAGMA_SUCCESS; 
    }
}





