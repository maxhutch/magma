/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

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

    This is an interface to the cuSPARSE routine csrmm computing the product
    of two sparse matrices stored in csr format. 


    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix 

    @param
    B           magma_z_sparse_matrix
                input matrix 

    @param
    AB          magma_z_sparse_matrix*
                output matrix AB = A * B


    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcuspmm( magma_z_sparse_matrix A, magma_z_sparse_matrix B, 
                                            magma_z_sparse_matrix *AB ){


    if(    A.memory_location == Magma_DEV 
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) ){

            magma_z_sparse_matrix C;
            C.num_rows = A.num_rows;
            C.num_cols = A.num_cols;
            C.storage_type = A.storage_type;
            C.memory_location = A.memory_location;


            // CUSPARSE context //
            cusparseHandle_t handle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&handle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");

            cusparseMatDescr_t descrA;
            cusparseMatDescr_t descrB;
            cusparseMatDescr_t descrC;
            cusparseStatus = cusparseCreateMatDescr(&descrA);
            cusparseStatus = cusparseCreateMatDescr(&descrB);
            cusparseStatus = cusparseCreateMatDescr(&descrC);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatType(descrC,CUSPARSE_MATRIX_TYPE_GENERAL);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatIndexBase(descrC,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");

            // multiply A and B on the device
            magma_int_t baseC;
            // nnzTotalDevHostPtr points to host memory
            magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &C.nnz;
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
            magma_index_malloc( &C.row, (A.num_rows + 1) );
            cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        A.num_rows, A.num_rows, A.num_rows, 
                                        descrA, A.nnz, A.row, A.col,
                                        descrB, B.nnz, B.row, B.col,
                                        descrC, C.row, nnzTotalDevHostPtr );
            if (NULL != nnzTotalDevHostPtr){
                C.nnz = *nnzTotalDevHostPtr;
            }else{
                // workaround as nnz and base C are magma_int_t 
                magma_index_t base_t, nnz_t; 
                magma_index_getvector( 1, C.row+C.num_rows, 1, &nnz_t, 1 );
                magma_index_getvector( 1, C.row,   1, &base_t,    1 );
                C.nnz = (magma_int_t) nnz_t;
                baseC = (magma_int_t) base_t;
                C.nnz -= baseC;
            }
            magma_index_malloc( &C.col, C.nnz );
            magma_zmalloc( &C.val, C.nnz );
            cusparseZcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            A.num_rows, A.num_rows, A.num_rows,
                            descrA, A.nnz,
                            A.val, A.row, A.col,
                            descrB, B.nnz,
                            B.val, B.row, B.col,
                            descrC,
                            C.val, C.row, C.col);



            cusparseDestroyMatDescr( descrA );
            cusparseDestroyMatDescr( descrB );
            cusparseDestroyMatDescr( descrC );
            cusparseDestroy( handle );
            // end CUSPARSE context //

            magma_z_mtransfer( C, AB, Magma_DEV, Magma_DEV );
            magma_z_mfree( &C );

        return MAGMA_SUCCESS; 
    }
    else{

        printf("error: CSRMM only supported on device and CSR format.\n");

        return MAGMA_SUCCESS; 
    }
}





