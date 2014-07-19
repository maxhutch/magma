/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_zilustruct.cpp normal z -> c, Fri Jul 18 17:34:30 2014
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from 
//  the IO functions provided by MatrixMarket

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>

#include "magmasparse_c.h"
#include "magma.h"
#include "mmio.h"

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>


using namespace std;



/**
    Purpose
    -------

    This routine computes the fill-in structure of an ILU(levels) factorization
    based on the successive multiplication of upper and lower triangular factors
    using the CUSPARSE library.

    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix*
                matrix in magma sparse matrix format

    @param
    levels      magma_int_t
                fill in level


    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t 
magma_cilustruct( magma_c_sparse_matrix *A, magma_int_t levels ){

    
    if( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ){

        magma_int_t i, j, k, m;

        magma_c_sparse_matrix B, L, U, L_d, U_d, LU_d;

        magma_c_mconvert( *A, &B, Magma_CSR, Magma_CSR );

        for( i=0; i<levels; i++ ){

            magma_c_mconvert( *A, &L, Magma_CSR, Magma_CSRL );
            magma_c_mconvert( *A, &U, Magma_CSR, Magma_CSRU );

            magma_c_mtransfer( L, &L_d, Magma_CPU, Magma_DEV ); 
            magma_c_mtransfer( U, &U_d, Magma_CPU, Magma_DEV ); 
            magma_c_mtransfer( U, &LU_d, Magma_CPU, Magma_DEV ); 

            magma_c_mfree( &L );
            magma_c_mfree( &U );

            magma_free( LU_d.val );
            magma_free( LU_d.col );
            magma_free( LU_d.row );

            // CUSPARSE context //
            cusparseHandle_t handle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&handle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");

            cusparseMatDescr_t descrL;
            cusparseMatDescr_t descrU;
            cusparseMatDescr_t descrLU;
            cusparseStatus = cusparseCreateMatDescr(&descrL);
            cusparseStatus = cusparseCreateMatDescr(&descrU);
            cusparseStatus = cusparseCreateMatDescr(&descrLU);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatType(descrLU,CUSPARSE_MATRIX_TYPE_GENERAL);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
            cusparseSetMatIndexBase(descrLU,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");

            // multiply L and U on the device
            magma_int_t baseC;
            // nnzTotalDevHostPtr points to host memory
            magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &LU_d.nnz;
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
            magma_index_malloc( &LU_d.row, (L_d.num_rows + 1) );
            cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        L_d.num_rows, L_d.num_rows, L_d.num_rows, 
                                        descrL, L_d.nnz, L_d.row, L_d.col,
                                        descrU, U_d.nnz, U_d.row, U_d.col,
                                        descrLU, LU_d.row, nnzTotalDevHostPtr );
            if (NULL != nnzTotalDevHostPtr){
                LU_d.nnz = *nnzTotalDevHostPtr;
            }else{
                // workaround as nnz and base C are magma_int_t 
                magma_index_t base_t, nnz_t; 
                magma_index_getvector( 1, LU_d.row+m, 1, &nnz_t, 1 );
                magma_index_getvector( 1, LU_d.row,   1, &base_t,    1 );
                LU_d.nnz = (magma_int_t) nnz_t;
                baseC = (magma_int_t) base_t;
                LU_d.nnz -= baseC;
            }
            magma_index_malloc( &LU_d.col, LU_d.nnz );
            magma_cmalloc( &LU_d.val, LU_d.nnz );
            cusparseCcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            L_d.num_rows, L_d.num_rows, L_d.num_rows,
                            descrL, L_d.nnz,
                            L_d.val, L_d.row, L_d.col,
                            descrU, U_d.nnz,
                            U_d.val, U_d.row, U_d.col,
                            descrLU,
                            LU_d.val, LU_d.row, LU_d.col);



            cusparseDestroyMatDescr( descrL );
            cusparseDestroyMatDescr( descrU );
            cusparseDestroyMatDescr( descrLU );
            cusparseDestroy( handle );
            // end CUSPARSE context //

            magma_c_mtransfer(LU_d, A, Magma_DEV, Magma_CPU);
            magma_c_mfree( &L_d );
            magma_c_mfree( &U_d );
            magma_c_mfree( &LU_d );

        }

        for( i=0; i<A->nnz; i++ )
            A->val[i] = MAGMA_C_MAKE( (float)(0.0/A->nnz), 0.0 );

        // take the original values as initial guess
        for(i=0; i<A->num_rows; i++){
            for(j=B.row[i]; j<B.row[i+1]; j++){
                magma_index_t lcol = B.col[j];
                for(k=A->row[i]; k<A->row[i+1]; k++){
                    if( A->col[k] == lcol ){
                        A->val[k] =  B.val[j];
                    }
                }
            }
        }
        magma_c_mfree( &B );


        return MAGMA_SUCCESS;
    }
    else{

        magma_c_sparse_matrix hA, CSRCOOA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_c_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_c_mconvert( hA, &CSRCOOA, hA.storage_type, Magma_CSR );

        magma_cilustruct( &CSRCOOA, levels );

        magma_c_mfree( &hA );
        magma_c_mfree( A );
        magma_c_mconvert( CSRCOOA, &hA, Magma_CSR, A_storage );
        magma_c_mtransfer( hA, A, Magma_CPU, A_location );
        magma_c_mfree( &hA );
        magma_c_mfree( &CSRCOOA );    

        return MAGMA_SUCCESS; 
    }
}

