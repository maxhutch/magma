/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @precisions normal z -> s d c
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

#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>


using namespace std;



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    This routine computes the fill-in structure of an ILU(levels) factorization
    based on the successive multiplication of upper and lower triangular factors
    using the CUSPARSE library.

    Arguments
    =========

    magma_z_sparse_matrix *A             matrix in magma sparse matrix format
    magma_int_t levels                   fill in level

    ========================================================================  */

extern "C"
magma_int_t 
magma_zilustruct( magma_z_sparse_matrix *A, magma_int_t levels ){

    
    if( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ){

        magma_int_t i, j, k, m;

        magma_z_sparse_matrix B, L, U, L_d, U_d, LU_d;

        magma_z_mconvert( *A, &B, Magma_CSR, Magma_CSR );

        for( i=0; i<levels; i++ ){

            magma_z_mconvert( *A, &L, Magma_CSR, Magma_CSRL );
            magma_z_mconvert( *A, &U, Magma_CSR, Magma_CSRU );

            magma_z_mtransfer( L, &L_d, Magma_CPU, Magma_DEV ); 
            magma_z_mtransfer( U, &U_d, Magma_CPU, Magma_DEV ); 
            magma_z_mtransfer( U, &LU_d, Magma_CPU, Magma_DEV ); 

            magma_z_mfree( &L );
            magma_z_mfree( &U );

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
            cudaMalloc((void**)&LU_d.row, sizeof(magma_index_t)*(L_d.num_rows+1));
            cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                        L_d.num_rows, L_d.num_rows, L_d.num_rows, 
                                        descrL, L_d.nnz, L_d.row, L_d.col,
                                        descrU, U_d.nnz, U_d.row, U_d.col,
                                        descrLU, LU_d.row, nnzTotalDevHostPtr );
            if (NULL != nnzTotalDevHostPtr){
                LU_d.nnz = *nnzTotalDevHostPtr;
            }else{
                cudaMemcpy(&LU_d.nnz, LU_d.row+m, sizeof(magma_index_t), 
                                                    cudaMemcpyDeviceToHost);
                cudaMemcpy(&baseC, LU_d.row, sizeof(magma_index_t), 
                                                    cudaMemcpyDeviceToHost);
                LU_d.nnz -= baseC;
            }
            cudaMalloc((void**)&LU_d.col, sizeof(magma_index_t)*LU_d.nnz);
            cudaMalloc((void**)&LU_d.val, sizeof(magmaDoubleComplex)*LU_d.nnz);
            cusparseZcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
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

            magma_z_mtransfer(LU_d, A, Magma_DEV, Magma_CPU);
            magma_z_mfree( &L_d );
            magma_z_mfree( &U_d );
            magma_z_mfree( &LU_d );

        }

        for( i=0; i<A->nnz; i++ )
            A->val[i] = MAGMA_Z_MAKE( (double)(0.0/A->nnz), 0.0 );

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
        magma_z_mfree( &B );


        return MAGMA_SUCCESS;
    }
    else{

        magma_z_sparse_matrix hA, CSRCOOA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_z_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRCOOA, hA.storage_type, Magma_CSR );

        magma_zilustruct( &CSRCOOA, levels );

        magma_z_mfree( &hA );
        magma_z_mfree( A );
        magma_z_mconvert( CSRCOOA, &hA, Magma_CSR, A_storage );
        magma_z_mtransfer( hA, A, Magma_CPU, A_location );
        magma_z_mfree( &hA );
        magma_z_mfree( &CSRCOOA );    

        return MAGMA_SUCCESS; 
    }
}
