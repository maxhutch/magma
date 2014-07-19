/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_z_chowutils.cpp normal z -> c, Fri Jul 18 17:34:30 2014
       @author Hartwig Anzt
*/

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include <math.h>       /* fabs */
#include "../include/magmasparse_c.h"
#include "../../include/magma.h"
#include "../include/mmio.h"


// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

using namespace std;

#define PRECISION_c

/**
    Purpose
    -------

    Computes the Frobenius norm of the difference between the CSR matrices A 
    and B. They need to share the same sparsity pattern!


    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix
                sparse matrix in CSR

    @param
    B           magma_c_sparse_matrix
                sparse matrix in CSR    
                
    @param
    res         real_Double_t* 
                residual 

    @ingroup magmasparse_c
    ********************************************************************/

magma_int_t 
magma_cfrobenius( magma_c_sparse_matrix A, magma_c_sparse_matrix B, 
                  real_Double_t *res ){

    real_Double_t tmp2;
    magma_int_t i,j;
    
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){

            tmp2 = (real_Double_t) fabs( MAGMA_C_REAL(A.val[j] )
                                            - MAGMA_C_REAL(B.val[j]) );

            (*res) = (*res) + tmp2* tmp2;
        }      
    }

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS; 
}



/**
    Purpose
    -------

    Computes the nonlinear residual A- LU and returns the difference as
    well es the Frobenius norm of the difference


    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix
                input sparse matrix in CSR

    @param
    L           magma_c_sparse_matrix
                input sparse matrix in CSR    

    @param
    U           magma_c_sparse_matrix
                input sparse matrix in CSR    

    @param
    LU          magma_c_sparse_matrix*
                output sparse matrix in A-LU in CSR    

    @param
    res         real_Double_t* 
                residual 

    @ingroup magmasparse_caux
    ********************************************************************/

magma_int_t 
magma_cnonlinres(   magma_c_sparse_matrix A, 
                    magma_c_sparse_matrix L,
                    magma_c_sparse_matrix U, 
                    magma_c_sparse_matrix *LU, 
                    real_Double_t *res ){

    real_Double_t tmp2;
    magma_int_t i,j,k,m;

    magma_c_sparse_matrix L_d, U_d, LU_d, A_t;

    magma_c_mtransfer( L, &L_d, Magma_CPU, Magma_DEV ); 
    magma_c_mtransfer( U, &U_d, Magma_CPU, Magma_DEV ); 

    magma_c_mtransfer( U, &LU_d, Magma_CPU, Magma_DEV ); 

    magma_c_mtransfer( A, &A_t, Magma_CPU, Magma_CPU ); 

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

    // end CUSPARSE context //

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

    LU_d.storage_type = Magma_CSR;

    magma_c_mtransfer(LU_d, LU, Magma_DEV, Magma_CPU);
    magma_c_mfree( &L_d );
    magma_c_mfree( &U_d );
    magma_c_mfree( &LU_d );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t lcol = A.col[j];
            magmaFloatComplex newval = MAGMA_C_MAKE(0.0, 0.0);
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol ){
                    newval = MAGMA_C_MAKE(
                        MAGMA_C_REAL( LU->val[k] )- MAGMA_C_REAL( A.val[j] )
                                                , 0.0 );
                }
            }
            A_t.val[j] = newval;
        }
    }

    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            tmp2 = (real_Double_t) fabs( MAGMA_C_REAL(A_t.val[j]) );
            (*res) = (*res) + tmp2* tmp2;
        }
    }

    magma_c_mfree( LU );
    magma_c_mfree( &A_t );

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS; 
}

/**
    Purpose
    -------

    Computes the ILU residual A- LU and returns the difference as
    well es the Frobenius norm of the difference


    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix
                input sparse matrix in CSR

    @param
    L           magma_c_sparse_matrix
                input sparse matrix in CSR    

    @param
    U           magma_c_sparse_matrix
                input sparse matrix in CSR    

    @param
    LU          magma_c_sparse_matrix*
                output sparse matrix in A-LU in CSR    

    @param
    res         real_Double_t* 
                residual 

    @ingroup magmasparse_caux
    ********************************************************************/

magma_int_t 
magma_cilures(   magma_c_sparse_matrix A, 
                    magma_c_sparse_matrix L,
                    magma_c_sparse_matrix U, 
                    magma_c_sparse_matrix *LU, 
                    real_Double_t *res ){

    real_Double_t tmp2;
    magma_int_t i,j,k,m;

    magma_c_sparse_matrix L_d, U_d, LU_d;

    magma_c_mtransfer( L, &L_d, Magma_CPU, Magma_DEV ); 
    magma_c_mtransfer( U, &U_d, Magma_CPU, Magma_DEV ); 

    magma_c_mtransfer( U, &LU_d, Magma_CPU, Magma_DEV ); 

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

    // end CUSPARSE context //

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

    LU_d.storage_type = Magma_CSR;

    magma_c_mtransfer(LU_d, LU, Magma_DEV, Magma_CPU);
    magma_c_mfree( &L_d );
    magma_c_mfree( &U_d );
    magma_c_mfree( &LU_d );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t lcol = A.col[j];
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol ){
                    LU->val[k] = MAGMA_C_MAKE(
                        MAGMA_C_REAL( LU->val[k] )- MAGMA_C_REAL( A.val[j] )
                                                , 0.0 );
                }
            }
        }
    }

    for(i=0; i<LU->num_rows; i++){
        for(j=LU->row[i]; j<LU->row[i+1]; j++){
            tmp2 = (real_Double_t) fabs( MAGMA_C_REAL(LU->val[j]) );
            (*res) = (*res) + tmp2* tmp2;
        }
    }

    magma_c_mfree( LU );

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS; 
}



/**
    Purpose
    -------

    Computes an initial guess for the iterative ILU/IC


    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix
                sparse matrix in CSR

    @param
    B           magma_c_sparse_matrix*
                sparse matrix in CSR    


    @ingroup magmasparse_c
    ********************************************************************/

magma_int_t 
magma_cinitguess( magma_c_sparse_matrix A, magma_c_sparse_matrix *L, magma_c_sparse_matrix *U ){

    magma_c_sparse_matrix hAL, hAU, hAUT, hALCOO, hAUCOO, dAL, dAU, dALU, hALU, hD, dD, dL, hL;
    magma_int_t i,j;

    // need only lower triangular
    hAL.diagorder_type == Magma_VALUE;
    magma_c_mconvert( A, &hAL, Magma_CSR, Magma_CSRL );
    //magma_c_mconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );

    // need only upper triangular
    //magma_c_mconvert( A, &hAU, Magma_CSR, Magma_CSRU );
    magma_c_cucsrtranspose(  hAL, &hAU );
    //magma_c_mconvert( hAU, &hAUCOO, Magma_CSR, Magma_CSRCOO );

    magma_c_mtransfer( hAL, &dAL, Magma_CPU, Magma_DEV );
    magma_c_mtransfer( hAU, &dAU, Magma_CPU, Magma_DEV );
    magma_c_mfree( &hAL);
    magma_c_mfree( &hAU);

    magma_ccuspmm( dAL, dAU, &dALU );

    magma_c_mtransfer( dALU, &hALU, Magma_DEV, Magma_CPU );


    magma_c_mfree( &dAU);
    magma_c_mfree( &dALU);


    // generate diagonal matrix 
    magma_int_t offdiags = 0;
    magma_index_t *diag_offset;
    magmaFloatComplex *diag_vals;
    magma_cmalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_vals[0] = MAGMA_C_MAKE( 1.0, 0.0 );
    magma_cmgenerator( hALU.num_rows, offdiags, diag_offset, diag_vals, &hD );
    magma_c_mfree( &hALU);

    
    for(i=0; i<hALU.num_rows; i++){
        for(j=hALU.row[i]; j<hALU.row[i+1]; j++){
            if( hALU.col[j] == i ){
                //printf("%d %d  %d == %d -> %f   -->", i, j, hALU.col[j], i, hALU.val[j]);
                hD.val[i] = MAGMA_C_MAKE(
                        1.0 / sqrt(fabs(MAGMA_C_REAL(hALU.val[j])))  , 0.0 );
                //printf("insert %f at %d\n", hD.val[i], i);
            }
        }      
    }


    magma_c_mtransfer( hD, &dD, Magma_CPU, Magma_DEV );
    magma_c_mfree( &hD);

  //  magma_c_mvisu(dD);
    magma_ccuspmm( dD, dAL, &dL );
    //magma_c_mvisu(dD);
    //magma_c_mvisu(dAL);
    //magma_c_mvisu(dL);
    magma_c_mfree( &dAL);
    magma_c_mfree( &dD);



/*
    // check for diagonal = 1
    magma_c_sparse_matrix dLt, dLL, LL;
    magma_c_cucsrtranspose(  dL, &dLt );
    magma_ccuspmm( dL, dLt, &dLL );
    magma_c_mtransfer( dLL, &LL, Magma_DEV, Magma_CPU );
    for(i=0; i<100; i++){//hALU.num_rows; i++){
        for(j=hALU.row[i]; j<hALU.row[i+1]; j++){
            if( hALU.col[j] == i ){
                printf("%d %d -> %f   -->", i, i, LL.val[j]);
            }
        }      
    }
*/
    magma_c_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );


    magma_c_mconvert( hL, L, Magma_CSR, Magma_CSRCOO );

    magma_c_mfree( &dL);
    magma_c_mfree( &hL);

    return MAGMA_SUCCESS; 
}

