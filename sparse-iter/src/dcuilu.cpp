/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Hartwig Anzt 

       @generated from zcuilu.cpp normal z -> d, Fri Jul 18 17:34:29 2014
*/
// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// project includes
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>


#define PRECISION_d






/**
    Purpose
    -------

    Prepares the ILU preconditioner via the asynchronous ILU iteration.

    Arguments
    ---------

    @param
    A           magma_d_sparse_matrix
                input matrix A

    @param
    precond     magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_dgepr
    ********************************************************************/

magma_int_t
magma_dcuilusetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond ){

        // copy matrix into preconditioner parameter
        magma_d_sparse_matrix hA, hACSR;
        magma_d_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_d_mconvert( hA, &hACSR, hA.storage_type, Magma_CSR);
        magma_d_mtransfer(hACSR, &(precond->M), Magma_CPU, Magma_DEV);
        magma_d_mfree( &hA );
        magma_d_mfree( &hACSR );


            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");


            cusparseMatDescr_t descrA;
            cusparseStatus = cusparseCreateMatDescr(&descrA);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatDiagType (descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");

            cusparseStatus =
            cusparseCreateSolveAnalysisInfo( &(precond->cuinfo) );
             if(cusparseStatus != 0)    printf("error in info.\n");

            // end CUSPARSE context //

            cusparseStatus =
            cusparseDcsrsv_analysis( cusparseHandle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        precond->M.num_rows, precond->M.nnz, descrA,
                        precond->M.val, precond->M.row, precond->M.col, 
                        precond->cuinfo); 
             if(cusparseStatus != 0)    printf("error in analysis:%d\n", precond->cuinfo);

            cusparseStatus =
            cusparseDcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              precond->M.num_rows, descrA, 
                              precond->M.val, 
                              precond->M.row, 
                              precond->M.col, 
                              precond->cuinfo);
             if(cusparseStatus != 0)    printf("error in ILU:%d\n", precond->cuinfo);


            cusparseStatus =
            cusparseDestroySolveAnalysisInfo( precond->cuinfo );
             if(cusparseStatus != 0)    printf("error in info-free.\n");

    cusparseDestroyMatDescr( descrA );

    magma_d_sparse_matrix hL, hU;

    magma_d_mtransfer( precond->M, &hA, Magma_DEV, Magma_CPU );

    hL.diagorder_type = Magma_UNITY;
    magma_d_mconvert( hA, &hL , Magma_CSR, Magma_CSRL );
    hU.diagorder_type = Magma_VALUE;
    magma_d_mconvert( hA, &hU , Magma_CSR, Magma_CSRU );
    magma_d_mtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV );


    cusparseMatDescr_t descrL;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

    cusparseStatus =
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
     if(cusparseStatus != 0)    printf("error in MatrType.\n");

    cusparseStatus =
    cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_UNIT);
     if(cusparseStatus != 0)    printf("error in DiagType.\n");

    cusparseStatus =
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n");

    cusparseStatus =
    cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n");


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL); 
     if(cusparseStatus != 0)    printf("error in info.\n");

    cusparseStatus =
    cusparseDcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->L.num_rows, 
        precond->L.nnz, descrL, 
        precond->L.val, precond->L.row, precond->L.col, precond->cuinfoL );
     if(cusparseStatus != 0)    printf("error in analysis.\n");

    cusparseDestroyMatDescr( descrL );

    cusparseMatDescr_t descrU;
    cusparseStatus = cusparseCreateMatDescr(&descrU);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

    cusparseStatus =
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
     if(cusparseStatus != 0)    printf("error in MatrType.\n");

    cusparseStatus =
    cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
     if(cusparseStatus != 0)    printf("error in DiagType.\n");

    cusparseStatus =
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n");

    cusparseStatus =
    cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_UPPER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n");

    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoU); 
     if(cusparseStatus != 0)    printf("error in info.\n");

    cusparseStatus =
    cusparseDcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows, 
        precond->U.nnz, descrU, 
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU );
     if(cusparseStatus != 0)    printf("error in analysis.\n");

    cusparseDestroyMatDescr( descrU );

    magma_d_mfree(&hA);
    magma_d_mfree(&hL);
    magma_d_mfree(&hU);

    cusparseDestroy( cusparseHandle );

    return MAGMA_SUCCESS;

}








/**
    Purpose
    -------

    Performs the left triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param
    b           magma_d_vector
                RHS

    @param
    x           magma_d_vector*
                vector to precondition

    @param
    precond     magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_dgepr
    ********************************************************************/

magma_int_t
magma_dapplycuilu_l( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond ){

            double one = MAGMA_D_MAKE( 1.0, 0.0);




            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");


            cusparseMatDescr_t descrL;
            cusparseStatus = cusparseCreateMatDescr(&descrL);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");

            cusparseStatus =
            cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER);
             if(cusparseStatus != 0)    printf("error in fillmode.\n");

            // end CUSPARSE context //

            cusparseStatus =
            cusparseDcsrsv_solve(   cusparseHandle, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                    precond->L.num_rows, &one, 
                                    descrL,
                                    precond->L.val,
                                    precond->L.row,
                                    precond->L.col,
                                    precond->cuinfoL,
                                    b.val,
                                    x->val );
             if(cusparseStatus != 0)   printf("error in L triangular solve.\n");

    cusparseDestroyMatDescr( descrL );
    cusparseDestroy( cusparseHandle );
    magma_device_sync();
    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Performs the right triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param
    b           magma_d_vector
                RHS

    @param
    x           magma_d_vector*
                vector to precondition

    @param
    precond     magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_dgepr
    ********************************************************************/

magma_int_t
magma_dapplycuilu_r( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond ){

            double one = MAGMA_D_MAKE( 1.0, 0.0);

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");


            cusparseMatDescr_t descrU;
            cusparseStatus = cusparseCreateMatDescr(&descrU);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");

            cusparseStatus =
            cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_UPPER);
             if(cusparseStatus != 0)    printf("error in fillmode.\n");

            // end CUSPARSE context //

            cusparseStatus =
            cusparseDcsrsv_solve(   cusparseHandle, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                    precond->U.num_rows, &one, 
                                    descrU,
                                    precond->U.val,
                                    precond->U.row,
                                    precond->U.col,
                                    precond->cuinfoU,
                                    b.val,
                                    x->val );
             if(cusparseStatus != 0)   printf("error in L triangular solve.\n");

    cusparseDestroyMatDescr( descrU );
    cusparseDestroy( cusparseHandle );
    magma_device_sync();
    return MAGMA_SUCCESS;

}




/**
    Purpose
    -------

    Prepares the IC preconditioner via the asynchronous ILU iteration.

    Arguments
    ---------

    @param
    A           magma_d_sparse_matrix
                input matrix A

    @param
    precond     magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_dsypr
    ********************************************************************/

magma_int_t
magma_dcuiccsetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond ){

    magma_d_sparse_matrix hA, hACSR, U, hD, hR, hAt;
    magma_d_mtransfer( A, &hA, A.memory_location, Magma_CPU );
    U.diagorder_type = Magma_VALUE;
    magma_d_mconvert( hA, &hACSR, hA.storage_type, Magma_CSR);
    magma_d_mconvert( hACSR, &U, Magma_CSR, Magma_CSRL);
    magma_d_mfree( &hACSR );

    magma_d_mtransfer(U, &(precond->M), Magma_CPU, Magma_DEV);

    // CUSPARSE context //
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
     if(cusparseStatus != 0)    printf("error in Handle.\n");

    cusparseMatDescr_t descrA;
    cusparseStatus = cusparseCreateMatDescr(&descrA);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

    cusparseStatus =
    cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_SYMMETRIC);
     if(cusparseStatus != 0)    printf("error in MatrType.\n");

    cusparseStatus =
    cusparseSetMatDiagType (descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
     if(cusparseStatus != 0)    printf("error in DiagType.\n");

    cusparseStatus =
    cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n");

    cusparseStatus =
    cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_LOWER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n");


    cusparseStatus =
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfo) );
     if(cusparseStatus != 0)    printf("error in info.\n");

    // end CUSPARSE context //

    cusparseStatus =
    cusparseDcsrsv_analysis( cusparseHandle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE, 
                precond->M.num_rows, precond->M.nnz, descrA,
                precond->M.val, precond->M.row, precond->M.col, 
                precond->cuinfo); 
     if(cusparseStatus != 0)    printf("error in analysis IC.\n");

    cusparseStatus =
    cusparseDcsric0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                      precond->M.num_rows, descrA, 
                      precond->M.val, 
                      precond->M.row, 
                      precond->M.col, 
                      precond->cuinfo);

    cusparseStatus =
    cusparseDestroySolveAnalysisInfo( precond->cuinfo );
     if(cusparseStatus != 0)    printf("error in info-free.\n");

     if(cusparseStatus != 0)    printf("error in ICC.\n");

    cusparseMatDescr_t descrL;
    cusparseStatus = cusparseCreateMatDescr(&descrL);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

    cusparseStatus =
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
     if(cusparseStatus != 0)    printf("error in MatrType.\n");

    cusparseStatus =
    cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
     if(cusparseStatus != 0)    printf("error in DiagType.\n");

    cusparseStatus =
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n");

    cusparseStatus =
    cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n");


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL); 
     if(cusparseStatus != 0)    printf("error in info.\n");

    cusparseStatus =
    cusparseDcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrL, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoL );
     if(cusparseStatus != 0)    printf("error in analysis L.\n");

    cusparseDestroyMatDescr( descrL );

    cusparseMatDescr_t descrU;
    cusparseStatus = cusparseCreateMatDescr(&descrU);
     if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

    cusparseStatus =
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
     if(cusparseStatus != 0)    printf("error in MatrType.\n");

    cusparseStatus =
    cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
     if(cusparseStatus != 0)    printf("error in DiagType.\n");

    cusparseStatus =
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
     if(cusparseStatus != 0)    printf("error in IndexBase.\n");

    cusparseStatus =
    cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_LOWER);
     if(cusparseStatus != 0)    printf("error in fillmode.\n");

    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoU); 
     if(cusparseStatus != 0)    printf("error in info.\n");

    cusparseStatus =
    cusparseDcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrU, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoU );
     if(cusparseStatus != 0)    printf("error in analysis U.\n");

    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( cusparseHandle );

    magma_d_mfree(&U);
    magma_d_mfree(&hA);

/*
    // to enable also the block-asynchronous iteration for the triangular solves
    magma_d_mtransfer( precond->M, &hA, Magma_DEV, Magma_CPU );
    hA.storage_type = Magma_CSR;

    magma_dcsrsplit( 256, hA, &hD, &hR );

    magma_d_mtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hR, &precond->L, Magma_CPU, Magma_DEV );

    magma_d_mfree(&hD);
    magma_d_mfree(&hR);

    magma_d_cucsrtranspose(   hA, &hAt );

    magma_dcsrsplit( 256, hAt, &hD, &hR );

    magma_d_mtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hR, &precond->U, Magma_CPU, Magma_DEV );
    
    magma_d_mfree(&hD);
    magma_d_mfree(&hR);
    magma_d_mfree(&hA);
    magma_d_mfree(&hAt);
*/

    return MAGMA_SUCCESS;

}



/**
    Purpose
    -------

    Performs the left triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param
    b           magma_d_vector
                RHS

    @param
    x           magma_d_vector*
                vector to precondition

    @param
    precond     magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_dsypr
    ********************************************************************/

magma_int_t
magma_dapplycuicc_l( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond ){

            double one = MAGMA_D_MAKE( 1.0, 0.0);

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");


            cusparseMatDescr_t descrL;
            cusparseStatus = cusparseCreateMatDescr(&descrL);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatDiagType (descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");


            cusparseStatus =
            cusparseSetMatFillMode(descrL,CUSPARSE_FILL_MODE_LOWER);
             if(cusparseStatus != 0)    printf("error in fillmode.\n");

            // end CUSPARSE context //

            cusparseStatus =
            cusparseDcsrsv_solve(   cusparseHandle, 
                                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                    precond->M.num_rows, &one, 
                                    descrL,
                                    precond->M.val,
                                    precond->M.row,
                                    precond->M.col,
                                    precond->cuinfoL,
                                    b.val,
                                    x->val );
             if(cusparseStatus != 0)   printf("error in L triangular solve:%d.\n", precond->cuinfoL );


    cusparseDestroyMatDescr( descrL );
    cusparseDestroy( cusparseHandle );
    magma_device_sync();
    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Performs the right triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param
    b           magma_d_vector
                RHS

    @param
    x           magma_d_vector*
                vector to precondition

    @param
    precond     magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_dsypr
    ********************************************************************/

magma_int_t
magma_dapplycuicc_r( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond ){

            double one = MAGMA_D_MAKE( 1.0, 0.0);

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");


            cusparseMatDescr_t descrU;
            cusparseStatus = cusparseCreateMatDescr(&descrU);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_TRIANGULAR);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatDiagType (descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");


            cusparseStatus =
            cusparseSetMatFillMode(descrU,CUSPARSE_FILL_MODE_LOWER);
             if(cusparseStatus != 0)    printf("error in fillmode.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");


            // end CUSPARSE context //

            cusparseStatus =
            cusparseDcsrsv_solve(   cusparseHandle, 
                                    CUSPARSE_OPERATION_TRANSPOSE, 
                                    precond->M.num_rows, &one, 
                                    descrU,
                                    precond->M.val,
                                    precond->M.row,
                                    precond->M.col,
                                    precond->cuinfoU,
                                    b.val,
                                    x->val );
             if(cusparseStatus != 0)   printf("error in U triangular solve:%d.\n", precond->cuinfoU );

    cusparseDestroyMatDescr( descrU );
    cusparseDestroy( cusparseHandle );
    magma_device_sync();
    return MAGMA_SUCCESS;



}














