/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @author Hartwig Anzt 

       @generated from zbailu.cpp normal z -> d, Fri May 30 10:41:42 2014
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


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the ILU preconditioner via the asynchronous ILU iteration.

    Arguments
    =========

    magma_d_sparse_matrix A                   input matrix A
    magma_d_preconditioner *precond           preconditioner parameters

    ========================================================================  */

magma_int_t
magma_dailusetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond ){

    magma_d_sparse_matrix hA, hAL, hALCOO, hAU, hAUT, hAUCOO, dAL, dAU, hL, hU, 
                                        dL, dU, DL, RL, DU, RU;

    // copy original matrix as CSRCOO to device
    magma_d_mtransfer(A, &hA, A.memory_location, Magma_CPU);

    // in case using fill-in
    magma_dilustruct( &hA, precond->levels);

    // need only lower triangular
    hAL.diagorder_type == Magma_UNITY;
    magma_d_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL );
    magma_d_mconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );
    magma_d_mtransfer( hALCOO, &dAL, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hALCOO, &dAU, Magma_CPU, Magma_DEV );

    // need only upper triangular
    magma_d_mconvert( hA, &hAU, Magma_CSR, Magma_CSRU );
    magma_d_cucsrtranspose(  hAU, &hAUT );
    magma_d_mconvert( hAUT, &hAUCOO, Magma_CSR, Magma_CSRCOO );
    magma_d_mtransfer( hAUCOO, &dL, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hAUCOO, &dU, Magma_CPU, Magma_DEV );

    magma_d_mfree(&hALCOO);
    magma_d_mfree(&hAL);
    magma_d_mfree(&hAUCOO);
    magma_d_mfree(&hAUT);
    magma_d_mfree(&hAU);

    for(int i=0; i<20; i++){
        magma_dailu_csr_s( dAL, dAU, dL, dU );

    }

    magma_d_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );
    magma_d_mtransfer( dU, &hU, Magma_DEV, Magma_CPU );

    magma_d_LUmergein( hL, hU, &hA);

    magma_d_mtransfer( hA, &precond->M, Magma_CPU, Magma_DEV );

    magma_d_mfree(&dL);
    magma_d_mfree(&dU);
    magma_d_mfree(&dAL);
    magma_d_mfree(&dAU);

    hAL.diagorder_type = Magma_UNITY;
    magma_d_mconvert(hA, &hAL, Magma_CSR, Magma_CSRL);
    hAL.storage_type = Magma_CSR;
    magma_d_mconvert(hA, &hAU, Magma_CSR, Magma_CSRU);
    hAU.storage_type = Magma_CSR;
    magma_d_mfree(&hA);

    magma_d_mfree(&hL);
    magma_d_mfree(&hU);

    magma_dcsrsplit( 256, hAL, &DL, &RL );
    magma_dcsrsplit( 256, hAU, &DU, &RU );

    magma_d_mtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( DU, &precond->UD, Magma_CPU, Magma_DEV );

    // for cusparse uncomment this
    magma_d_mtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV );

    // for ba-solve uncomment this
/*
    if( RL.nnz != 0 )
        magma_d_mtransfer( RL, &precond->L, Magma_CPU, Magma_DEV );
    else{ 
        precond->L.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    if( RU.nnz != 0 )
        magma_d_mtransfer( RU, &precond->U, Magma_CPU, Magma_DEV );
    else{ 
        precond->U.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }
*/
    magma_d_mfree(&hAL);
    magma_d_mfree(&hAU);
    magma_d_mfree(&DL);
    magma_d_mfree(&RL);
    magma_d_mfree(&DU);
    magma_d_mfree(&RU);

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
    cusparseDestroy( cusparseHandle );

    return MAGMA_SUCCESS;

}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Performs the left triangular solves using the ILU preconditioner.

    Arguments
    =========

    magma_d_vector b                        RHS
    magma_d_vector *x                       vector to precondition
    magma_d_preconditioner *precond         preconditioner parameters

    ========================================================================  */

magma_int_t
magma_dapplyailu_l( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond ){

    magma_int_t iters = 1;
    for(int k=0; k<40; k++)
        magma_dbajac_csr( iters, precond->LD, precond->L, b, x );
           
    return MAGMA_SUCCESS;

}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Performs the right triangular solves using the ILU preconditioner.

    Arguments
    =========

    magma_d_vector b                        RHS
    magma_d_vector *x                       vector to precondition
    magma_d_preconditioner *precond         preconditioner parameters

    ========================================================================  */

magma_int_t
magma_dapplyailu_r( magma_d_vector b, magma_d_vector *x, 
                    magma_d_preconditioner *precond ){

    magma_int_t iters = 1;
    for(int k=0; k<40; k++)
        magma_dbajac_csr( iters, precond->UD, precond->U, b, x );

    return MAGMA_SUCCESS;

}





/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prepares the IC preconditioner via the asynchronous IC iteration.

    Arguments
    =========

    magma_d_sparse_matrix A                   input matrix A
    magma_d_preconditioner *precond           preconditioner parameters

    ========================================================================  */

magma_int_t
magma_daiccsetup( magma_d_sparse_matrix A, magma_d_preconditioner *precond ){


    magma_d_sparse_matrix hA, hAL, hALCOO, dAL, hL, dL, DL, RL;



    // copy original matrix as CSRCOO to device
    magma_d_mtransfer(A, &hA, A.memory_location, Magma_CPU);

    // in case using fill-in
    magma_dilustruct( &hA, precond->levels);

    magma_d_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL );
    magma_d_mconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );

    magma_d_mtransfer( hALCOO, &dAL, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( hALCOO, &dL, Magma_CPU, Magma_DEV );
    magma_d_mfree(&hALCOO);
    magma_d_mfree(&hAL);
    magma_d_mfree(&hA);

    for(int i=0; i<25; i++){
        magma_daic_csr_s( dAL, dL );

    }
    magma_d_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );

    magma_d_mfree(&dL);
    magma_d_mfree(&dAL);

    magma_d_mconvert(hL, &hAL, hL.storage_type, Magma_CSR);

    magma_d_mtransfer( hAL, &precond->M, Magma_CPU, Magma_DEV );

    magma_dcsrsplit( 256, hAL, &DL, &RL );

    magma_d_mtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( RL, &precond->L, Magma_CPU, Magma_DEV );

    magma_d_mfree(&hL);

    magma_d_cucsrtranspose(   hAL, &hL );

    magma_dcsrsplit( 256, hL, &DL, &RL );

    magma_d_mtransfer( DL, &precond->UD, Magma_CPU, Magma_DEV );
    magma_d_mtransfer( RL, &precond->U, Magma_CPU, Magma_DEV );

    magma_d_mfree(&hAL);
    magma_d_mfree(&hL);

    magma_d_mfree(&DL);
    magma_d_mfree(&RL);


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


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL); 
     if(cusparseStatus != 0)    printf("error in info.\n");

    cusparseStatus =
    cusparseDcsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows, 
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
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrU, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoU );
     if(cusparseStatus != 0)    printf("error in analysis U.\n");

    cusparseDestroyMatDescr( descrU );
    cusparseDestroy( cusparseHandle );

    return MAGMA_SUCCESS;

}
