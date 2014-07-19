/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Hartwig Anzt 

       @generated from zbailu.cpp normal z -> s, Fri Jul 18 17:34:29 2014
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


#define PRECISION_s


/**
    Purpose
    -------

    Prepares the ILU preconditioner via the asynchronous ILU iteration.

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    precond     magma_s_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_sgepr
    ********************************************************************/

magma_int_t
magma_ialusetup( magma_s_sparse_matrix A, magma_s_preconditioner *precond ){

    magma_s_sparse_matrix hAh, hA, hAL, hALCOO, hAU, hAUT, hAUCOO, dAL, dAU, 
                                        hL, hU, dL, dU, DL, RL, DU, RU;

    // copy original matrix as CSRCOO to device
    magma_s_mtransfer(A, &hAh, A.memory_location, Magma_CPU);
    magma_s_mconvert( hAh, &hA, hAh.storage_type, Magma_CSR );
    magma_s_mfree(&hAh);

    // in case using fill-in
    magma_silustruct( &hA, precond->levels);

    // need only lower triangular
    hAL.diagorder_type == Magma_UNITY;
    magma_s_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL );
    magma_s_mconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );
    magma_s_mtransfer( hALCOO, &dAL, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( hALCOO, &dAU, Magma_CPU, Magma_DEV );

    // need only upper triangular
    magma_s_mconvert( hA, &hAU, Magma_CSR, Magma_CSRU );
    magma_s_cucsrtranspose(  hAU, &hAUT );
    magma_s_mconvert( hAUT, &hAUCOO, Magma_CSR, Magma_CSRCOO );
    magma_s_mtransfer( hAUCOO, &dL, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( hAUCOO, &dU, Magma_CPU, Magma_DEV );

    magma_s_mfree(&hALCOO);
    magma_s_mfree(&hAL);
    magma_s_mfree(&hAUCOO);
    magma_s_mfree(&hAUT);
    magma_s_mfree(&hAU);

    for(int i=0; i<precond->sweeps; i++){
        magma_ialu_csr_s( dAL, dAU, dL, dU );

    }

    magma_s_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );
    magma_s_mtransfer( dU, &hU, Magma_DEV, Magma_CPU );

    magma_s_LUmergein( hL, hU, &hA);

    magma_s_mtransfer( hA, &precond->M, Magma_CPU, Magma_DEV );

    magma_s_mfree(&dL);
    magma_s_mfree(&dU);
    magma_s_mfree(&dAL);
    magma_s_mfree(&dAU);

    hAL.diagorder_type = Magma_UNITY;
    magma_s_mconvert(hA, &hAL, Magma_CSR, Magma_CSRL);
    hAL.storage_type = Magma_CSR;
    magma_s_mconvert(hA, &hAU, Magma_CSR, Magma_CSRU);
    hAU.storage_type = Magma_CSR;
    magma_s_mfree(&hA);

    magma_s_mfree(&hL);
    magma_s_mfree(&hU);

    magma_scsrsplit( 256, hAL, &DL, &RL );
    magma_scsrsplit( 256, hAU, &DU, &RU );

    magma_s_mtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( DU, &precond->UD, Magma_CPU, Magma_DEV );

    // for cusparse uncomment this
    magma_s_mtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV );

    // for ba-solve uncomment this
/*
    if( RL.nnz != 0 )
        magma_s_mtransfer( RL, &precond->L, Magma_CPU, Magma_DEV );
    else{ 
        precond->L.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    if( RU.nnz != 0 )
        magma_s_mtransfer( RU, &precond->U, Magma_CPU, Magma_DEV );
    else{ 
        precond->U.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }
*/
    magma_s_mfree(&hAL);
    magma_s_mfree(&hAU);
    magma_s_mfree(&DL);
    magma_s_mfree(&RL);
    magma_s_mfree(&DU);
    magma_s_mfree(&RU);

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
    cusparseScsrsv_analysis(cusparseHandle, 
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
    cusparseScsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows, 
        precond->U.nnz, descrU, 
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU );
     if(cusparseStatus != 0)    printf("error in analysis.\n");

    cusparseDestroyMatDescr( descrU );
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
    b           magma_s_vector
                RHS

    @param
    x           magma_s_vector*
                vector to precondition

    @param
    precond     magma_s_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_sgepr
    ********************************************************************/

magma_int_t
magma_sapplyailu_l( magma_s_vector b, magma_s_vector *x, 
                    magma_s_preconditioner *precond ){

    magma_int_t iters = 1;
    for(int k=0; k<40; k++)
        magma_sbajac_csr( iters, precond->LD, precond->L, b, x );
           
    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Performs the right triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param
    b           magma_s_vector
                RHS

    @param
    x           magma_s_vector*
                vector to precondition

    @param
    precond     magma_s_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_sgepr
    ********************************************************************/

magma_int_t
magma_sapplyailu_r( magma_s_vector b, magma_s_vector *x, 
                    magma_s_preconditioner *precond ){

    magma_int_t iters = 1;
    for(int k=0; k<40; k++)
        magma_sbajac_csr( iters, precond->UD, precond->U, b, x );

    return MAGMA_SUCCESS;

}





/**
    Purpose
    -------

    Prepares the IC preconditioner via the asynchronous IC iteration.

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    precond     magma_s_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_ssypr
    ********************************************************************/

magma_int_t
magma_saiccsetup( magma_s_sparse_matrix A, magma_s_preconditioner *precond ){


    magma_s_sparse_matrix hAh, hA, hAL, hALCOO, dAL, hL, dL, DL, RL;



    // copy original matrix as CSRCOO to device
    magma_s_mtransfer(A, &hAh, A.memory_location, Magma_CPU);
    magma_s_mconvert( hAh, &hA, hAh.storage_type, Magma_CSR );
    magma_s_mfree(&hAh);

    // in case using fill-in
    magma_silustruct( &hA, precond->levels);

    magma_s_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL );
    magma_s_mconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );

    magma_s_mtransfer( hALCOO, &dAL, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( hALCOO, &dL, Magma_CPU, Magma_DEV );
    magma_s_mfree(&hALCOO);
    magma_s_mfree(&hAL);
    magma_s_mfree(&hA);

    for(int i=0; i<precond->sweeps; i++){
        magma_saic_csr_s( dAL, dL );

    }
    magma_s_mtransfer( dL, &hL, Magma_DEV, Magma_CPU );

    magma_s_mfree(&dL);
    magma_s_mfree(&dAL);

    magma_s_mconvert(hL, &hAL, hL.storage_type, Magma_CSR);

    // for CUSPARSE
    magma_s_mtransfer( hAL, &precond->M, Magma_CPU, Magma_DEV );

    magma_scsrsplit( 256, hAL, &DL, &RL );

    magma_s_mtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( RL, &precond->L, Magma_CPU, Magma_DEV );

    magma_s_mfree(&hL);

    magma_s_cucsrtranspose(   hAL, &hL );

    magma_scsrsplit( 256, hL, &DL, &RL );

    magma_s_mtransfer( DL, &precond->UD, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( RL, &precond->U, Magma_CPU, Magma_DEV );

    magma_s_mfree(&hAL);
    magma_s_mfree(&hL);

    magma_s_mfree(&DL);
    magma_s_mfree(&RL);


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


    cusparseStatus = cusparseCreateSolveAnalysisInfo(&precond->cuinfoL); 
     if(cusparseStatus != 0)    printf("error in info.\n");

    cusparseStatus =
    cusparseScsrsv_analysis(cusparseHandle, 
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
    cusparseScsrsv_analysis(cusparseHandle, 
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows, 
        precond->M.nnz, descrU, 
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoU );
     if(cusparseStatus != 0)    printf("error in analysis U.\n");

    cusparseDestroyMatDescr( descrU );
    cusparseDestroy( cusparseHandle );

    return MAGMA_SUCCESS;

}
