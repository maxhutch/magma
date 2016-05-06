/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zcustomilu.cpp normal z -> c, Mon May  2 23:31:02 2016
*/
#include "magmasparse_internal.h"

#define COMPLEX


/**
    Purpose
    -------

    Reads in an Incomplete LU preconditioner.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A
                
    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_ccustomilusetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    
    magma_c_matrix hA={Magma_CSR};
    char preconditionermatrix[255];
    
    // first L
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "precondL.mtx" );
    
    CHECK( magma_c_csr_mtx( &hA, preconditionermatrix , queue) );
    CHECK( magma_cmtransfer( hA, &precond->L, Magma_CPU, Magma_DEV , queue ));
    // extract the diagonal of L into precond->d
    CHECK( magma_cjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_cvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_C_ZERO, queue ));

    magma_cmfree( &hA, queue );
    
    // now U
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "precondU.mtx" );

    CHECK( magma_c_csr_mtx( &hA, preconditionermatrix , queue) );
    CHECK( magma_cmtransfer( hA, &precond->U, Magma_CPU, Magma_DEV , queue ));
    // extract the diagonal of U into precond->d2
    CHECK( magma_cjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_cvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_C_ZERO, queue ));


    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrL ));
    CHECK_CUSPARSE( cusparseSetMatType( descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrL, CUSPARSE_DIAG_TYPE_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrL, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrL, CUSPARSE_FILL_MODE_LOWER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoL ));
    CHECK_CUSPARSE( cusparseCcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->L.num_rows,
        precond->L.nnz, descrL,
        precond->L.val, precond->L.row, precond->L.col, precond->cuinfoL ));

    
    
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrU ));
    CHECK_CUSPARSE( cusparseSetMatType( descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrU, CUSPARSE_FILL_MODE_UPPER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoU ));
    CHECK_CUSPARSE( cusparseCcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows,
        precond->U.nnz, descrU,
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU ));

    
    cleanup:
        
    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;    
    magma_cmfree( &hA, queue );
    
    return info;
}
    
