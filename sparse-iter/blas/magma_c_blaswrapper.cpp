/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_z_blaswrapper.cpp normal z -> c, Fri Jan 30 19:00:29 2015
       @author Hartwig Anzt

*/

//#include <cusparse_v2.h>

#include "common_magma.h"
#include "magmablas.h"
#include "magmasparse_types.h"
#include "magmasparse.h"




/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and scalars alpha, beta
    the wrapper determines the suitable SpMV computing
              y = alpha * A * x + beta * y.  
    Arguments
    ---------

    @param[in]
    alpha       magmaFloatComplex
                scalar alpha

    @param[in]
    A           magma_c_sparse_matrix
                sparse matrix A    

    @param[in]
    x           magma_c_vector
                input vector x  
                
    @param[in]
    beta        magmaFloatComplex
                scalar beta
    @param[out]
    y           magma_c_vector
                output vector y      
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_c
    ********************************************************************/

extern "C" magma_int_t
magma_c_spmv(
    magmaFloatComplex alpha, 
    magma_c_sparse_matrix A, 
    magma_c_vector x, 
    magmaFloatComplex beta, 
    magma_c_vector y,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( A.memory_location != x.memory_location || 
                            x.memory_location != y.memory_location ) {
        printf("error: linear algebra objects are not located in same memory!\n");
        printf("memory locations are: %d   %d   %d\n", 
                        A.memory_location, x.memory_location, y.memory_location );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_INVALID_PTR;
    }

    // DEV case
    if ( A.memory_location == Magma_DEV ) {
        if ( A.num_cols == x.num_rows && x.num_cols == 1 ) {

             if ( A.storage_type == Magma_CSR 
                            || A.storage_type == Magma_CSRL 
                            || A.storage_type == Magma_CSRU ) {
                 //printf("using CSR kernel for SpMV: ");
                 //magma_cgecsrmv( MagmaNoTrans, A.num_rows, A.num_cols, alpha, 
                 //                A.dval, A.drow, A.dcol, x.dval, beta, y.dval );
                 //printf("done.\n");

                cusparseHandle_t cusparseHandle = 0;
                cusparseStatus_t cusparseStatus;
                cusparseStatus = cusparseCreate(&cusparseHandle);
                cusparseSetStream( cusparseHandle, queue );
                cusparseMatDescr_t descr = 0;
                cusparseStatus = cusparseCreateMatDescr(&descr);

                cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
                cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

                cusparseCcsrmv( cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
                            A.num_rows, A.num_cols, A.nnz, &alpha, descr, 
                            A.dval, A.drow, A.dcol, x.dval, &beta, y.dval );

                cusparseDestroyMatDescr( descr );
                cusparseDestroy( cusparseHandle );

             }
             else if ( A.storage_type == Magma_ELL ) {
                 //printf("using ELLPACKT kernel for SpMV: ");
                 magma_cgeelltmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, beta, 
                    y.dval, queue );
                 //printf("done.\n");
             }
             else if ( A.storage_type == Magma_ELLPACKT ) {
                 //printf("using ELL kernel for SpMV: ");
                 magma_cgeellmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, beta, 
                    y.dval, queue );
                 //printf("done.\n");
             }
             else if ( A.storage_type == Magma_ELLRT ) {
                 //printf("using ELLRT kernel for SpMV: ");
                 magma_cgeellrtmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                            A.max_nnz_row, alpha, A.dval, A.dcol, A.drow, x.dval, 
                         beta, y.dval, A.alignment, A.blocksize, queue );
                 //printf("done.\n");
             }
             else if ( A.storage_type == Magma_SELLP ) {
                 //printf("using SELLP kernel for SpMV: ");
                 magma_cgesellpmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    A.blocksize, A.numblocks, A.alignment, 
                    alpha, A.dval, A.dcol, A.drow, x.dval, beta, y.dval, queue );

                 //printf("done.\n");
             }
             else if ( A.storage_type == Magma_DENSE ) {
                 //printf("using DENSE kernel for SpMV: ");
                 magmablas_cgemv( MagmaNoTrans, A.num_rows, A.num_cols, alpha, 
                                A.dval, A.num_rows, x.dval, 1, beta,  y.dval, 
                                1 );
                 //printf("done.\n");
             }
             else if ( A.storage_type == Magma_SPMVFUNCTION ) {
                 //printf("using DENSE kernel for SpMV: ");
                 magma_ccustomspmv( alpha, x, beta, y, queue );
                 //printf("done.\n");
             }
/*             else if ( A.storage_type == Magma_BCSR ) {
                 //printf("using CUSPARSE BCSR kernel for SpMV: ");
                // CUSPARSE context //
                cusparseHandle_t cusparseHandle = 0;
                cusparseStatus_t cusparseStatus;
                cusparseStatus = cusparseCreate(&cusparseHandle);
                cusparseSetStream( cusparseHandle, queue );
                cusparseMatDescr_t descr = 0;
                cusparseStatus = cusparseCreateMatDescr(&descr);
                // end CUSPARSE context //
                cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;
                int mb = (A.num_rows + A.blocksize-1)/A.blocksize;
                int nb = (A.num_cols + A.blocksize-1)/A.blocksize;
                cusparseCbsrmv( cusparseHandle, dirA, 
                    CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, A.numblocks, 
                    &alpha, descr, A.dval, A.drow, A.dcol, A.blocksize, x.dval, 
                    &beta, y.dval );
                 //printf("done.\n");
                 magmablasSetKernelStream( orig_queue );
                 return MAGMA_SUCCESS;
             }*/
             else {
                 printf("error: format not supported.\n");
                 magmablasSetKernelStream( orig_queue );
                 return MAGMA_ERR_NOT_SUPPORTED;
             }
        }
        else if ( A.num_cols < x.num_rows || x.num_cols > 1 ) {
            magma_int_t num_vecs = x.num_rows / A.num_cols * x.num_cols;
            if ( A.storage_type == Magma_CSR ) {

                cusparseHandle_t cusparseHandle = 0;
                cusparseStatus_t cusparseStatus;
                cusparseStatus = cusparseCreate(&cusparseHandle);
                cusparseSetStream( cusparseHandle, queue );
                cusparseMatDescr_t descr = 0;
                cusparseStatus = cusparseCreateMatDescr(&descr);

                cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
                cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

                if ( x.major == MagmaColMajor) {
                    cusparseCcsrmm(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    A.num_rows,   num_vecs, A.num_cols, A.nnz, 
                    &alpha, descr, A.dval, A.drow, A.dcol,
                    x.dval, A.num_cols, &beta, y.dval, A.num_cols);
                } else if ( x.major == MagmaRowMajor) {
                    cusparseCcsrmm2(cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    CUSPARSE_OPERATION_TRANSPOSE, 
                    A.num_rows,   num_vecs, A.num_cols, A.nnz, 
                    &alpha, descr, A.dval, A.drow, A.dcol,
                    x.dval, A.num_cols, &beta, y.dval, A.num_cols);
                }

                cusparseDestroyMatDescr( descr );
                cusparseDestroy( cusparseHandle );
             }
             else if ( A.storage_type == Magma_ELL ) {

                if ( x.major == MagmaColMajor) {
                 magma_cmgeelltmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                 num_vecs, A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, 
                 beta, y.dval, queue );
                }
                else if ( x.major == MagmaRowMajor) {
                    // transpose first to col major
                    magma_c_vector x2;                    
                    magma_cvtranspose( x, &x2, queue );
                    magma_cmgeellmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    num_vecs, A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, 
                    beta, y.dval, queue );
                    magma_c_vfree(&x2, queue );
                }
             }
             else if ( A.storage_type == Magma_ELLPACKT ) {
                if ( x.major == MagmaColMajor) {
                 magma_cmgeellmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                 num_vecs, A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, 
                 beta, y.dval, queue );
                }
                else if ( x.major == MagmaRowMajor) {
                    // transpose first to col major
                    magma_c_vector x2;                    
                    magma_cvtranspose( x, &x2, queue );
                    magma_cmgeelltmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    num_vecs, A.max_nnz_row, alpha, A.dval, A.dcol, x.dval, 
                    beta, y.dval, queue );
                    magma_c_vfree(&x2, queue );
                }
             } else if ( A.storage_type == Magma_SELLP ) {
                if ( x.major == MagmaRowMajor) {
                 magma_cmgesellpmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    num_vecs, A.blocksize, A.numblocks, A.alignment, 
                    alpha, A.dval, A.dcol, A.drow, x.dval, beta, y.dval, queue );
                }
                else if ( x.major == MagmaColMajor) {
                    // transpose first to row major
                    magma_c_vector x2; 
                    magma_cvtranspose( x, &x2, queue );
                    magma_cmgesellpmv( MagmaNoTrans, A.num_rows, A.num_cols, 
                    num_vecs, A.blocksize, A.numblocks, A.alignment, 
                    alpha, A.dval, A.dcol, A.drow, x2.dval, beta, y.dval, queue );
                    magma_c_vfree(&x2, queue );
                }
             }/*
             if ( A.storage_type == Magma_DENSE ) {
                 //printf("using DENSE kernel for SpMV: ");
                 magmablas_cmgemv( MagmaNoTrans, A.num_rows, A.num_cols, 
                            num_vecs, alpha, A.dval, A.num_rows, x.dval, 1, 
                            beta,  y.dval, 1 );
                 //printf("done.\n");
                 magmablasSetKernelStream( orig_queue );
                 return MAGMA_SUCCESS;
             }*/
             else {
                 printf("error: format not supported.\n");
                 magmablasSetKernelStream( orig_queue );
                 return MAGMA_ERR_NOT_SUPPORTED;
             }
        }
         
         
    }
    // CPU case missing!     
    else {
        printf("error: CPU not yet supported.\n");
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and scalars alpha, beta
    the wrapper determines the suitable SpMV computing
              y = alpha * ( A - lambda I ) * x + beta * y.  
    Arguments
    ---------

    @param
    alpha       magmaFloatComplex
                scalar alpha

    @param
    A           magma_c_sparse_matrix
                sparse matrix A   

    @param
    lambda      magmaFloatComplex
                scalar lambda 

    @param
    x           magma_c_vector
                input vector x  

    @param
    beta        magmaFloatComplex
                scalar beta   
                
    @param
    offset      magma_int_t 
                in case not the main diagonal is scaled
                
    @param
    blocksize   magma_int_t 
                in case of processing multiple vectors  
                
    @param
    add_rows    magma_int_t*
                in case the matrixpowerskernel is used
                
    @param
    y           magma_c_vector
                output vector y    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_spmv_shift(
    magmaFloatComplex alpha, 
    magma_c_sparse_matrix A, 
    magmaFloatComplex lambda,
    magma_c_vector x, 
    magmaFloatComplex beta, 
    magma_int_t offset, 
    magma_int_t blocksize, 
    magma_index_t *add_rows, 
    magma_c_vector y,
    magma_queue_t queue )
{
    if ( A.memory_location != x.memory_location 
                || x.memory_location != y.memory_location ) {
    printf("error: linear algebra objects are not located in same memory!\n");
    printf("memory locations are: %d   %d   %d\n", 
                    A.memory_location, x.memory_location, y.memory_location );
    return MAGMA_ERR_INVALID_PTR;
    }
    // DEV case
    if ( A.memory_location == Magma_DEV ) {
         if ( A.storage_type == Magma_CSR ) {
             //printf("using CSR kernel for SpMV: ");
             magma_cgecsrmv_shift( MagmaNoTrans, A.num_rows, A.num_cols, 
                alpha, lambda, A.dval, A.drow, A.dcol, x.dval, beta, offset, 
                blocksize, add_rows, y.dval, queue );
             //printf("done.\n");
         }
         else if ( A.storage_type == Magma_ELLPACKT ) {
             //printf("using ELLPACKT kernel for SpMV: ");
             magma_cgeellmv_shift( MagmaNoTrans, A.num_rows, A.num_cols, 
                A.max_nnz_row, alpha, lambda, A.dval, A.dcol, x.dval, beta, offset, 
                blocksize, add_rows, y.dval, queue );
             //printf("done.\n");
         }
         else if ( A.storage_type == Magma_ELL ) {
             //printf("using ELL kernel for SpMV: ");
             magma_cgeelltmv_shift( MagmaNoTrans, A.num_rows, A.num_cols, 
                A.max_nnz_row, alpha, lambda, A.dval, A.dcol, x.dval, beta, offset, 
                blocksize, add_rows, y.dval, queue );
             //printf("done.\n");
         }
         else {
             printf("error: format not supported.\n");
             return MAGMA_ERR_NOT_SUPPORTED;
         }
         return MAGMA_SUCCESS;
    }
    // CPU case missing!     
    else {
        printf("error: CPU not yet supported.\n");
        return MAGMA_ERR_NOT_SUPPORTED;
    }
}
