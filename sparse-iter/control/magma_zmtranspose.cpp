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

/**
    Purpose
    -------

    Transposes a matrix stored in CSR format.


    Arguments
    ---------

    @param[in]
    n_rows      magma_int_t
                number of rows in input matrix

    @param[in]
    n_cols      magma_int_t
                number of columns in input matrix

    @param[in]
    nnz         magma_int_t
                number of nonzeros in input matrix

    @param[in]
    val         magmaDoubleComplex*
                value array of input matrix 

    @param[in]
    row         magma_index_t*
                row pointer of input matrix

    @param[in]
    col         magma_index_t*
                column indices of input matrix 

    @param[in]
    new_n_rows  magma_index_t*
                number of rows in transposed matrix

    @param[in]
    new_n_cols  magma_index_t*
                number of columns in transposed matrix

    @param[in]
    new_nnz     magma_index_t*
                number of nonzeros in transposed matrix

    @param[in]
    new_val     magmaDoubleComplex**
                value array of transposed matrix 

    @param[in]
    new_row     magma_index_t**
                row pointer of transposed matrix

    @param[in]
    new_col     magma_index_t**
                column indices of transposed matrix

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t 
z_transpose_csr(    
    magma_int_t n_rows, 
    magma_int_t n_cols, 
    magma_int_t nnz, 
    magmaDoubleComplex *val, 
    magma_index_t *row, 
    magma_index_t *col, 
    magma_int_t *new_n_rows, 
    magma_int_t *new_n_cols, 
    magma_int_t *new_nnz, 
    magmaDoubleComplex **new_val, 
    magma_index_t **new_row, 
    magma_index_t **new_col,
    magma_queue_t queue )
{



  nnz = row[n_rows];
  *new_n_rows = n_cols;
  *new_n_cols = n_rows;
  *new_nnz = nnz;

  magmaDoubleComplex ** valtemp;
  magma_index_t ** coltemp;
  valtemp = (magmaDoubleComplex**)malloc((n_rows)*sizeof(magmaDoubleComplex*));
  coltemp =(magma_index_t**)malloc((n_rows)*sizeof(magma_index_t*));

  // temporary 2-dimensional arrays valtemp/coltemp 
  // where val[i] is the array with the values of the i-th column of the matrix
  magma_index_t *nnztemp;
  magma_index_malloc_cpu( &nnztemp, n_rows );
  
  for( magma_int_t i=0; i<n_rows; i++ )
    nnztemp[i]=0;
  for( magma_int_t i=0; i<nnz; i++ )
    nnztemp[col[i]]++;    

  for( magma_int_t i=0; i<n_rows; i++ ){
    valtemp[i] = 
        (magmaDoubleComplex*)malloc((nnztemp[i])*sizeof(magmaDoubleComplex));
    coltemp[i] = (magma_index_t*)malloc(nnztemp[i]*sizeof(magma_index_t));
  }

  for( magma_int_t i=0; i<n_rows; i++ )
    nnztemp[i]=0;

  for( magma_int_t j=0; j<n_rows; j++ ){
    for( magma_int_t i=row[j]; i<row[j+1]; i++ ){
      valtemp[col[i]][nnztemp[col[i]]]=val[i];
      coltemp[col[i]][nnztemp[col[i]]]=j;
      nnztemp[col[i]]++;    
    }
  }

  //csr structure for transposed matrix
  *new_val = new magmaDoubleComplex[nnz];
  *new_row = new magma_index_t[n_rows+1];
  *new_col = new magma_index_t[nnz];

  //fill the transposed csr structure
  magma_int_t nnztmp=0;
  (*new_row)[0]=0;
  for( magma_int_t j=0; j<n_rows; j++ ){
    for( magma_int_t i=0; i<nnztemp[j]; i++ ){
      (*new_val)[nnztmp]=valtemp[j][i];
      (*new_col)[nnztmp]=coltemp[j][i];
      nnztmp++;
    }
    (*new_row)[j+1]=nnztmp;
  }

//usually the temporary memory should be freed afterwards
//however, it does not work
/*
  for( magma_int_t j=0; j<n_rows; j++ ){
    free(valtemp[j]);
    free(coltemp[j]);
  }
  free(valtemp);free(coltemp);
      printf("check9\n");
    fflush(stdout);
*/

    magma_free_cpu( nnztemp );
    
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Interface to cuSPARSE transpose.

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                input matrix (CSR)

    @param[out]
    B           magma_z_sparse_matrix*
                output matrix (CSR)
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/
    
    
extern "C" magma_int_t
magma_z_mtranspose(
    magma_z_sparse_matrix A, magma_z_sparse_matrix *B,
    magma_queue_t queue )
{
    
    magma_z_cucsrtranspose( A, B, queue );
    return MAGMA_SUCCESS;

}


/**
    Purpose
    -------

    Helper function to transpose CSR matrix. 
    Using the CUSPARSE CSR2CSC function.


    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                input matrix (CSR)

    @param[out]
    B           magma_z_sparse_matrix*
                output matrix (CSR)
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_cucsrtranspose(
    magma_z_sparse_matrix A, 
    magma_z_sparse_matrix *B,
    magma_queue_t queue )
{
    // for symmetric matrices: convert to csc using cusparse

    if( A.storage_type == Magma_CSR && A.memory_location == Magma_DEV ) {
                  
         magma_z_sparse_matrix C;
         magma_z_mtransfer( A, &C, Magma_DEV, Magma_DEV, queue );
        // CUSPARSE context //
        cusparseHandle_t handle;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&handle);
        cusparseSetStream( handle, queue );
         if (cusparseStatus != 0)    printf("error in Handle.\n");


        cusparseMatDescr_t descrA;
        cusparseMatDescr_t descrB;
        cusparseStatus = cusparseCreateMatDescr(&descrA);
        cusparseStatus = cusparseCreateMatDescr(&descrB);
         if (cusparseStatus != 0)    printf("error in MatrDescr.\n");

        cusparseStatus =
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
         if (cusparseStatus != 0)    printf("error in MatrType.\n");

        cusparseStatus =
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);
         if (cusparseStatus != 0)    printf("error in IndexBase.\n");

        cusparseStatus = 
        cusparseZcsr2csc( handle, A.num_rows, A.num_rows, A.nnz,
                         A.dval, A.drow, A.dcol, C.dval, C.dcol, C.drow,
                         CUSPARSE_ACTION_NUMERIC, 
                         CUSPARSE_INDEX_BASE_ZERO);
         if (cusparseStatus != 0)    
                printf("error in transpose: %d.\n", cusparseStatus);

        cusparseDestroyMatDescr( descrA );
        cusparseDestroyMatDescr( descrB );
        cusparseDestroy( handle );
        
        magma_z_mtransfer( C, B, Magma_DEV, Magma_DEV, queue );   
        
        if( A.fill_mode == Magma_FULL ){
             B->fill_mode = Magma_FULL;
        }
        else if( A.fill_mode == Magma_LOWER ){
             B->fill_mode = Magma_UPPER;
        }
        else if ( A.fill_mode == Magma_UPPER ){
             B->fill_mode = Magma_LOWER;
        }

        // end CUSPARSE context //

        return MAGMA_SUCCESS;
        
    }else if( A.storage_type == Magma_CSR && A.memory_location == Magma_CPU ){
               
        magma_z_sparse_matrix A_d, B_d;

        magma_z_mtransfer( A, &A_d, A.memory_location, Magma_DEV, queue );
        magma_z_cucsrtranspose( A_d, &B_d, queue );
        magma_z_mtransfer( B_d, B, Magma_DEV, A.memory_location, queue );
        
        magma_z_mfree( &A_d, queue );
        magma_z_mfree( &B_d, queue );
        
        return MAGMA_SUCCESS;
                
    }else {

        magma_z_sparse_matrix ACSR, BCSR;
        
        magma_z_mconvert( A, &ACSR, A.storage_type, Magma_CSR, queue );
        magma_z_cucsrtranspose( ACSR, &BCSR, queue );
        magma_z_mconvert( BCSR, B, Magma_CSR, A.storage_type, queue );
       
        magma_z_mfree( &ACSR, queue );
        magma_z_mfree( &BCSR, queue );

        return MAGMA_SUCCESS;
    }
}



