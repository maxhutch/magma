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



/**
    Purpose
    -------

    Transposes a matrix stored in CSR format.


    Arguments
    ---------

    @param
    n_rows      magma_int_t
                number of rows in input matrix

    @param
    n_cols      magma_int_t
                number of columns in input matrix

    @param
    nnz         magma_int_t
                number of nonzeros in input matrix

    @param
    val         magmaDoubleComplex*
                value array of input matrix 

    @param
    row         magma_index_t*
                row pointer of input matrix

    @param
    col         magma_index_t*
                column indices of input matrix 

    @param
    new_n_rows  magma_index_t*
                number of rows in transposed matrix

    @param
    new_n_cols  magma_index_t*
                number of columns in transposed matrix

    @param
    new_nnz     magma_index_t*
                number of nonzeros in transposed matrix

    @param
    new_val     magmaDoubleComplex**
                value array of transposed matrix 

    @param
    new_row     magma_index_t**
                row pointer of transposed matrix

    @param
    new_col     magma_index_t**
                column indices of transposed matrix


    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t z_transpose_csr(    magma_int_t n_rows, 
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
                                magma_index_t **new_col ){



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

magma_int_t 
magma_z_mtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B ){

        magma_z_cucsrtranspose( A, B );
/*
    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_CSR ){
            z_transpose_csr( A.num_rows, A.num_cols, A.nnz, A.val, A.row, A.col, 
                                &(B->num_rows), &(B->num_cols), &(B->nnz), 
                                &(B->val), &(B->row), &(B->col) );
            B->memory_location = Magma_CPU;
            B->storage_type = Magma_CSR;
        }
        else{
            magma_z_sparse_matrix C, D;
            magma_z_mconvert( A, &C, A.storage_type, Magma_CSR);
            magma_z_mtranspose( C, &D );
            magma_z_mconvert( D, B, Magma_CSR, A.storage_type );
            magma_z_mfree(&C);
            magma_z_mfree(&D);

        }
        return MAGMA_SUCCESS;   
    }
    else{
        magma_z_sparse_matrix C, D;
        magma_z_mtransfer( A, &C, A.memory_location, Magma_CPU);
        magma_z_mtranspose( C, &D );
        magma_z_mtransfer( D, B, Magma_CPU, A.memory_location );
        magma_z_mfree(&C);
        magma_z_mfree(&D);
        return MAGMA_SUCCESS;
    }

*/
    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Helper function to transpose CSR matrix.


    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix (CSR)

    @param
    B           magma_z_sparse_matrix*
                output matrix (CSR)

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t 
magma_z_csrtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B ){

    if( A.storage_type != Magma_CSR ){
        magma_z_sparse_matrix ACSR, BCSR;
        magma_z_mconvert( A, &ACSR, A.storage_type, Magma_CSR );
        magma_z_cucsrtranspose( ACSR, &BCSR );
        magma_z_mconvert( BCSR, B, Magma_CSR, A.storage_type );
        
        return MAGMA_SUCCESS;
    }
    else{

        magma_int_t i, j, new_nnz=0, lrow;

        // fill in information for B
        B->storage_type = Magma_CSR;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        B->max_nnz_row = A.max_nnz_row;
        B->diameter = A.diameter;

        magma_zmalloc_cpu( &B->val, A.nnz );
        magma_index_malloc_cpu( &B->row, A.num_rows+1 );
        magma_index_malloc_cpu( &B->col, A.nnz );

        for( magma_int_t i=0; i<A.nnz; i++){
            B->val[i] = A.val[i];
            B->col[i] = A.col[i];
        }
        for( magma_int_t i=0; i<A.num_rows+1; i++){
            B->row[i] = A.row[i];
        }
        for( lrow = 0; lrow < A.num_rows; lrow++ ){
            B->row[lrow] = new_nnz;
            for( i=0; i<A.num_rows; i++ ){
                for( j=A.row[i]; j<A.row[i+1]; j++ ){
                    if( A.col[j] == lrow ){
                        B->val[ new_nnz ] = A.val[ j ];
                        B->col[ new_nnz ] = i;
                        new_nnz++; 
                    }
                }
            }
        }
        B->row[ B->num_rows ] = new_nnz;

        return MAGMA_SUCCESS;

    }
    return MAGMA_SUCCESS; 
}


/**
    Purpose
    -------

    Helper function to transpose CSR matrix. 
    Using the CUSPARSE CSR2CSC function.


    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix (CSR)

    @param
    B           magma_z_sparse_matrix*
                output matrix (CSR)

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t 
magma_z_cucsrtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B ){
// for symmetric matrices: convert to csc using cusparse


    if( A.storage_type != Magma_CSR && A.storage_type != Magma_CSRCOO ){
        magma_z_sparse_matrix ACSR, BCSR;
        magma_z_mconvert( A, &ACSR, A.storage_type, Magma_CSR );
        magma_z_cucsrtranspose( ACSR, &BCSR );

        if( A.storage_type == Magma_CSRL )
            B->storage_type = Magma_CSRU;
        else if( A.storage_type == Magma_CSRU )
            B->storage_type = Magma_CSRL;
        else
            B->storage_type = A.storage_type;

        magma_z_mconvert( BCSR, B, Magma_CSR, B->storage_type );
        magma_z_mfree( &ACSR );
        magma_z_mfree( &BCSR );
        return MAGMA_SUCCESS;
    }
    else{

        magma_z_sparse_matrix A_d, B_d;

        magma_z_mtransfer( A, &A_d, A.memory_location, Magma_DEV );

        magma_z_mtransfer( A, &B_d, A.memory_location, Magma_DEV );


        // CUSPARSE context //
        cusparseHandle_t handle;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&handle);
         if(cusparseStatus != 0)    printf("error in Handle.\n");


        cusparseMatDescr_t descrA;
        cusparseMatDescr_t descrB;
        cusparseStatus = cusparseCreateMatDescr(&descrA);
        cusparseStatus = cusparseCreateMatDescr(&descrB);
         if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

        cusparseStatus =
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatType(descrB,CUSPARSE_MATRIX_TYPE_GENERAL);
         if(cusparseStatus != 0)    printf("error in MatrType.\n");

        cusparseStatus =
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatIndexBase(descrB,CUSPARSE_INDEX_BASE_ZERO);
         if(cusparseStatus != 0)    printf("error in IndexBase.\n");

        cusparseStatus = 
        cusparseZcsr2csc( handle, A.num_rows, A.num_rows, A.nnz,
                         A_d.val, A_d.row, A_d.col, B_d.val, B_d.col, B_d.row,
                         CUSPARSE_ACTION_NUMERIC, 
                         CUSPARSE_INDEX_BASE_ZERO);
         if(cusparseStatus != 0)    
                printf("error in transpose: %d.\n", cusparseStatus);

        cusparseDestroyMatDescr( descrA );
        cusparseDestroyMatDescr( descrB );
        cusparseDestroy( handle );

        // end CUSPARSE context //

        magma_z_mtransfer( B_d, B, Magma_DEV, A.memory_location );
        magma_z_mfree( &A_d );
        magma_z_mfree( &B_d );

        return MAGMA_SUCCESS;
    }
    return MAGMA_SUCCESS;
}



