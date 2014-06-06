/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from matrix_zio.cpp normal z -> c, Fri May 30 10:41:46 2014
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


using namespace std;

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Reads in a matrix stored in coo format from a binary and converts it
    into CSR format. It duplicates the off-diagonal entries in the 
    symmetric case.


    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaFloatComplex **val             value array of CSR output 
    magma_index_t **row                  row pointer of CSR output
    magma_index_t **col                  column indices of CSR output
    const char * filename                filname of the binary matrix

    ========================================================================  */

extern "C"
magma_int_t read_c_csr_from_binary( magma_int_t* n_row, 
                                    magma_int_t* n_col, 
                                    magma_int_t* nnz, 
                                    magmaFloatComplex **val, 
                                    magma_index_t **row, 
                                    magma_index_t **col, 
                                    const char * filename ){


  std::fstream binary_test(filename);


  if(binary_test){
    printf("#Start reading...");
    fflush(stdout);
  }
  else{
    printf("#Unable to open file %s.\n", filename);
    fflush(stdout);
    exit(1);
  }
  binary_test.close();
  
  
  std::fstream binary_rfile(filename,std::ios::binary|std::ios::in);
  
  
  //read number of rows
  binary_rfile.read(reinterpret_cast<char *>(n_row),sizeof(int));
  
  //read number of columns
  binary_rfile.read(reinterpret_cast<char *>(n_col),sizeof(int));
  
  //read number of nonzeros
  binary_rfile.read(reinterpret_cast<char *>(nnz),sizeof(int));
  
  
  *val = new magmaFloatComplex[*nnz];
  *col = new magma_index_t[*nnz];
  *row = new magma_index_t[*n_row+1];
  
  
  //read row pointer
  for(magma_int_t i=0;i<=*n_row;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*row)[i]),sizeof(int));
  }
  
  //read col
  for(magma_int_t i=0;i<*nnz;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*col)[i]),sizeof(int));
  }
  
  //read val
  for(magma_int_t i=0;i<*nnz;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*val)[i]),sizeof(float));
  }

  binary_rfile.close();
  
  printf("#Finished reading.");
  fflush(stdout);

  return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Reads in a matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It duplicates the off-diagonal
    entries in the symmetric case.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaFloatComplex **val             value array of CSR output 
    magma_index_t **row                  row pointer of CSR output
    magma_index_t **col                  column indices of CSR output
    const char * filename                filname of the mtx matrix

    ========================================================================  */


extern "C"
magma_int_t read_c_csr_from_mtx(    magma_storage_t *type, 
                                    magma_location_t *location, 
                                    magma_int_t* n_row, 
                                    magma_int_t* n_col, 
                                    magma_int_t* nnz, 
                                    magmaFloatComplex **val, 
                                    magma_index_t **row, 
                                    magma_index_t **col, 
                                    const char *filename ){
  
  FILE *fid;
  MM_typecode matcode;
    
  fid = fopen(filename, "r");
  
  if (fid == NULL) {
    printf("#Unable to open file %s\n", filename);
    exit(1);
  }
  
  if (mm_read_banner(fid, &matcode) != 0) {
    printf("#Could not process lMatrix Market banner.\n");
    exit(1);
  }
  
  if (!mm_is_valid(matcode)) {
    printf("#Invalid lMatrix Market file.\n");
    exit(1);
  }
  
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) 
      || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) 
                                        && mm_is_sparse(matcode))) {
    printf("#Sorry, this application does not support ");
    printf("#Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    printf("#Only real-valued or pattern coordinate matrices supported\n");
    exit(1);
  }
  
  magma_index_t num_rows, num_cols, num_nonzeros;
  if (mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
    exit(1);
  
  (*type) = Magma_CSR;
  (*location) = Magma_CPU;
  (*n_row) = (magma_index_t) num_rows;
  (*n_col) = (magma_index_t) num_cols;
  (*nnz)   = (magma_index_t) num_nonzeros;

  magma_index_t *coo_col, *coo_row;
  magmaFloatComplex *coo_val;
  
  coo_col = (magma_index_t *) malloc(*nnz*sizeof(magma_index_t));
  assert(coo_col != NULL);

  coo_row = (magma_index_t *) malloc(*nnz*sizeof(magma_index_t)); 
  assert( coo_row != NULL);

  coo_val = (magmaFloatComplex *) malloc(*nnz*sizeof(magmaFloatComplex));
  assert( coo_val != NULL);


  printf("# Reading sparse matrix from file (%s):",filename);
  fflush(stdout);


  if (mm_is_real(matcode) || mm_is_integer(matcode)){
    for(magma_int_t i = 0; i < *nnz; ++i){
      magma_index_t ROW ,COL;
      float VAL;  // always read in a float and convert later if necessary
      
      fscanf(fid, " %d %d %f \n", &ROW, &COL, &VAL);   
      
      coo_row[i] = (magma_index_t) ROW - 1; 
      coo_col[i] = (magma_index_t) COL - 1;
      coo_val[i] = MAGMA_C_MAKE( VAL, 0.);
    }
  } else {
    printf("Unrecognized data type\n");
    exit(1);
  }
  
  fclose(fid);
  printf(" done\n");
  
  

  if(mm_is_symmetric(matcode)) { //duplicate off diagonal entries
  printf("detected symmetric case\n");
    magma_index_t off_diagonals = 0;
    for(magma_int_t i = 0; i < *nnz; ++i){
      if(coo_row[i] != coo_col[i])
        ++off_diagonals;
    }
    
    magma_index_t true_nonzeros = 2*off_diagonals + (*nnz - off_diagonals);
    
    
    printf("total number of nonzeros: %d\n", (int) *nnz);

    
    
    magma_index_t* new_row = 
        (magma_index_t *) malloc(true_nonzeros*sizeof(magma_index_t)) ; 
    magma_index_t* new_col = 
        (magma_index_t *) malloc(true_nonzeros*sizeof(magma_index_t)) ; 
    magmaFloatComplex* new_val = 
      (magmaFloatComplex *) malloc(true_nonzeros*sizeof(magmaFloatComplex)) ; 
    
    magma_index_t ptr = 0;
    for(magma_int_t i = 0; i < *nnz; ++i) {
        if(coo_row[i] != coo_col[i]) {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
        new_col[ptr] = coo_row[i];  
        new_row[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;  
      } else 
      {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
      }
    }      
    
    free (coo_row);
    free (coo_col);
    free (coo_val);

    coo_row = new_row;  
    coo_col = new_col; 
    coo_val = new_val;   
    
    *nnz = true_nonzeros;
    

  } //end symmetric case
  
  magmaFloatComplex tv;
  magma_index_t ti;
  
  
  //If matrix is not in standard format, sorting is necessary
  /*
  
    std::cout << "Sorting the cols...." << std::endl;
  // bubble sort (by cols)
  for (int i=0; i<*nnz-1; ++i)
    for (int j=0; j<*nnz-i-1; ++j)
      if (coo_col[j] > coo_col[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }

  std::cout << "Sorting the rows...." << std::endl;
  // bubble sort (by rows)
  for (int i=0; i<*nnz-1; ++i)
    for (int j=0; j<*nnz-i-1; ++j)
      if ( coo_row[j] > coo_row[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }
  std::cout << "Sorting: done" << std::endl;
  
  */
  
  
  (*val) = (magmaFloatComplex *) malloc(*nnz*sizeof(magmaFloatComplex)) ;
  assert((*val) != NULL);
  
  (*col) = (magma_index_t *) malloc(*nnz*sizeof(magma_index_t));
  assert((*col) != NULL);
  
  (*row) = (magma_index_t *) malloc((*n_row+1)*sizeof(magma_index_t)) ;
  assert((*row) != NULL);
  

  
  
  // original code from  Nathan Bell and Michael Garland
  // the output CSR structure is NOT sorted!

  for (magma_index_t i = 0; i < num_rows; i++)
    (*row)[i] = 0;
  
  for (magma_index_t i = 0; i < *nnz; i++)
    (*row)[coo_row[i]]++;
  
  
  //cumsum the nnz per row to get Bp[]
  for(magma_int_t i = 0, cumsum = 0; i < num_rows; i++){     
    magma_index_t temp = (*row)[i];
    (*row)[i] = cumsum;
    cumsum += temp;
  }
  (*row)[num_rows] = *nnz;
  
  //write Aj,Ax into Bj,Bx
  for(magma_int_t i = 0; i < *nnz; i++){
    magma_index_t row_  = coo_row[i];
    magma_index_t dest = (*row)[row_];
    
    (*col)[dest] = coo_col[i];
    
    (*val)[dest] = coo_val[i];
    
    (*row)[row_]++;
  }
  
  for(int i = 0, last = 0; i <= num_rows; i++){
    int temp = (*row)[i];
    (*row)[i]  = last;
    last   = temp;
  }
  
  (*row)[*n_row]=*nnz;
     

  for (magma_index_t k=0; k<*n_row; ++k)
    for (magma_index_t i=(*row)[k]; i<(*row)[k+1]-1; ++i) 
      for (magma_index_t j=(*row)[k]; j<(*row)[k+1]-1; ++j) 

      if ( (*col)[j] > (*col)[j+1] ){

        ti = (*col)[j];
        (*col)[j] = (*col)[j+1];
        (*col)[j+1] = ti;

        tv = (*val)[j];
        (*val)[j] = (*val)[j+1];
        (*val)[j+1] = tv;

      }

  return MAGMA_SUCCESS;
}




extern "C"
magma_int_t write_c_csrtomtx( magma_c_sparse_matrix B, const char *filename){

    write_c_csr_mtx( B.num_rows, B.num_cols, B.nnz, &B.val, &B.row, &B.col, 
                     MagmaColMajor, filename );
    return MAGMA_SUCCESS; 
}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Writes a CSR matrix to a file using Matrix Market format.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaFloatComplex **val             value array of CSR  
    magma_index_t **row                  row pointer of CSR 
    magma_index_t **col                  column indices of CSR 
    magma_index_t MajorType                Row or Column sort
                                         default: 0 = RowMajor, 1 = ColMajor
    const char * filename                output filename for the matrix

    ========================================================================  */

extern "C"
magma_int_t write_c_csr_mtx(    magma_int_t n_row, 
                                magma_int_t n_col, 
                                magma_int_t nnz, 
                                magmaFloatComplex **val, 
                                magma_index_t **row, 
                                magma_index_t **col, 
                                magma_order_t MajorType, 
                                const char *filename ){



  if( MajorType == MagmaColMajor){
    //to obtain ColMajr output we transpose the matrix 
    //and flip in the output the row and col pointer
    magmaFloatComplex *new_val;
    magma_index_t *new_row;                    
    magma_index_t *new_col;
    magma_int_t new_n_row;
    magma_int_t new_n_col;
    magma_int_t new_nnz;

    c_transpose_csr( n_row, n_col, nnz, *val, *row, *col, 
        &new_n_row, &new_n_col, &new_nnz, &new_val, &new_row, &new_col);
    printf("Writing sparse matrix to file (%s):",filename);
    fflush(stdout);

    std::ofstream file(filename);
    file<< "%%MatrixMarket matrix coordinate real general ColMajor" <<std::endl;
    file << new_n_row <<" "<< new_n_col <<" "<< new_nnz << std::endl;
   
    magma_index_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {    
      magma_index_t rowtemp1=(new_row)[i];
      magma_index_t rowtemp2=(new_row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        file << ((new_col)[rowtemp1+j]+1) <<" "<< rowindex <<" "<< 
                        MAGMA_C_REAL((new_val)[rowtemp1+j]) << std::endl;
      rowindex++;
    }
    printf(" done\n");

  }
  else{
    printf("Writing sparse matrix to file (%s):",filename);
    fflush(stdout);

    std::ofstream file(filename);
    file<< "%%MatrixMarket matrix coordinate real general RowMajor" <<std::endl;
    file << n_row <<" "<< n_col <<" "<< nnz << std::endl;
   
    magma_index_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {
      magma_index_t rowtemp1=(*row)[i];
      magma_index_t rowtemp2=(*row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        file << rowindex <<" "<< ((*col)[rowtemp1+j]+1) <<" "<< 
                        MAGMA_C_REAL((*val)[rowtemp1+j]) << std::endl;
      rowindex++;
    }
    printf(" done\n");
  }
  return MAGMA_SUCCESS;
}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prints a CSR matrix in Matrix Market format.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaFloatComplex **val             value array of CSR  
    magma_index_t **row                    row pointer of CSR 
    magma_index_t **col                    column indices of CSR
    magma_index_t MajorType                Row or Column sort
                                         default: 0 = RowMajor, 1 = ColMajor

    ========================================================================  */

extern "C"
magma_int_t print_c_csr_mtx(    magma_int_t n_row, 
                                magma_int_t n_col, 
                                magma_int_t nnz, 
                                magmaFloatComplex **val, 
                                magma_index_t **row, 
                                magma_index_t **col, 
                                magma_order_t MajorType ){

  if( MajorType == MagmaColMajor ){
    //to obtain ColMajr output we transpose the matrix 
    //and flip in the output the row and col pointer
    magmaFloatComplex *new_val;
    magma_index_t *new_row;                    
    magma_index_t *new_col;
    magma_int_t new_n_row;
    magma_int_t new_n_col;
    magma_int_t new_nnz;

    c_transpose_csr( n_row, n_col, nnz, *val, *row, *col, 
        &new_n_row, &new_n_col, &new_nnz, &new_val, &new_row, &new_col);

    cout<< "%%MatrixMarket matrix coordinate real general ColMajor" <<std::endl;
    cout << new_n_row <<" "<< new_n_col <<" "<< new_nnz << std::endl;
   
    magma_index_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {    
      magma_index_t rowtemp1=(new_row)[i];
      magma_index_t rowtemp2=(new_row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        cout << ((new_col)[rowtemp1+j]+1) <<" "<< rowindex <<" "<< 
                        MAGMA_C_REAL((new_val)[rowtemp1+j]) << std::endl;
      rowindex++;
    }
  }
  else{
    cout<< "%%MatrixMarket matrix coordinate real general RowMajor" <<std::endl;
    cout << n_row <<" "<< n_col <<" "<< nnz << std::endl;
   
    magma_index_t i=0, j=0, rowindex=1;

    for(i=0; i<n_col; i++)
    {
      magma_index_t rowtemp1=(*row)[i];
      magma_index_t rowtemp2=(*row)[i+1];
      for(j=0; j<rowtemp2-rowtemp1; j++)  
        cout<< rowindex <<" "<< (*col)[rowtemp1+j]+1 <<" "<< 
                        MAGMA_C_REAL((*val)[rowtemp1+j]) <<endl;
      rowindex++;
    }
  }
  return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prints a CSR matrix in CSR format.

    Arguments
    =========

    magma_int_t* n_row                   number of rows in matrix
    magma_int_t* n_col                   number of columns in matrix
    magma_int_t* nnz                     number of nonzeros 
    magmaFloatComplex **val             value array of CSR  
    magma_index_t **row                  row pointer of CSR 
    magma_index_t **col                  column indices of CSR 

    ========================================================================  */

extern "C"
magma_int_t print_c_csr(    magma_int_t n_row, 
                            magma_int_t n_col, 
                            magma_int_t nnz, 
                            magmaFloatComplex **val, 
                            magma_index_t **row, 
                            magma_index_t **col ){

  cout << "Matrix in CSR format (row col val)" << endl;
  cout << n_row <<" "<< n_col <<" "<< nnz <<endl;
   
  magma_index_t i=0,j=0;

  for(i=0; i<n_col; i++)
  {
    magma_index_t rowtemp1=(*row)[i];
    magma_index_t rowtemp2=(*row)[i+1];
    for(j=0; j<rowtemp2-rowtemp1; j++)  
      cout<< (rowtemp1+1) <<" "<< (*col)[rowtemp1+j]+1 <<" "<< 
            MAGMA_C_REAL((*val)[rowtemp1+j]) <<endl;
  }
  return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Prints a sparse matrix in CSR format.

    Arguments
    =========

    magma_c_sparse_matrix A              sparse matrix in Magma_CSR format

    ========================================================================  */

extern "C"
magma_int_t magma_c_mvisu( magma_c_sparse_matrix A )
{
    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_DENSE ){
            for( magma_index_t i=0; i<(A.num_rows); i++ ){
              for( magma_index_t j=0; j<A.num_cols; j++ )
                printf("%4.2f ", MAGMA_C_REAL( A.val[i*(A.num_cols)+j] ) );
                //cout << MAGMA_C_REAL( A.val[i*(A.num_cols)+j] ) << " " ;
              cout << endl;
            }
        }
        else if( A.storage_type == Magma_CSR ){
            magma_c_sparse_matrix C;
            magma_c_mconvert( A, &C, A.storage_type, Magma_DENSE );
            magma_c_mvisu(  C );
            magma_c_mfree(&C);
        }
        else{
            magma_c_sparse_matrix C, D;
            magma_c_mconvert( A, &C, A.storage_type, Magma_CSR );
            magma_c_mconvert( C, &D, Magma_CSR, Magma_DENSE );
            magma_c_mvisu(  D );
            magma_c_mfree(&C);
            magma_c_mfree(&D);
        }
    }
    else{
        magma_c_sparse_matrix C;
        magma_c_mtransfer( A, &C, A.memory_location, Magma_CPU );
        magma_c_mvisu(  C );
        magma_c_mfree(&C);
    }

  return MAGMA_SUCCESS;
}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Reads in a matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It duplicates the off-diagonal
    entries in the symmetric case.

    Arguments
    =========

    magma_c_sparse_matrix *A             matrix in magma sparse matrix format
    const char * filename                filname of the mtx matrix

    ========================================================================  */

extern "C"
magma_int_t magma_c_csr_mtx( magma_c_sparse_matrix *A, const char *filename ){

  int csr_compressor = 0;       // checks for zeros in original file

  FILE *fid;
  MM_typecode matcode;
    
  fid = fopen(filename, "r");
  
  if (fid == NULL) {
    printf("#Unable to open file %s\n", filename);
    exit(1);
  }
  
  if (mm_read_banner(fid, &matcode) != 0) {
    printf("#Could not process lMatrix Market banner.\n");
    exit(1);
  }
  
  if (!mm_is_valid(matcode)) {
    printf("#Invalid lMatrix Market file.\n");
    exit(1);
  }
  
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) 
        || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) 
            && mm_is_sparse(matcode))) {
    printf("#Sorry, this application does not support ");
    printf("#Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    printf("#Only real-valued or pattern coordinate matrices are supported\n");
    exit(1);
  }
  
  magma_index_t num_rows, num_cols, num_nonzeros;
  if (mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
    exit(1);
  
  (A->storage_type) = Magma_CSR;
  (A->memory_location) = Magma_CPU;
  (A->num_rows) = (magma_index_t) num_rows;
  (A->num_cols) = (magma_index_t) num_cols;
  (A->nnz)   = (magma_index_t) num_nonzeros;

  magma_index_t *coo_col, *coo_row;
  magmaFloatComplex *coo_val;
  
  coo_col = (magma_index_t *) malloc(A->nnz*sizeof(magma_index_t));
  assert(coo_col != NULL);

  coo_row = (magma_index_t *) malloc(A->nnz*sizeof(magma_index_t)); 
  assert( coo_row != NULL);

  coo_val = (magmaFloatComplex *) malloc(A->nnz*sizeof(magmaFloatComplex));
  assert( coo_val != NULL);


  printf("# Reading sparse matrix from file (%s):",filename);
  fflush(stdout);


  if (mm_is_real(matcode) || mm_is_integer(matcode)){
    for(magma_int_t i = 0; i < A->nnz; ++i){
      magma_index_t ROW ,COL;
      float VAL;  // always read in a float and convert later if necessary
      
      fscanf(fid, " %d %d %f \n", &ROW, &COL, &VAL);   
      if( VAL == 0 ) 
        csr_compressor=1;
      coo_row[i] = (magma_index_t) ROW - 1; 
      coo_col[i] = (magma_index_t) COL - 1;
      coo_val[i] = MAGMA_C_MAKE( VAL, 0.);
    }
  } else {
    printf("Unrecognized data type\n");
    exit(1);
  }
  
  fclose(fid);
  printf(" done\n");
  
  
   A->sym = Magma_GENERAL;

  if(mm_is_symmetric(matcode)) { //duplicate off diagonal entries
    A->sym = Magma_SYMMETRIC;
  //printf("detected symmetric case\n");
    magma_index_t off_diagonals = 0;
    for(magma_int_t i = 0; i < A->nnz; ++i){
      if(coo_row[i] != coo_col[i])
        ++off_diagonals;
    }
    magma_index_t true_nonzeros = 2 * off_diagonals + (A->nnz - off_diagonals);
      
    magmaFloatComplex *new_val;
    magma_index_t* new_row;
    magma_index_t* new_col;
    magma_cmalloc_cpu( &new_val, true_nonzeros );
    magma_indexmalloc_cpu( &new_row, true_nonzeros );
    magma_indexmalloc_cpu( &new_col, true_nonzeros );

    magma_index_t ptr = 0;
    for(magma_int_t i = 0; i < A->nnz; ++i) {
        if(coo_row[i] != coo_col[i]) {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
        new_col[ptr] = coo_row[i];  
        new_row[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;  
      } else 
      {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
      }
    }      
    
    free (coo_row);
    free (coo_col);
    free (coo_val);

    coo_row = new_row;  
    coo_col = new_col; 
    coo_val = new_val;     
    A->nnz = true_nonzeros;
    //printf("total number of nonzeros: %d\n",A->nnz);    

  } //end symmetric case
  
  magmaFloatComplex tv;
  magma_index_t ti;
  
  
  //If matrix is not in standard format, sorting is necessary
  /*
  
    std::cout << "Sorting the cols...." << std::endl;
  // bubble sort (by cols)
  for (int i=0; i<A->nnz-1; ++i)
    for (int j=0; j<A->nnz-i-1; ++j)
      if (coo_col[j] > coo_col[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }

  std::cout << "Sorting the rows...." << std::endl;
  // bubble sort (by rows)
  for (int i=0; i<A->nnz-1; ++i)
    for (int j=0; j<A->nnz-i-1; ++j)
      if ( coo_row[j] > coo_row[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }
  std::cout << "Sorting: done" << std::endl;
  
  */
  
  
  magma_cmalloc_cpu( &A->val, A->nnz );
  assert((A->val) != NULL);
  
  magma_indexmalloc_cpu( &A->col, A->nnz );
  assert((A->col) != NULL);
  
  magma_indexmalloc_cpu( &A->row, A->num_rows+1 );
  assert((A->row) != NULL);
  

  
  
  // original code from  Nathan Bell and Michael Garland
  // the output CSR structure is NOT sorted!

  for (magma_index_t i = 0; i < num_rows; i++)
    (A->row)[i] = 0;
  
  for (magma_index_t i = 0; i < A->nnz; i++)
    (A->row)[coo_row[i]]++;
  
  
  //cumsum the nnz per row to get Bp[]
  for(magma_int_t i = 0, cumsum = 0; i < num_rows; i++){     
    magma_index_t temp = (A->row)[i];
    (A->row)[i] = cumsum;
    cumsum += temp;
  }
  (A->row)[num_rows] = A->nnz;
  
  //write Aj,Ax into Bj,Bx
  for(magma_int_t i = 0; i < A->nnz; i++){
    magma_index_t row_  = coo_row[i];
    magma_index_t dest = (A->row)[row_];
    
    (A->col)[dest] = coo_col[i];
    
    (A->val)[dest] = coo_val[i];
    
    (A->row)[row_]++;
  }
  free (coo_row);
  free (coo_col);
  free (coo_val);
  
  for(int i = 0, last = 0; i <= num_rows; i++){
    int temp = (A->row)[i];
    (A->row)[i]  = last;
    last   = temp;
  }
  
  (A->row)[A->num_rows]=A->nnz;
     

  for (magma_index_t k=0; k<A->num_rows; ++k)
    for (magma_index_t i=(A->row)[k]; i<(A->row)[k+1]-1; ++i) 
      for (magma_index_t j=(A->row)[k]; j<(A->row)[k+1]-1; ++j) 

      if ( (A->col)[j] > (A->col)[j+1] ){

        ti = (A->col)[j];
        (A->col)[j] = (A->col)[j+1];
        (A->col)[j+1] = ti;

        tv = (A->val)[j];
        (A->val)[j] = (A->val)[j+1];
        (A->val)[j+1] = tv;

      }
  if( csr_compressor > 0){ // run the CSR compressor to remove zeros
      //printf("removing zeros: ");
      magma_c_sparse_matrix B;
      magma_c_mtransfer( *A, &B, Magma_CPU, Magma_CPU ); 
      magma_c_csr_compressor(&(A->val), 
                        &(A->row),
                         &(A->col), 
                       &B.val, &B.row, &B.col, &B.num_rows, &B.num_rows); 
      B.nnz = B.row[num_rows];
     // printf(" remaining nonzeros:%d ", B.nnz); 
      magma_free_cpu( A->val ); 
      magma_free_cpu( A->row ); 
      magma_free_cpu( A->col ); 
      magma_c_mtransfer( B, A, Magma_CPU, Magma_CPU ); 
      magma_c_mfree( &B ); 
     // printf("done.\n");
  }
  return MAGMA_SUCCESS;
}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Reads in a SYMMETRIC matrix stored in coo format from a Matrix Market (.mtx)
    file and converts it into CSR format. It does not duplicate the off-diagonal
    entries!

    Arguments
    =========

    magma_c_sparse_matrix *A             matrix in magma sparse matrix format
    const char * filename                filname of the mtx matrix

    =====================================================================  */

extern "C"
magma_int_t magma_c_csr_mtxsymm( magma_c_sparse_matrix *A, 
                                 const char *filename ){

  int csr_compressor = 0;       // checks for zeros in original file

  FILE *fid;
  MM_typecode matcode;
    
  fid = fopen(filename, "r");
  
  if (fid == NULL) {
    printf("#Unable to open file %s\n", filename);
    exit(1);
  }
  
  if (mm_read_banner(fid, &matcode) != 0) {
    printf("#Could not process lMatrix Market banner.\n");
    exit(1);
  }
  
  if (!mm_is_valid(matcode)) {
    printf("#Invalid lMatrix Market file.\n");
    exit(1);
  }
  
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) 
        || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) 
            && mm_is_sparse(matcode))) {
    printf("#Sorry, this application does not support ");
    printf("#Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    printf("#Only real-valued or pattern coordinate matrices are supported\n");
    exit(1);
  }
  
  magma_index_t num_rows, num_cols, num_nonzeros;
  if (mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
    exit(1);
  
  (A->storage_type) = Magma_CSR;
  (A->memory_location) = Magma_CPU;
  (A->num_rows) = (magma_index_t) num_rows;
  (A->num_cols) = (magma_index_t) num_cols;
  (A->nnz)   = (magma_index_t) num_nonzeros;

  magma_index_t *coo_col, *coo_row;
  magmaFloatComplex *coo_val;
  
  coo_col = (magma_index_t *) malloc(A->nnz*sizeof(magma_index_t));
  assert(coo_col != NULL);

  coo_row = (magma_index_t *) malloc(A->nnz*sizeof(magma_index_t)); 
  assert( coo_row != NULL);

  coo_val = (magmaFloatComplex *) malloc(A->nnz*sizeof(magmaFloatComplex));
  assert( coo_val != NULL);


  printf("# Reading sparse matrix from file (%s):",filename);
  fflush(stdout);


  if (mm_is_real(matcode) || mm_is_integer(matcode)){
    for(magma_int_t i = 0; i < A->nnz; ++i){
      magma_index_t ROW ,COL;
      float VAL;  // always read in a float and convert later if necessary
      
      fscanf(fid, " %d %d %f \n", &ROW, &COL, &VAL);   
      if( VAL == 0 ) 
        csr_compressor=1;
      coo_row[i] = (magma_index_t) ROW - 1; 
      coo_col[i] = (magma_index_t) COL - 1;
      coo_val[i] = MAGMA_C_MAKE( VAL, 0.);
    }
  } else {
    printf("Unrecognized data type\n");
    exit(1);
  }
  
  fclose(fid);
  printf(" done\n");
  
  
   A->sym = Magma_GENERAL;

  if(mm_is_symmetric(matcode)) { //do not duplicate off diagonal entries!
    A->sym = Magma_SYMMETRIC;
  } //end symmetric case
  
  magmaFloatComplex tv;
  magma_index_t ti;
  
  
  //If matrix is not in standard format, sorting is necessary
  /*
  
    std::cout << "Sorting the cols...." << std::endl;
  // bubble sort (by cols)
  for (int i=0; i<A->nnz-1; ++i)
    for (int j=0; j<A->nnz-i-1; ++j)
      if (coo_col[j] > coo_col[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }

  std::cout << "Sorting the rows...." << std::endl;
  // bubble sort (by rows)
  for (int i=0; i<A->nnz-1; ++i)
    for (int j=0; j<A->nnz-i-1; ++j)
      if ( coo_row[j] > coo_row[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }
  std::cout << "Sorting: done" << std::endl;
  
  */
  
  
  magma_cmalloc_cpu( &A->val, A->nnz );
  assert((A->val) != NULL);
  
  magma_indexmalloc_cpu( &A->col, A->nnz );
  assert((A->col) != NULL);
  
  magma_indexmalloc_cpu( &A->row, A->num_rows+1 );
  assert((A->row) != NULL);
  

  
  
  // original code from  Nathan Bell and Michael Garland
  // the output CSR structure is NOT sorted!

  for (magma_index_t i = 0; i < num_rows; i++)
    (A->row)[i] = 0;
  
  for (magma_index_t i = 0; i < A->nnz; i++)
    (A->row)[coo_row[i]]++;
  
  
  //cumsum the nnz per row to get Bp[]
  for(magma_int_t i = 0, cumsum = 0; i < num_rows; i++){     
    magma_index_t temp = (A->row)[i];
    (A->row)[i] = cumsum;
    cumsum += temp;
  }
  (A->row)[num_rows] = A->nnz;
  
  //write Aj,Ax into Bj,Bx
  for(magma_int_t i = 0; i < A->nnz; i++){
    magma_index_t row_  = coo_row[i];
    magma_index_t dest = (A->row)[row_];
    
    (A->col)[dest] = coo_col[i];
    
    (A->val)[dest] = coo_val[i];
    
    (A->row)[row_]++;
  }
  free (coo_row);
  free (coo_col);
  free (coo_val);
  
  for(int i = 0, last = 0; i <= num_rows; i++){
    int temp = (A->row)[i];
    (A->row)[i]  = last;
    last   = temp;
  }
  
  (A->row)[A->num_rows]=A->nnz;
     

  for (magma_index_t k=0; k<A->num_rows; ++k)
    for (magma_index_t i=(A->row)[k]; i<(A->row)[k+1]-1; ++i) 
      for (magma_index_t j=(A->row)[k]; j<(A->row)[k+1]-1; ++j) 

      if ( (A->col)[j] > (A->col)[j+1] ){

        ti = (A->col)[j];
        (A->col)[j] = (A->col)[j+1];
        (A->col)[j+1] = ti;

        tv = (A->val)[j];
        (A->val)[j] = (A->val)[j+1];
        (A->val)[j+1] = tv;

      }
  if( csr_compressor > 0){ // run the CSR compressor to remove zeros
      //printf("removing zeros: ");
      magma_c_sparse_matrix B;
      magma_c_mtransfer( *A, &B, Magma_CPU, Magma_CPU ); 
      magma_c_csr_compressor(&(A->val), 
                        &(A->row),
                         &(A->col), 
                       &B.val, &B.row, &B.col, &B.num_rows, &B.num_rows); 
      B.nnz = B.row[num_rows];
     // printf(" remaining nonzeros:%d ", B.nnz); 
      magma_free_cpu( A->val ); 
      magma_free_cpu( A->row ); 
      magma_free_cpu( A->col ); 
      magma_c_mtransfer( B, A, Magma_CPU, Magma_CPU ); 
      magma_c_mfree( &B ); 
     // printf("done.\n");
  }
  return MAGMA_SUCCESS;
}
