/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_z_transfer.cpp normal z -> c, Fri May 30 10:41:46 2014
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
#include "../include/magmasparse_c.h"
#include "../../include/magma.h"
#include "../include/mmio.h"



using namespace std;

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Copies a matrix from memory location src to memory location dst.


    Arguments
    =========

    magma_c_sparse_matrix A              sparse matrix A    
    magma_c_sparse_matrix *B             copy of A      
    magma_location_t src                 original location A
    magma_location_t dst                 location of the copy of A
   

    ========================================================================  */

magma_int_t 
magma_c_mtransfer( magma_c_sparse_matrix A, 
                   magma_c_sparse_matrix *B, 
                   magma_location_t src, 
                   magma_location_t dst){
    cublasStatus stat;

    // first case: copy matrix from host to device
    if( src == Magma_CPU && dst == Magma_DEV ){
        //CSR-type
        if( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC 
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaFloatComplex ), 
                                                        ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  A.num_rows+1 , sizeof( magma_index_t ), 
                                                        ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                        ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.nnz , sizeof( magmaFloatComplex ), 
                                                    A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows+1 , sizeof( magma_index_t ), 
                                                    A.row, 1, B->row, 1 );
            cublasSetVector( A.nnz , sizeof( magma_index_t ), 
                                                    A.col, 1, B->col, 1 ); 
        } 
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_DEV;
            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaFloatComplex ), 
                                                        ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  A.num_rows+1 , sizeof( magma_index_t ), 
                                                        ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                        ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                        ( void** )&B->rowidx );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.nnz , sizeof( magmaFloatComplex ), 
                                                    A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows+1 , sizeof( magma_index_t ), 
                                                    A.row, 1, B->row, 1 );
            cublasSetVector( A.nnz , sizeof( magma_index_t ), 
                                                    A.col, 1, B->col, 1 ); 
            cublasSetVector( A.nnz , sizeof( magma_index_t ), 
                                                    A.rowidx, 1, B->rowidx, 1 ); 
        } 
   /*     //CSRCSC-type 
        if( A.storage_type == Magma_CSRCSCL || 
            A.storage_type == Magma_CSRCSCU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaFloatComplex ), 
                                                        ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
        //    stat = cublasAlloc( A.num_rows, sizeof( magmaFloatComplex ), 
        //                                                ( void** )&B->diag );
         //   if( ( int )stat != 0 ) 
         //   {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  (A.num_rows+1) , sizeof( magma_index_t ), 
                                                        ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                        ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                    ( void** )&B->blockinfo );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.nnz , sizeof( magmaFloatComplex ), 
                                                    A.val, 1, B->val, 1 );
         //   cublasSetVector( A.num_rows , sizeof( magmaFloatComplex ), 
           //                                         A.diag, 1, B->diag, 1 );
            cublasSetVector( (A.num_rows+1) , sizeof( magma_index_t ), 
                                                    A.row, 1, B->row, 1 );
            cublasSetVector( A.nnz , sizeof( magma_index_t ), 
                                                    A.col, 1, B->col, 1 ); 
            cublasSetVector( A.nnz , sizeof( magma_index_t ), 
                                            A.blockinfo, 1, B->blockinfo, 1 ); 
        } */
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                            sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                                sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows * A.max_nnz_row , 
                            sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows * A.max_nnz_row , 
                            sizeof( magma_index_t )  , A.col, 1, B->col, 1 ); 
        } 
        //ELL-type
        if( A.storage_type == Magma_ELL || A.storage_type == Magma_ELLD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                            sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                                  sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows * A.max_nnz_row , 
                            sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows * A.max_nnz_row , 
                            sizeof( magma_index_t )  , A.col, 1, B->col, 1 ); 
        } 
        //ELLDD-type
        if( A.storage_type == Magma_ELLDD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( 2*A.num_rows * A.max_nnz_row, 
                            sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( 2*A.num_rows * A.max_nnz_row, 
                                  sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( 2 * A.num_rows * A.max_nnz_row , 
                            sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( 2 * A.num_rows * A.max_nnz_row , 
                            sizeof( magma_index_t )  , A.col, 1, B->col, 1 ); 
        } 
        //ELLRT-type
        if( A.storage_type == Magma_ELLRT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            int threads_per_row = A.alignment; 
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                        /threads_per_row) ) * threads_per_row;
            // memory allocation
            stat = cublasAlloc( A.num_rows * rowlength, 
                            sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * rowlength, sizeof( magma_index_t ), 
                                ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows , sizeof( magma_index_t ), 
                                ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows * rowlength , 
                        sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( A.num_rows * rowlength , 
                        sizeof( magma_index_t )  , A.col, 1, B->col, 1 ); 
            cublasSetVector( A.num_rows, sizeof( magma_index_t ), 
                        A.row, 1, B->row, 1 ); 
        } 
        //SELLC-type
        if( A.storage_type == Magma_SELLC || A.storage_type == Magma_SELLP ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaFloatComplex ), 
                                                ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.numblocks+1 , sizeof( magma_index_t ), 
                                ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.nnz, sizeof( magmaFloatComplex ), 
                                                    A.val, 1, B->val, 1 );
            cublasSetVector( A.nnz, sizeof( magma_index_t ), 
                                                    A.col, 1, B->col, 1 ); 
            cublasSetVector( A.numblocks+1, sizeof( magma_index_t ), 
                                                    A.row, 1, B->row, 1 ); 
        } 
        //BCSR-type
        if( A.storage_type == Magma_BCSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                    // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                    // max number of blocks per column
            // memory allocation
            stat = cublasAlloc( size_b*size_b*A.numblocks, 
                            sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  r_blocks+1 , 
                                sizeof( magma_index_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.numblocks, 
                                sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 )             
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            magma_indexmalloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            cublasSetVector( size_b*size_b*A.numblocks , 
                           sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasSetVector( r_blocks+1 , 
                            sizeof( magma_index_t )  , A.row, 1, B->row, 1 );
            cublasSetVector( A.numblocks , 
                            sizeof( magma_index_t )  , A.col, 1, B->col, 1 ); 
            for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
                B->blockinfo[i] = A.blockinfo[i];
            }
        }
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows*A.num_cols, 
                            sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cublasSetVector( A.num_rows*A.num_cols , 
                            sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
        }
    }

    // second case: copy matrix from host to host
    if( src == Magma_CPU && dst == Magma_CPU ){
        //CSR-type
        if( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC 
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.num_rows+1 );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.num_rows+1; i++ ){
                B->row[i] = A.row[i];
            }
        } 
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_CPU;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.num_rows+1 );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            magma_indexmalloc_cpu( &B->rowidx, A.nnz );
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
                B->rowidx[i] = A.rowidx[i];
            }
            for( magma_int_t i=0; i<A.num_rows+1; i++ ){
                B->row[i] = A.row[i];
            }
        } 
  /*      //CSRCSC-type
        if( A.storage_type == Magma_CSRCSCL || 
            A.storage_type == Magma_CSRCSCU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
          //  magma_cmalloc_cpu( &B->diag, A.num_rows );
            magma_indexmalloc_cpu( &B->row, (A.num_rows+1) );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            magma_indexmalloc_cpu( &B->blockinfo, A.nnz );
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
                B->blockinfo[i] = A.blockinfo[i];
            }
            for( magma_int_t i=0; i<(A.num_rows+1); i++ ){
                B->row[i] = A.row[i];
            }
    /*      for( magma_int_t i=0; i<A.num_rows; i++ ){
                B->diag[i] = A.diag[i];
            }*/
   //     } 
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        }
        //ELL-type
        if( A.storage_type == Magma_ELL || A.storage_type == Magma_ELLD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        } 
        //ELLDD-type
        if( A.storage_type == Magma_ELLDD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, 2*A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &B->col, 2*A.num_rows*A.max_nnz_row );
            // data transfer
            for( magma_int_t i=0; i<2*A.num_rows*A.max_nnz_row; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        } 
        //ELLRT-type
        if( A.storage_type == Magma_ELLRT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int threads_per_row = A.alignment; 
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                    /threads_per_row) ) * threads_per_row;
            // memory allocation
            magma_cmalloc_cpu( &B->val, rowlength*A.num_rows );
            magma_indexmalloc_cpu( &B->row, A.num_rows );
            magma_indexmalloc_cpu( &B->col, rowlength*A.num_rows );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*rowlength; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.num_rows; i++ ){
                B->row[i] = A.row[i];
            }
        } 
        //SELLC-type
        if( A.storage_type == Magma_SELLC || A.storage_type == Magma_SELLP ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            B->numblocks = A.numblocks;

            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.numblocks+1 );
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ){
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.numblocks+1; i++ ){
                B->row[i] = A.row[i];
            }
        } 
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.num_rows*A.num_cols );
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.num_cols; i++ ){
                B->val[i] = A.val[i];
            }
        }
    }

    // third case: copy matrix from device to host
    if( src == Magma_DEV && dst == Magma_CPU ){
        //CSR-type
        if( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC 
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.num_rows+1 );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            // data transfer
            cublasGetVector( A.nnz, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows+1, 
                    sizeof( magma_index_t ), A.row, 1, B->row, 1 );            
            cublasGetVector( A.nnz, 
                        sizeof( magma_index_t ), A.col, 1, B->col, 1 );
        } 
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_CPU;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.num_rows+1 );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            magma_indexmalloc_cpu( &B->rowidx, A.nnz );
            // data transfer
            cublasGetVector( A.nnz, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows+1, 
                    sizeof( magma_index_t ), A.row, 1, B->row, 1 );            
            cublasGetVector( A.nnz, 
                        sizeof( magma_index_t ), A.col, 1, B->col, 1 );
            cublasGetVector( A.nnz, 
                        sizeof( magma_index_t ), A.rowidx, 1, B->rowidx, 1 );
        } 
 /*       //CSRCSC-type
        if( A.storage_type == Magma_CSRCSCL || 
            A.storage_type == Magma_CSRCSCU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
          //  magma_cmalloc_cpu( &B->diag, A.num_rows );
            magma_indexmalloc_cpu( &B->row, (A.num_rows+1) );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            magma_indexmalloc_cpu( &B->blockinfo, A.nnz );
            // data transfer
            cublasGetVector( A.nnz, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
       //     cublasGetVector( A.num_rows, 
           //        sizeof( magmaFloatComplex ), A.diag, 1, B->diag, 1 );
            cublasGetVector( (A.num_rows+1), 
                    sizeof( magma_index_t ), A.row, 1, B->row, 1 );            
            cublasGetVector( A.nnz, 
                    sizeof( magma_index_t ), A.col, 1, B->col, 1 );
            cublasGetVector( A.nnz, 
                    sizeof( magma_index_t ), A.blockinfo, 1, B->blockinfo, 1 );
        } */
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            cublasGetVector( A.num_rows*A.max_nnz_row, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows*A.max_nnz_row, 
                            sizeof( magma_index_t ), A.col, 1, B->col, 1 );
        } 
        //ELL-type
        if( A.storage_type == Magma_ELL || A.storage_type == Magma_ELLD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            cublasGetVector( A.num_rows*A.max_nnz_row, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows*A.max_nnz_row, 
                    sizeof( magma_index_t ), A.col, 1, B->col, 1 );
        } 
        //ELLDD-type
        if( A.storage_type == Magma_ELLDD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, 2*A.num_rows*A.max_nnz_row );
            magma_indexmalloc_cpu( &B->col, 2*A.num_rows*A.max_nnz_row );
            // data transfer
            cublasGetVector( 2*A.num_rows*A.max_nnz_row, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( 2*A.num_rows*A.max_nnz_row, 
                    sizeof( magma_index_t ), A.col, 1, B->col, 1 );
        } 
        //ELLRT-type
        if( A.storage_type == Magma_ELLRT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int threads_per_row = A.alignment; 
            // memory allocation
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                /threads_per_row) ) * threads_per_row;
            magma_cmalloc_cpu( &B->val, rowlength*A.num_rows );
            magma_indexmalloc_cpu( &B->row, A.num_rows );
            magma_indexmalloc_cpu( &B->col, rowlength*A.num_rows );
            // data transfer
            cublasGetVector( A.num_rows*rowlength, sizeof( magmaFloatComplex ), 
                                    A.val, 1, B->val, 1 );
            cublasGetVector( A.num_rows*rowlength, sizeof( magma_index_t ), 
                                    A.col, 1, B->col, 1 );
            cublasGetVector( A.num_rows, sizeof( magma_index_t ), 
                                    A.row, 1, B->row, 1 );
        } 
        //SELLC-type
        if( A.storage_type == Magma_SELLC || A.storage_type == Magma_SELLP ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->alignment = A.alignment;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;

            // memory allocation
            magma_cmalloc_cpu( &B->val, A.nnz );
            magma_indexmalloc_cpu( &B->col, A.nnz );
            magma_indexmalloc_cpu( &B->row, A.numblocks+1 );
            // data transfer
            cublasGetVector( A.nnz, sizeof( magmaFloatComplex ), 
                                    A.val, 1, B->val, 1 );
            cublasGetVector( A.nnz, sizeof( magma_index_t ), 
                                    A.col, 1, B->col, 1 );
            cublasGetVector( A.numblocks+1, sizeof( magma_index_t ), 
                                    A.row, 1, B->row, 1 );
        } 
        //BCSR-type
        if( A.storage_type == Magma_BCSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                    // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                    // max number of blocks per column
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.numblocks*A.blocksize*A.blocksize );
            magma_indexmalloc_cpu( &B->row, r_blocks+1 );
            magma_indexmalloc_cpu( &B->col, A.numblocks );
            magma_indexmalloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            cublasGetVector( A.numblocks * A.blocksize * A.blocksize, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
            cublasGetVector( r_blocks+1, 
                    sizeof( magma_index_t ), A.row, 1, B->row, 1 );            
            cublasGetVector(  A.numblocks, 
                    sizeof( magma_index_t ), A.col, 1, B->col, 1 );
            for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
                B->blockinfo[i] = A.blockinfo[i];
            }
        } 
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            magma_cmalloc_cpu( &B->val, A.num_rows*A.num_cols );
            // data transfer
            cublasGetVector( A.num_rows*A.num_cols, 
                    sizeof( magmaFloatComplex ), A.val, 1, B->val, 1 );
        }
    }

    // fourth case: copy matrix from device to device
    if( src == Magma_DEV && dst == Magma_DEV ){
        //CSR-type
        if( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC 
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.nnz, 
                    sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  A.num_rows+1 , 
                    sizeof( magma_index_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, 
                    sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, 
                A.nnz*sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, 
              (A.num_rows+1)*sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, 
                    A.nnz*sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        } 
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_DEV;
            // memory allocation
            stat = cublasAlloc( A.nnz, 
                    sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  A.num_rows+1 , 
                    sizeof( magma_index_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, 
                    sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.nnz, 
                    sizeof( magma_index_t ), ( void** )&B->rowidx );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, 
                A.nnz*sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, 
              (A.num_rows+1)*sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, 
                    A.nnz*sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->rowidx, A.rowidx, 
                    A.nnz*sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        } 
  /*      //CSRCSC-type
        if( A.storage_type == Magma_CSRCSCL || 
            A.storage_type == Magma_CSRCSCU ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.nnz, 
                    sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
          //  stat = cublasAlloc( A.num_rows, 
            //        sizeof( magmaFloatComplex ), ( void** )&B->diag );
            //if( ( int )stat != 0 ) 
            //{printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  (A.num_rows+1) , 
                    sizeof( magma_index_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.nnz, 
                    sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.nnz, 
                    sizeof( magma_index_t ), ( void** )&B->blockinfo );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.nnz*sizeof( magmaFloatComplex ), 
                    cudaMemcpyDeviceToDevice );
       //     cudaMemcpy( B->diag, A.diag,A.num_rows*sizeof( magmaFloatComplex ), 
             //       cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, (A.num_rows+1)*sizeof( magma_index_t ), 
                    cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.nnz*sizeof( magma_index_t ), 
                    cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->blockinfo, A.blockinfo, 
                    A.nnz*sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        } */
        //ELLPACK-type
        if( A.storage_type == Magma_ELLPACK ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                        sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                            sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.max_nnz_row
                *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.num_rows*A.max_nnz_row
                *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        } 
        //ELL-type
        if( A.storage_type == Magma_ELL || A.storage_type == Magma_ELLD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * A.max_nnz_row, 
                sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.max_nnz_row
                    *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.num_rows*A.max_nnz_row
                         *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        }
        //ELLDD-type
        if( A.storage_type == Magma_ELLDD ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( 2 * A.num_rows * A.max_nnz_row, 
                sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( 2 * A.num_rows * A.max_nnz_row, 
                sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, 2 * A.num_rows*A.max_nnz_row
                    *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, 2 * A.num_rows*A.max_nnz_row
                         *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        }
        //ELLRT-type
        if( A.storage_type == Magma_ELLRT ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int threads_per_row = A.alignment; 
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                    /threads_per_row) ) * threads_per_row;
            // memory allocation
            stat = cublasAlloc( A.num_rows * rowlength, 
                    sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows * rowlength, sizeof( magma_index_t ), 
                        ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.num_rows , sizeof( magma_index_t ), 
                        ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows * rowlength
                    *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.num_rows * rowlength
                    *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, A.num_rows
                    *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
        }
        //SELLC/SELLP-type
        if( A.storage_type == Magma_SELLC || A.storage_type == Magma_SELLP ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;

            // memory allocation
            stat = cublasAlloc( A.nnz, sizeof( magmaFloatComplex ), 
                                                ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.nnz, sizeof( magma_index_t ), 
                                                ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc( A.numblocks+1 , sizeof( magma_index_t ), 
                                ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.nnz
                    *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.nnz
                    *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
           cudaMemcpy( B->row, A.row, A.numblocks+1
                    *sizeof( magma_int_t ), cudaMemcpyDeviceToDevice );
        } 
        //BCSR-type
        if( A.storage_type == Magma_BCSR ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                    // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                    // max number of blocks per column
            // memory allocation
            stat = cublasAlloc( size_b*size_b*A.numblocks, 
                    sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = cublasAlloc(  r_blocks+1 , 
                    sizeof( magma_index_t ), ( void** )&B->row );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
            stat = cublasAlloc( A.numblocks, 
                    sizeof( magma_index_t ), ( void** )&B->col );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            magma_indexmalloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            cudaMemcpy( B->val, A.val, size_b*size_b*A.numblocks
                    *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->row, A.row, (r_blocks+1)
                    *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
            cudaMemcpy( B->col, A.col, A.numblocks
                    *sizeof( magma_index_t ), cudaMemcpyDeviceToDevice );
            for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
                B->blockinfo[i] = A.blockinfo[i];
            }
        }
        //DENSE-type
        if( A.storage_type == Magma_DENSE ){
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            stat = cublasAlloc( A.num_rows*A.num_cols, 
                sizeof( magmaFloatComplex ), ( void** )&B->val );
            if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            cudaMemcpy( B->val, A.val, A.num_rows*A.num_cols
                    *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
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

    Copies a vector from memory location src to memory location dst.


    Arguments
    =========

    magma_c_vector x              vector x    
    magma_c_vector *y             copy of x      
    magma_location_t src          original location x
    magma_location_t dst          location of the copy of x
   

    ========================================================================  */

magma_int_t 
magma_c_vtransfer( magma_c_vector x, 
                   magma_c_vector *y, 
                   magma_location_t src, 
                   magma_location_t dst){

    cublasStatus stat;

    // first case: copy matrix from host to device
    if( src == Magma_CPU && dst == Magma_DEV ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        stat = cublasAlloc( x.num_rows, 
                sizeof( magmaFloatComplex ), ( void** )&y->val );
        if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        cublasSetVector( x.num_rows , 
                sizeof( magmaFloatComplex ), x.val, 1, y->val, 1 );
    }
    // second case: copy matrix from host to host
    if( src == Magma_CPU && dst == Magma_CPU ){
        // fill in information for B
        y->memory_location = Magma_CPU;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        magma_cmalloc_cpu( &y->val, x.num_rows );
        // data transfer
        for( magma_int_t i=0; i<x.num_rows; i++ )
            y->val[i] = x.val[i];
    }
    // third case: copy matrix from device to host
    if( src == Magma_DEV && dst == Magma_CPU ){
        // fill in information for B
        y->memory_location = Magma_CPU;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        magma_cmalloc_cpu( &y->val, x.num_rows );
        // data transfer
        cublasGetVector( x.num_rows, 
                sizeof( magmaFloatComplex ), x.val, 1, y->val, 1 );
    }
    // fourth case: copy matrix from device to device
    if( src == Magma_DEV && dst == Magma_DEV ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        stat = cublasAlloc( x.num_rows, 
                    sizeof( magmaFloatComplex ), ( void** )&y->val );
        if( ( int )stat != 0 ) 
            {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        cudaMemcpy( y->val, x.val, x.num_rows
                *sizeof( magmaFloatComplex ), cudaMemcpyDeviceToDevice );
    }

    return MAGMA_SUCCESS;
}


