/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
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
#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"



using namespace std;

/**
    Purpose
    -------

    Copies a matrix from memory location src to memory location dst.


    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                sparse matrix A

    @param
    B           magma_z_sparse_matrix*
                copy of A

    @param
    src         magma_location_t
                original location A

    @param
    dst         magma_location_t
                location of the copy of A

   

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_mtransfer( magma_z_sparse_matrix A,
                   magma_z_sparse_matrix *B,
                   magma_location_t src,
                   magma_location_t dst){
    magma_int_t stat;

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
            stat = magma_zmalloc( &B->val, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.num_rows + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_setvector( A.num_rows+1, A.row, 1, B->row, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->col, 1 );
        }
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_DEV;
            // memory allocation
            stat = magma_zmalloc( &B->val, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.num_rows + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->rowidx, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_setvector( A.num_rows+1, A.row, 1, B->row, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->col, 1 );
            magma_index_setvector( A.nnz, A.rowidx, 1, B->rowidx, 1 );
        }
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
            stat = magma_zmalloc( &B->val, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.num_rows * A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_setvector( A.num_rows * A.max_nnz_row, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.num_rows * A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_setvector( A.num_rows * A.max_nnz_row, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, 2*A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, 2*A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( 2 * A.num_rows * A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_setvector( 2 * A.num_rows * A.max_nnz_row, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, A.num_rows * rowlength );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.num_rows * rowlength );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.num_rows );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.num_rows * rowlength, A.val, 1, B->val, 1 );
            magma_index_setvector( A.num_rows * rowlength, A.col, 1, B->col, 1 );
            magma_index_setvector( A.num_rows, A.row, 1, B->row, 1 );
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
            stat = magma_zmalloc( &B->val, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.numblocks + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->col, 1 );
            magma_index_setvector( A.numblocks+1, A.row, 1, B->row, 1 );
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
            stat = magma_zmalloc( &B->val, size_b*size_b*A.numblocks );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, r_blocks + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.numblocks );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            magma_index_malloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            magma_zsetvector( size_b*size_b*A.numblocks, A.val, 1, B->val, 1 );
            magma_index_setvector( r_blocks+1, A.row, 1, B->row, 1 );
            magma_index_setvector( A.numblocks, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, A.num_rows*A.num_cols );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zsetvector( A.num_rows*A.num_cols, A.val, 1, B->val, 1 );
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
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_index_malloc_cpu( &B->row, A.num_rows+1 );
            magma_index_malloc_cpu( &B->col, A.nnz );
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
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_index_malloc_cpu( &B->row, A.num_rows+1 );
            magma_index_malloc_cpu( &B->col, A.nnz );
            magma_index_malloc_cpu( &B->rowidx, A.nnz );
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
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
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
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
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
            magma_zmalloc_cpu( &B->val, 2*A.num_rows*A.max_nnz_row );
            magma_index_malloc_cpu( &B->col, 2*A.num_rows*A.max_nnz_row );
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
            magma_zmalloc_cpu( &B->val, rowlength*A.num_rows );
            magma_index_malloc_cpu( &B->row, A.num_rows );
            magma_index_malloc_cpu( &B->col, rowlength*A.num_rows );
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
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_index_malloc_cpu( &B->col, A.nnz );
            magma_index_malloc_cpu( &B->row, A.numblocks+1 );
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
            magma_zmalloc_cpu( &B->val, A.num_rows*A.num_cols );
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
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_index_malloc_cpu( &B->row, A.num_rows+1 );
            magma_index_malloc_cpu( &B->col, A.nnz );
            // data transfer
            magma_zgetvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_getvector( A.num_rows+1, A.row, 1, B->row, 1 );
            magma_index_getvector( A.nnz, A.col, 1, B->col, 1 );
        }
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_CPU;
            // memory allocation
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_index_malloc_cpu( &B->row, A.num_rows+1 );
            magma_index_malloc_cpu( &B->col, A.nnz );
            magma_index_malloc_cpu( &B->rowidx, A.nnz );
            // data transfer
            magma_zgetvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_getvector( A.num_rows+1, A.row, 1, B->row, 1 );
            magma_index_getvector( A.nnz, A.col, 1, B->col, 1 );
            magma_index_getvector( A.nnz, A.rowidx, 1, B->rowidx, 1 );
        }
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
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            magma_zgetvector( A.num_rows*A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_getvector( A.num_rows*A.max_nnz_row, A.col, 1, B->col, 1 );
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
            magma_zmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row );
            magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row );
            // data transfer
            magma_zgetvector( A.num_rows*A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_getvector( A.num_rows*A.max_nnz_row, A.col, 1, B->col, 1 );
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
            magma_zmalloc_cpu( &B->val, 2*A.num_rows*A.max_nnz_row );
            magma_index_malloc_cpu( &B->col, 2*A.num_rows*A.max_nnz_row );
            // data transfer
            magma_zgetvector( 2*A.num_rows*A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_getvector( 2*A.num_rows*A.max_nnz_row, A.col, 1, B->col, 1 );
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
            magma_zmalloc_cpu( &B->val, rowlength*A.num_rows );
            magma_index_malloc_cpu( &B->row, A.num_rows );
            magma_index_malloc_cpu( &B->col, rowlength*A.num_rows );
            // data transfer
            magma_zgetvector( A.num_rows*rowlength, A.val, 1, B->val, 1 );
            magma_index_getvector( A.num_rows*rowlength, A.col, 1, B->col, 1 );
            magma_index_getvector( A.num_rows, A.row, 1, B->row, 1 );
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
            magma_zmalloc_cpu( &B->val, A.nnz );
            magma_index_malloc_cpu( &B->col, A.nnz );
            magma_index_malloc_cpu( &B->row, A.numblocks+1 );
            // data transfer
            magma_zgetvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_getvector( A.nnz, A.col, 1, B->col, 1 );
            magma_index_getvector( A.numblocks+1, A.row, 1, B->row, 1 );
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
            magma_zmalloc_cpu( &B->val, A.numblocks*A.blocksize*A.blocksize );
            magma_index_malloc_cpu( &B->row, r_blocks+1 );
            magma_index_malloc_cpu( &B->col, A.numblocks );
            magma_index_malloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            magma_zgetvector( A.numblocks * A.blocksize * A.blocksize, A.val, 1, B->val, 1 );
            magma_index_getvector( r_blocks+1, A.row, 1, B->row, 1 );
            magma_index_getvector( A.numblocks, A.col, 1, B->col, 1 );
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
            magma_zmalloc_cpu( &B->val, A.num_rows*A.num_cols );
            // data transfer
            magma_zgetvector( A.num_rows*A.num_cols, A.val, 1, B->val, 1 );
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
            stat = magma_zmalloc( &B->val, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.num_rows + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_copyvector( (A.num_rows+1), A.row, 1, B->row, 1 );
            magma_index_copyvector( A.nnz, A.col, 1, B->col, 1 );
        }
        //CSRCOO-type
        if( A.storage_type == Magma_CSRCOO ){
            // fill in information for B
            *B = A;
            B->memory_location = Magma_DEV;
            // memory allocation
            stat = magma_zmalloc( &B->val, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.num_rows + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->rowidx, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_copyvector( (A.num_rows+1), A.row, 1, B->row, 1 );
            magma_index_copyvector( A.nnz, A.col, 1, B->col, 1 );
            magma_index_copyvector( A.nnz, A.rowidx, 1, B->rowidx, 1 );
        }
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
            stat = magma_zmalloc( &B->val, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.num_rows*A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_copyvector( A.num_rows*A.max_nnz_row, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.num_rows*A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_copyvector( A.num_rows*A.max_nnz_row, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, 2 * A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, 2 * A.num_rows * A.max_nnz_row );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( 2 * A.num_rows*A.max_nnz_row, A.val, 1, B->val, 1 );
            magma_index_copyvector( 2 * A.num_rows*A.max_nnz_row, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, A.num_rows * rowlength );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.num_rows * rowlength );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.num_rows );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.num_rows * rowlength, A.val, 1, B->val, 1 );
            magma_index_copyvector( A.num_rows * rowlength, A.col, 1, B->col, 1 );
            magma_index_copyvector( A.num_rows, A.row, 1, B->row, 1 );
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
            stat = magma_zmalloc( &B->val, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.nnz );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, A.numblocks + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.nnz, A.val, 1, B->val, 1 );
            magma_index_copyvector( A.nnz,         A.col, 1, B->col, 1 );
            magma_index_copyvector( A.numblocks+1, A.row, 1, B->row, 1 );
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
            stat = magma_zmalloc( &B->val, size_b*size_b*A.numblocks );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->row, r_blocks + 1 );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            stat = magma_index_malloc( &B->col, A.numblocks );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            magma_index_malloc_cpu( &B->blockinfo, r_blocks * c_blocks );
            // data transfer
            magma_zcopyvector( size_b*size_b*A.numblocks, A.val, 1, B->val, 1 );
            magma_index_copyvector( (r_blocks+1), A.row, 1, B->row, 1 );
            magma_index_copyvector( A.numblocks, A.col, 1, B->col, 1 );
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
            stat = magma_zmalloc( &B->val, A.num_rows*A.num_cols );
            if( stat != 0 )
            {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
            // data transfer
            magma_zcopyvector( A.num_rows*A.num_cols, A.val, 1, B->val, 1 );
        }
    }


    return MAGMA_SUCCESS;
}

























/**
    Purpose
    -------

    Copies a vector from memory location src to memory location dst.


    Arguments
    ---------

    @param
    x           magma_z_vector
                vector x

    @param
    y           magma_z_vector*
                copy of x

    @param
    src         magma_location_t
                original location x

    @param
    dst         magma_location_t
                location of the copy of x

   

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_vtransfer( magma_z_vector x,
                   magma_z_vector *y,
                   magma_location_t src,
                   magma_location_t dst){

    magma_int_t stat;

    // first case: copy matrix from host to device
    if( src == Magma_CPU && dst == Magma_DEV ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        stat = magma_zmalloc( &y->val, x.num_rows );
        if( stat != 0 )
            {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        magma_zsetvector( x.num_rows, x.val, 1, y->val, 1 );
    }
    // second case: copy matrix from host to host
    if( src == Magma_CPU && dst == Magma_CPU ){
        // fill in information for B
        y->memory_location = Magma_CPU;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        magma_zmalloc_cpu( &y->val, x.num_rows );
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
        magma_zmalloc_cpu( &y->val, x.num_rows );
        // data transfer
        magma_zgetvector( x.num_rows, x.val, 1, y->val, 1 );
    }
    // fourth case: copy matrix from device to device
    if( src == Magma_DEV && dst == Magma_DEV ){
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->nnz = x.nnz;
        // memory allocation
        stat = magma_zmalloc( &y->val, x.num_rows );
        if( stat != 0 )
            {printf("Memory Allocation Error transferring vector\n"); exit(0); }
        // data transfer
        magma_zcopyvector( x.num_rows, x.val, 1, y->val, 1 );
    }

    return MAGMA_SUCCESS;
}
