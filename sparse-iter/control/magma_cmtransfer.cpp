/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zmtransfer.cpp normal z -> c, Sun May  3 11:23:01 2015
       @author Hartwig Anzt
*/
#include "common_magmasparse.h"


/**
    Purpose
    -------

    Copies a matrix from memory location src to memory location dst.


    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                sparse matrix A

    @param[out]
    B           magma_c_matrix*
                copy of A

    @param[in]
    src         magma_location_t
                original location A

    @param[in]
    dst         magma_location_t
                location of the copy of A

   
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmtransfer(
    magma_c_matrix A,
    magma_c_matrix *B,
    magma_location_t src,
    magma_location_t dst,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    B->val = NULL;
    B->col = NULL;
    B->row = NULL;
    B->rowidx = NULL;
    B->blockinfo = NULL;
    B->diag = NULL;
    B->dval = NULL;
    B->dcol = NULL;
    B->drow = NULL;
    B->drowidx = NULL;
    B->ddiag = NULL;

    // first case: copy matrix from host to device
    if ( src == Magma_CPU && dst == Magma_DEV ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            // data transfer
            magma_csetvector( A.nnz, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows+1, A.row, 1, B->drow, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->dcol, 1 );
        }
        //CSRCOO-type
        if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            *B = A;
            B->memory_location = Magma_DEV;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drowidx, A.nnz ));
            // data transfer
            magma_csetvector( A.nnz, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows+1, A.row, 1, B->drow, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->dcol, 1 );
            magma_index_setvector( A.nnz, A.rowidx, 1, B->drowidx, 1 );
        }
        //ELL/ELLPACKT-type
        if ( A.storage_type == Magma_ELLPACKT || A.storage_type == Magma_ELL ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_csetvector( A.num_rows * A.max_nnz_row, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows * A.max_nnz_row, A.col, 1, B->dcol, 1 );
        }
        //ELLD-type
        if ( A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_csetvector( A.num_rows * A.max_nnz_row, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows * A.max_nnz_row, A.col, 1, B->dcol, 1 );
        }
        //ELLRT-type
        if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            int threads_per_row = A.alignment;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                        /threads_per_row) ) * threads_per_row;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows ));
            // data transfer
            magma_csetvector( A.num_rows * rowlength, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows * rowlength, A.col, 1, B->dcol, 1 );
            magma_index_setvector( A.num_rows, A.row, 1, B->drow, 1 );
        }
        //SELLP-type
        if ( A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.numblocks + 1 ));
            // data transfer
            magma_csetvector( A.nnz, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->dcol, 1 );
            magma_index_setvector( A.numblocks+1, A.row, 1, B->drow, 1 );
        }
        //BCSR-type
        if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, size_b*size_b*A.numblocks ));
            CHECK( magma_index_malloc( &B->drow, r_blocks + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.numblocks ));
            // data transfer
            magma_csetvector( size_b*size_b*A.numblocks, A.val, 1, B->dval, 1 );
            magma_index_setvector( r_blocks+1, A.row, 1, B->drow, 1 );
            magma_index_setvector( A.numblocks, A.col, 1, B->dcol, 1 );
        }
        //DENSE-type
        if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows*A.num_cols ));
            // data transfer
            magma_csetvector( A.num_rows*A.num_cols, A.val, 1, B->dval, 1 );
        }
    }

    // second case: copy matrix from host to host
    if ( src == Magma_CPU && dst == Magma_CPU ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.num_rows+1; i++ ) {
                B->row[i] = A.row[i];
            }
        }
        //CSRCOO-type
        if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            *B = A;
            B->memory_location = Magma_CPU;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz ));
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
                B->rowidx[i] = A.rowidx[i];
            }
            for( magma_int_t i=0; i<A.num_rows+1; i++ ) {
                B->row[i] = A.row[i];
            }
        }
        //ELL/ELLPACKT-type
        if ( A.storage_type == Magma_ELLPACKT || A.storage_type == Magma_ELL ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        }
        //ELLD-type
        if ( A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        }
        //ELLRT-type
        if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int threads_per_row = A.alignment;
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                    /threads_per_row) ) * threads_per_row;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, rowlength*A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->col, rowlength*A.num_rows ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*rowlength; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.num_rows; i++ ) {
                B->row[i] = A.row[i];
            }
        }
        //SELLP-type
        if (  A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            B->numblocks = A.numblocks;

            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.numblocks+1 ));
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.numblocks+1; i++ ) {
                B->row[i] = A.row[i];
            }
        }
        //DENSE-type
        if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.num_rows*A.num_cols ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.num_cols; i++ ) {
                B->val[i] = A.val[i];
            }
        }
    }

    // third case: copy matrix from device to host
    if ( src == Magma_DEV && dst == Magma_CPU ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            // data transfer
            magma_cgetvector( A.nnz, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows+1, A.drow, 1, B->row, 1 );
            magma_index_getvector( A.nnz, A.dcol, 1, B->col, 1 );
        }
        //CSRCOO-type
        if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            *B = A;
            B->memory_location = Magma_CPU;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz ));
            // data transfer
            magma_cgetvector( A.nnz, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows+1, A.drow, 1, B->row, 1 );
            magma_index_getvector( A.nnz, A.dcol, 1, B->col, 1 );
            magma_index_getvector( A.nnz, A.drowidx, 1, B->rowidx, 1 );
        }
        //ELL/ELLPACKT-type
        if ( A.storage_type == Magma_ELL || A.storage_type == Magma_ELLPACKT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row ));
            // data transfer
            magma_cgetvector( A.num_rows*A.max_nnz_row, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows*A.max_nnz_row, A.dcol, 1, B->col, 1 );
        }
        //ELLD-type
        if (  A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.num_rows*A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows*A.max_nnz_row ));
            // data transfer
            magma_cgetvector( A.num_rows*A.max_nnz_row, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows*A.max_nnz_row, A.dcol, 1, B->col, 1 );
        }
        //ELLRT-type
        if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int threads_per_row = A.alignment;
            // memory allocation
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                /threads_per_row) ) * threads_per_row;
            CHECK( magma_cmalloc_cpu( &B->val, rowlength*A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->col, rowlength*A.num_rows ));
            // data transfer
            magma_cgetvector( A.num_rows*rowlength, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows*rowlength, A.dcol, 1, B->col, 1 );
            magma_index_getvector( A.num_rows, A.drow, 1, B->row, 1 );
        }
        //SELLP-type
        if ( A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->alignment = A.alignment;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;

            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.numblocks+1 ));
            // data transfer
            magma_cgetvector( A.nnz, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.nnz, A.dcol, 1, B->col, 1 );
            magma_index_getvector( A.numblocks+1, A.drow, 1, B->row, 1 );
        }
        //BCSR-type
        if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.numblocks*A.blocksize*A.blocksize ));
            CHECK( magma_index_malloc_cpu( &B->row, r_blocks+1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.numblocks ));
            // data transfer
            magma_cgetvector( A.numblocks * A.blocksize * A.blocksize, A.dval, 1, B->val, 1 );
            magma_index_getvector( r_blocks+1, A.drow, 1, B->row, 1 );
            magma_index_getvector( A.numblocks, A.dcol, 1, B->col, 1 );
        }
        //DENSE-type
        if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_CPU;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_cmalloc_cpu( &B->val, A.num_rows*A.num_cols ));
            // data transfer
            magma_cgetvector( A.num_rows*A.num_cols, A.dval, 1, B->val, 1 );
        }
    }

    // fourth case: copy matrix from device to device
    if ( src == Magma_DEV && dst == Magma_DEV ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            // data transfer
            magma_ccopyvector( A.nnz, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( (A.num_rows+1), A.drow, 1, B->drow, 1 );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1 );
        }
        //CSRCOO-type
        if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            *B = A;
            B->memory_location = Magma_DEV;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drowidx, A.nnz ));
            // data transfer
            magma_ccopyvector( A.nnz, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( (A.num_rows+1), A.drow, 1, B->drow, 1 );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1 );
            magma_index_copyvector( A.nnz, A.drowidx, 1, B->drowidx, 1 );
        }
        //ELL/ELLPACKT-type
        if ( A.storage_type == Magma_ELL || A.storage_type == Magma_ELLPACKT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_ccopyvector( A.num_rows*A.max_nnz_row, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows*A.max_nnz_row, A.dcol, 1, B->dcol, 1 );
        }
        //ELLD-type
        if ( A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_ccopyvector( A.num_rows*A.max_nnz_row, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows*A.max_nnz_row, A.dcol, 1, B->dcol, 1 );
        }
        //ELLRT-type
        if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            int threads_per_row = A.alignment;
            int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
                                    /threads_per_row) ) * threads_per_row;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows ));
            // data transfer
            magma_ccopyvector( A.num_rows * rowlength, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows * rowlength, A.dcol, 1, B->dcol, 1 );
            magma_index_copyvector( A.num_rows, A.drow, 1, B->drow, 1 );
        }
        //SELLP-type
        if ( A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.numblocks + 1 ));
            // data transfer
            magma_ccopyvector( A.nnz, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.nnz,         A.dcol, 1, B->dcol, 1 );
            magma_index_copyvector( A.numblocks+1, A.drow, 1, B->drow, 1 );
        }
        //BCSR-type
        if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, size_b*size_b*A.numblocks ));
            CHECK( magma_index_malloc( &B->drow, r_blocks + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.numblocks ));
            // data transfer
            magma_ccopyvector( size_b*size_b*A.numblocks, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( (r_blocks+1), A.drow, 1, B->drow, 1 );
            magma_index_copyvector( A.numblocks, A.dcol, 1, B->dcol, 1 );
        }
        //DENSE-type
        if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->diagorder_type = A.diagorder_type;
            B->memory_location = Magma_DEV;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->fill_mode = A.fill_mode;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_cmalloc( &B->dval, A.num_rows*A.num_cols ));
            // data transfer
            magma_ccopyvector( A.num_rows*A.num_cols, A.dval, 1, B->dval, 1 );
        }
    }
    
    
cleanup:
    if( info != 0 ){
        magma_cmfree( B, queue );
    }
    magmablasSetKernelStream( orig_queue );
    return info;
}

