/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from magma_zmtransfer.cpp normal z -> s, Fri Sep 11 18:29:46 2015
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
    A           magma_s_matrix
                sparse matrix A

    @param[out]
    B           magma_s_matrix*
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

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smtransfer(
    magma_s_matrix A,
    magma_s_matrix *B,
    magma_location_t src,
    magma_location_t dst,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    B->val = NULL;
    B->diag = NULL;
    B->row = NULL;
    B->rowidx = NULL;
    B->col = NULL;
    B->blockinfo = NULL;
    B->dval = NULL;
    B->ddiag = NULL;
    B->drow = NULL;
    B->drowidx = NULL;
    B->dcol = NULL;

    // first case: copy matrix from host to device
    if ( src == Magma_CPU && dst == Magma_DEV ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            // data transfer
            magma_ssetvector( A.nnz, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows + 1, A.row, 1, B->drow, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->dcol, 1 );
        }
        //CSRCOO-type
        else if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drowidx, A.nnz ));
            // data transfer
            magma_ssetvector( A.nnz, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows + 1, A.row, 1, B->drow, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->dcol, 1 );
            magma_index_setvector( A.nnz, A.rowidx, 1, B->drowidx, 1 );
        }
        //ELL/ELLPACKT-type
        else if ( A.storage_type == Magma_ELLPACKT || A.storage_type == Magma_ELL ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_ssetvector( A.num_rows * A.max_nnz_row, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows * A.max_nnz_row, A.col, 1, B->dcol, 1 );
        }
        //ELLD-type
        else if ( A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_ssetvector( A.num_rows * A.max_nnz_row, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows * A.max_nnz_row, A.col, 1, B->dcol, 1 );
        }
        //ELLRT-type
        else if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            magma_int_t rowlength = magma_roundup( A.max_nnz_row, A.alignment );
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows ));
            // data transfer
            magma_ssetvector( A.num_rows * rowlength, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.num_rows * rowlength, A.col, 1, B->dcol, 1 );
            magma_index_setvector( A.num_rows, A.row, 1, B->drow, 1 );
        }
        //SELLP-type
        else if ( A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.numblocks + 1 ));
            // data transfer
            magma_ssetvector( A.nnz, A.val, 1, B->dval, 1 );
            magma_index_setvector( A.nnz, A.col, 1, B->dcol, 1 );
            magma_index_setvector( A.numblocks + 1, A.row, 1, B->drow, 1 );
        }
        //BCSR-type
        else if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            //magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
            magma_int_t r_blocks = magma_ceildiv( A.num_rows, size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_smalloc( &B->dval, size_b * size_b * A.numblocks ));
            CHECK( magma_index_malloc( &B->drow, r_blocks + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.numblocks ));
            // data transfer
            magma_ssetvector( size_b * size_b * A.numblocks, A.val, 1, B->dval, 1 );
            magma_index_setvector( r_blocks + 1, A.row, 1, B->drow, 1 );
            magma_index_setvector( A.numblocks, A.col, 1, B->dcol, 1 );
        }
        //DENSE-type
        else if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->major = A.major;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * A.num_cols ));
            // data transfer
            magma_ssetvector( A.num_rows * A.num_cols, A.val, 1, B->dval, 1 );
        }
    }

    // second case: copy matrix from host to host
    else if ( src == Magma_CPU && dst == Magma_CPU ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows + 1 ));
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
        else if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows + 1 ));
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
        else if ( A.storage_type == Magma_ELLPACKT || A.storage_type == Magma_ELL ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows * A.max_nnz_row ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        }
        //ELLD-type
        else if ( A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows * A.max_nnz_row ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.max_nnz_row; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
        }
        //ELLRT-type
        else if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            //int threads_per_row = A.alignment;
            //int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
            //                        /threads_per_row) ) * threads_per_row;
            magma_int_t rowlength = magma_roundup( A.max_nnz_row, A.alignment );
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, rowlength * A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->col, rowlength * A.num_rows ));
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
        else if (  A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            B->numblocks = A.numblocks;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.numblocks + 1 ));
            // data transfer
            for( magma_int_t i=0; i<A.nnz; i++ ) {
                B->val[i] = A.val[i];
                B->col[i] = A.col[i];
            }
            for( magma_int_t i=0; i<A.numblocks+1; i++ ) {
                B->row[i] = A.row[i];
            }
        }
        //BCSR-type
        else if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            //magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
            magma_int_t r_blocks = magma_ceildiv( A.num_rows, size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, size_b * size_b * A.numblocks ));
            CHECK( magma_index_malloc_cpu( &B->row, r_blocks + 1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.numblocks ));
            // data transfer
            //magma_ssetvector( size_b * size_b * A.numblocks, A.val, 1, B->dval, 1 );
            for( magma_int_t i=0; i<size_b*size_b*A.numblocks; i++ ) {
                B->dval[i] = A.val[i];
            }
            //magma_index_setvector( r_blocks + 1, A.row, 1, B->drow, 1 );
            for( magma_int_t i=0; i<r_blocks+1; i++ ) {
                B->drow[i] = A.row[i];
            }
            //magma_index_setvector( A.numblocks, A.col, 1, B->dcol, 1 );
            for( magma_int_t i=0; i<A.numblocks; i++ ) {
                B->dcol[i] = A.col[i];
            }
        }
        //DENSE-type
        else if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->major = A.major;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.num_rows * A.num_cols ));
            // data transfer
            for( magma_int_t i=0; i<A.num_rows*A.num_cols; i++ ) {
                B->val[i] = A.val[i];
            }
        }
    }

    // third case: copy matrix from device to host
    else if ( src == Magma_DEV && dst == Magma_CPU ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows + 1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            // data transfer
            magma_sgetvector( A.nnz, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows + 1, A.drow, 1, B->row, 1 );
            magma_index_getvector( A.nnz, A.dcol, 1, B->col, 1 );
        }
        //CSRCOO-type
        else if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows + 1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz ));
            // data transfer
            magma_sgetvector( A.nnz, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows + 1, A.drow, 1, B->row, 1 );
            magma_index_getvector( A.nnz, A.dcol, 1, B->col, 1 );
            magma_index_getvector( A.nnz, A.drowidx, 1, B->rowidx, 1 );
        }
        //ELL/ELLPACKT-type
        else if ( A.storage_type == Magma_ELLPACKT || A.storage_type == Magma_ELL ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_sgetvector( A.num_rows * A.max_nnz_row, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows * A.max_nnz_row, A.dcol, 1, B->col, 1 );
        }
        //ELLD-type
        else if (  A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc_cpu( &B->col, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_sgetvector( A.num_rows * A.max_nnz_row, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows * A.max_nnz_row, A.dcol, 1, B->col, 1 );
        }
        //ELLRT-type
        else if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            //int threads_per_row = A.alignment;
            //int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
            //                    /threads_per_row) ) * threads_per_row;
            magma_int_t rowlength = magma_roundup( A.max_nnz_row, A.alignment );
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, rowlength * A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->row, A.num_rows ));
            CHECK( magma_index_malloc_cpu( &B->col, rowlength * A.num_rows ));
            // data transfer
            magma_sgetvector( A.num_rows * rowlength, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.num_rows * rowlength, A.dcol, 1, B->col, 1 );
            magma_index_getvector( A.num_rows, A.drow, 1, B->row, 1 );
        }
        //SELLP-type
        else if ( A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
            CHECK( magma_index_malloc_cpu( &B->row, A.numblocks + 1 ));
            // data transfer
            magma_sgetvector( A.nnz, A.dval, 1, B->val, 1 );
            magma_index_getvector( A.nnz, A.dcol, 1, B->col, 1 );
            magma_index_getvector( A.numblocks + 1, A.drow, 1, B->row, 1 );
        }
        //BCSR-type
        else if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            //magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
            magma_int_t r_blocks = magma_ceildiv( A.num_rows, size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, size_b * size_b * A.numblocks ));
            CHECK( magma_index_malloc_cpu( &B->row, r_blocks + 1 ));
            CHECK( magma_index_malloc_cpu( &B->col, A.numblocks ));
            // data transfer
            magma_sgetvector( size_b * size_b * A.numblocks, A.dval, 1, B->val, 1 );
            magma_index_getvector( r_blocks + 1, A.drow, 1, B->row, 1 );
            magma_index_getvector( A.numblocks, A.dcol, 1, B->col, 1 );
        }
        //DENSE-type
        else if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_CPU;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->major = A.major;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_smalloc_cpu( &B->val, A.num_rows * A.num_cols ));
            // data transfer
            magma_sgetvector( A.num_rows * A.num_cols, A.dval, 1, B->val, 1 );
        }
    }

    // fourth case: copy matrix from device to device
    else if ( src == Magma_DEV && dst == Magma_DEV ) {
        //CSR-type
        if ( A.storage_type == Magma_CSR || A.storage_type == Magma_CSC
                                        || A.storage_type == Magma_CSRD
                                        || A.storage_type == Magma_CSRL
                                        || A.storage_type == Magma_CSRU ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            // data transfer
            magma_scopyvector( A.nnz, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows + 1, A.drow, 1, B->drow, 1 );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1 );
        }
        //CSRCOO-type
        else if ( A.storage_type == Magma_CSRCOO ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drowidx, A.nnz ));
            // data transfer
            magma_scopyvector( A.nnz, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows + 1, A.drow, 1, B->drow, 1 );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1 );
            magma_index_copyvector( A.nnz, A.drowidx, 1, B->drowidx, 1 );
        }
        //ELL/ELLPACKT-type
        else if ( A.storage_type == Magma_ELLPACKT || A.storage_type == Magma_ELL ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_scopyvector( A.num_rows * A.max_nnz_row, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows * A.max_nnz_row, A.dcol, 1, B->dcol, 1 );
        }
        //ELLD-type
        else if ( A.storage_type == Magma_ELLD ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * A.max_nnz_row ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * A.max_nnz_row ));
            // data transfer
            magma_scopyvector( A.num_rows * A.max_nnz_row, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows * A.max_nnz_row, A.dcol, 1, B->dcol, 1 );
        }
        //ELLRT-type
        else if ( A.storage_type == Magma_ELLRT ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->alignment = A.alignment;
            //int threads_per_row = A.alignment;
            //int rowlength = ( (int)((A.max_nnz_row+threads_per_row-1)
            //                        /threads_per_row) ) * threads_per_row;
            magma_int_t rowlength = magma_roundup( A.max_nnz_row, A.alignment );
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->dcol, A.num_rows * rowlength ));
            CHECK( magma_index_malloc( &B->drow, A.num_rows ));
            // data transfer
            magma_scopyvector( A.num_rows * rowlength, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.num_rows * rowlength, A.dcol, 1, B->dcol, 1 );
            magma_index_copyvector( A.num_rows, A.drow, 1, B->drow, 1 );
        }
        //SELLP-type
        else if ( A.storage_type == Magma_SELLP ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.nnz ));
            CHECK( magma_index_malloc( &B->dcol, A.nnz ));
            CHECK( magma_index_malloc( &B->drow, A.numblocks + 1 ));
            // data transfer
            magma_scopyvector( A.nnz, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1 );
            magma_index_copyvector( A.numblocks + 1, A.drow, 1, B->drow, 1 );
        }
        //BCSR-type
        else if ( A.storage_type == Magma_BCSR ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->blocksize = A.blocksize;
            B->numblocks = A.numblocks;
            B->alignment = A.alignment;
            magma_int_t size_b = A.blocksize;
            //magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );
                    // max number of blocks per row
            //magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );
            magma_int_t r_blocks = magma_ceildiv( A.num_rows, size_b );
                    // max number of blocks per column
            // memory allocation
            CHECK( magma_smalloc( &B->dval, size_b * size_b * A.numblocks ));
            CHECK( magma_index_malloc( &B->drow, r_blocks + 1 ));
            CHECK( magma_index_malloc( &B->dcol, A.numblocks ));
            // data transfer
            magma_scopyvector( size_b * size_b * A.numblocks, A.dval, 1, B->dval, 1 );
            magma_index_copyvector( r_blocks + 1, A.drow, 1, B->drow, 1 );
            magma_index_copyvector( A.numblocks, A.dcol, 1, B->dcol, 1 );
        }
        //DENSE-type
        else if ( A.storage_type == Magma_DENSE ) {
            // fill in information for B
            B->storage_type = A.storage_type;
            B->memory_location = Magma_DEV;
            B->sym = A.sym;
            B->diagorder_type = A.diagorder_type;
            B->fill_mode = A.fill_mode;
            B->num_rows = A.num_rows;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            B->major = A.major;
            B->ld = A.ld;
            // memory allocation
            CHECK( magma_smalloc( &B->dval, A.num_rows * A.num_cols ));
            // data transfer
            magma_scopyvector( A.num_rows * A.num_cols, A.dval, 1, B->dval, 1 );
        }
    }
    
    
cleanup:
    if( info != 0 ){
        magma_smfree( B, queue );
    }
    magmablasSetKernelStream( orig_queue );
    return info;
}
