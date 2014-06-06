/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @precisions normal z -> s d c
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

#include "magmasparse_z.h"
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

    Sorts columns and rows for a matrix in COO or CSRCOO format. 

    Arguments
    =========

    magma_z_sparse_matrix *A             matrix in magma sparse matrix format

    ========================================================================  */

extern "C"
magma_int_t 
magma_zmsort( magma_z_sparse_matrix *A ){

    
    if( (A->storage_type == Magma_CSRCOO 
        || A->storage_type == Magma_COO) 
        && A->memory_location == Magma_CPU ){ 

        magmaDoubleComplex tv;
        magma_index_t ti;
        std::cout << "Sorting the cols...." << std::endl;
        // bubble sort (by cols)
        for (magma_index_t i=0; i<A->nnz-1; ++i)
        for (magma_index_t j=0; j<A->nnz-i-1; ++j)
            if (A->col[j] > A->col[j+1] ){

                ti = A->col[j];
                A->col[j] = A->col[j+1];
                A->col[j+1] = ti;

                ti = A->rowidx[j];
                A->rowidx[j] = A->rowidx[j+1];
                A->rowidx[j+1] = ti;

                tv = A->val[j];
                A->val[j] = A->val[j+1];
                A->val[j+1] = tv;

          }

        std::cout << "Sorting the rows...." << std::endl;
        // bubble sort (by rows)
        for (magma_index_t i=0; i<A->nnz-1; ++i)
        for (magma_index_t j=0; j<A->nnz-i-1; ++j)
        if ( A->rowidx[j] > A->rowidx[j+1] ){

            ti = A->col[j];
            A->col[j] = A->col[j+1];
            A->col[j+1] = ti;

            ti = A->rowidx[j];
            A->rowidx[j] = A->rowidx[j+1];
            A->rowidx[j+1] = ti;

            tv = A->val[j];
            A->val[j] = A->val[j+1];
            A->val[j+1] = tv;

        }
        std::cout << "Sorting: done" << std::endl;

        return MAGMA_SUCCESS;
    }
    else{

        magma_z_sparse_matrix hA, CSRCOOA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_z_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRCOOA, hA.storage_type, Magma_CSRCOO );

        magma_zmsort( &CSRCOOA );

        magma_z_mfree( &hA );
        magma_z_mfree( A );
        magma_z_mconvert( CSRCOOA, &hA, Magma_CSRCOO, A_storage );
        magma_z_mtransfer( hA, A, Magma_CPU, A_location );
        magma_z_mfree( &hA );
        magma_z_mfree( &CSRCOOA );    

        return MAGMA_SUCCESS; 
    }
}
