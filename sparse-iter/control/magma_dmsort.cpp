/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_zmsort.cpp normal z -> d, Fri Jul 18 17:34:30 2014
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

#include "magmasparse_d.h"
#include "magma.h"
#include "mmio.h"


using namespace std;



/**
    Purpose
    -------

    Sorts columns and rows for a matrix in COO or CSRCOO format. 

    Arguments
    ---------

    @param
    A           magma_d_sparse_matrix*
                matrix in magma sparse matrix format

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t 
magma_dmsort( magma_d_sparse_matrix *A ){

    
    if( (A->storage_type == Magma_CSRCOO 
        || A->storage_type == Magma_COO) 
        && A->memory_location == Magma_CPU ){ 

        double tv;
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

        magma_d_sparse_matrix hA, CSRCOOA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_d_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_d_mconvert( hA, &CSRCOOA, hA.storage_type, Magma_CSRCOO );

        magma_dmsort( &CSRCOOA );

        magma_d_mfree( &hA );
        magma_d_mfree( A );
        magma_d_mconvert( CSRCOOA, &hA, Magma_CSRCOO, A_storage );
        magma_d_mtransfer( hA, A, Magma_CPU, A_location );
        magma_d_mfree( &hA );
        magma_d_mfree( &CSRCOOA );    

        return MAGMA_SUCCESS; 
    }
}
