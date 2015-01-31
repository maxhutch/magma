/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zmdiff.cpp normal z -> s, Fri Jan 30 19:00:32 2015
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
#include "magmasparse_s.h"
#include "magma.h"
#include "mmio.h"


#define THRESHOLD 10e-99

using namespace std;


/**
    Purpose
    -------

    Computes the Frobenius norm of the difference between the CSR matrices A 
    and B. They do not need to share the same sparsity pattern!
        
            res = ||A-B||_F = sqrt( sum_ij (A_ij-B_ij)^2 )


    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                sparse matrix in CSR

    @param[in]
    B           magma_s_sparse_matrix
                sparse matrix in CSR    
                
    @param[out]
    res         real_Double_t* 
                residual 
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smdiff(
    magma_s_sparse_matrix A, magma_s_sparse_matrix B, 
    real_Double_t *res,
    magma_queue_t queue )
{
    
    if( A.memory_location == Magma_CPU && B.memory_location == Magma_CPU
            && A.storage_type == Magma_CSR && B.storage_type == Magma_CSR ){
        real_Double_t tmp2;
        magma_int_t i,j,k;
        *res = 0.0;
        
        for(i=0; i<A.num_rows; i++) {
            for(j=A.row[i]; j<A.row[i+1]; j++) {
                magma_index_t localcol = A.col[j];
                for( k=B.row[i]; k<B.row[i+1]; k++) {
                    if (B.col[k] == localcol) {
                        tmp2 = (real_Double_t) fabs( MAGMA_S_REAL(A.val[j] )
                                                        - MAGMA_S_REAL(B.val[k]) );
    
                        (*res) = (*res) + tmp2* tmp2;
                    }
                }
            }      
        }

        (*res) =  sqrt((*res));

        return MAGMA_SUCCESS;
    }
    else{
        printf("error: mdiff only supported for CSR matrices on the CPU.\n");
        return MAGMA_ERR_NOT_SUPPORTED;
    }
}

