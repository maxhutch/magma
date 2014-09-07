/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from magma_zmdiff.cpp normal z -> c, Tue Sep  2 12:38:36 2014
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
#include "magmasparse_c.h"
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

    @param
    A           magma_c_sparse_matrix
                sparse matrix in CSR

    @param
    B           magma_c_sparse_matrix
                sparse matrix in CSR    
                
    @param
    res         real_Double_t* 
                residual 

    @ingroup magmasparse_caux
    ********************************************************************/

magma_int_t 
magma_cmdiff( magma_c_sparse_matrix A, magma_c_sparse_matrix B, 
                  real_Double_t *res ){

    real_Double_t tmp2;
    magma_int_t i,j,k;
    *res = 0.0;
    
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t localcol = A.col[j];
            for( k=B.row[i]; k<B.row[i+1]; k++){
                if(B.col[k] == localcol){
                    tmp2 = (real_Double_t) fabs( MAGMA_C_REAL(A.val[j] )
                                                    - MAGMA_C_REAL(B.val[k]) );

                    (*res) = (*res) + tmp2* tmp2;
                }
            }
        }      
    }

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS; 
}

