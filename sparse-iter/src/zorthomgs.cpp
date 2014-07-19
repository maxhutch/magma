/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define  X(i)     (X + (i)*num_rows)

/**
    Purpose
    -------

    This routine orthogonalizes a set of vectors stored in a n x m - matrix X
    in column major:

        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    X = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[1] x2[2] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /

    This routine performs a modified Gram-Schmidt orthogonalization.
    
    Arguments
    ---------

    @param
    num_rows    magma_int_t
                number of rows

    @param
    num_vecs    magma_int_t
                number of vectors

    @param
    X           magmaDoubleComplex*
                input/output vector-block/matrix X


    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zorthomgs(        magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magmaDoubleComplex *X ){

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);

    //start to orthogonalize the second, then proceed to the right
    for( magma_int_t i=0; i<num_vecs; i++){  
        if( i==0 ){
            // normalize first vector to norm 1
            magmaDoubleComplex nrm = MAGMA_Z_MAKE(
                               1.0 / magma_dznrm2( num_rows, X(i), 1 ), 0.0 );
            magma_zscal( num_rows, nrm , X(i), 1 );   
        }
        else{
            // normalize vector i to norm 1
            magmaDoubleComplex nrm = MAGMA_Z_MAKE(
                               1.0 / magma_dznrm2( num_rows, X(i), 1 ), 0.0 );
            magma_zscal( num_rows, nrm , X(i), 1 );  

            // orthogonalize against the vectors right of i
            for (magma_int_t j=0; j<i; j++) {
                magmaDoubleComplex dot = 
                    magma_zdotc( num_rows, X(j), 1, X(i), 1);    
                    // compute X(i) = X(i) - <X(j),X(i)> * X(j)           
                    magma_zaxpy(num_rows,-dot, X(j), 1, X(i), 1);            
            }
            // normalize vector i to norm 1
            nrm = MAGMA_Z_MAKE(
                               1.0 / magma_dznrm2( num_rows, X(i), 1 ), 0.0 );
            magma_zscal( num_rows, nrm , X(i), 1 );  

            // orthogonalize against the vectors right of i
            for (magma_int_t j=0; j<i; j++) {
                magmaDoubleComplex dot = 
                    magma_zdotc( num_rows, X(j), 1, X(i), 1);    
                    // compute X(i) = X(i) - <X(j),X(i)> * X(j)           
                    magma_zaxpy(num_rows,-dot, X(j), 1, X(i), 1);            
            }
            // normalize vector i to norm 1
            nrm = MAGMA_Z_MAKE(
                               1.0 / magma_dznrm2( num_rows, X(i), 1 ), 0.0 );
            magma_zscal( num_rows, nrm , X(i), 1 );   
        }
    }

    return MAGMA_SUCCESS;
}


