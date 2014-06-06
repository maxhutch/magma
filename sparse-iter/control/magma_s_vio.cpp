/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_z_vio.cpp normal z -> s, Fri May 30 10:41:45 2014
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


using namespace std;


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Visualizes part of a vector of type magma_s_vector.
    With input vector x , offset, displaylength the entries 
    offset - (offset + displaylength) of x are visualized.

    Arguments
    =========

    magma_s_vector x                     vector to visualize
    magma_int_t offset                   start inex of visualization
    magma_int_t displaylength            number of entries to visualize       

    ========================================================================  */

extern "C"
magma_int_t
magma_s_vvisu(      magma_s_vector x, 
                    magma_int_t offset, 
                    magma_int_t displaylength ){

    printf("visualize entries %d - %d of vector ", 
                    (int) offset, (int) (offset+displaylength) );
    fflush(stdout);  
    if( x.memory_location == Magma_CPU ){
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset+displaylength; i++ )
            printf("%f\n", MAGMA_S_REAL(x.val[i]));
    return MAGMA_SUCCESS;
    }
    else if( x.memory_location == Magma_DEV ){
        printf("located on DEV:\n");
        magma_s_vector y;
        magma_s_vtransfer( x, &y, Magma_DEV, Magma_CPU);
        for( magma_int_t i=offset; i<offset+displaylength; i++ )
            printf("%f\n", MAGMA_S_REAL(y.val[i]));
    free(y.val);
    return MAGMA_SUCCESS;
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

    Reads in a float vector of length "length".

    Arguments
    =========

    magma_s_vector x                     vector to read in
    magma_int_t length                   length of vector
    char filename                        file where vector is stored

    ========================================================================  */


// small helper function
extern "C"
float magma_sstring_to_float( const std::string& s )
{
    std::istringstream i(s);
    float x;
    if (!(i >> x))
        return 0;
    return x;
} 



extern "C"
magma_int_t
magma_s_vread(      magma_s_vector *x, 
                    magma_int_t length,
                    char * filename ){
    
    x->memory_location = Magma_CPU;
    x->num_rows = length;
    
    magma_int_t nnz=0, i=0;
    string line;
    ifstream fin(filename);  
    getline(fin, line, '\n');  
    while(!(fin.eof()))  // eof() is 'true' at the end of data
    {
        getline(fin, line, '\n');
        if( magma_sstring_to_float(line) != 0 )
            nnz++;
        x->val[i] = MAGMA_S_MAKE(magma_sstring_to_float(line), 0.0);
        i++;
    }
    fin.close();
    x->nnz = nnz;
    return MAGMA_SUCCESS;
}   



