/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

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


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Visualizes part of a vector of type magma_z_vector.
    With input vector x , offset, displaylength the entries 
    offset - (offset + displaylength) of x are visualized.

    Arguments
    =========

    magma_z_vector x                     vector to visualize
    magma_int_t offset                   start inex of visualization
    magma_int_t displaylength            number of entries to visualize       

    ========================================================================  */

extern "C"
magma_int_t
magma_z_vvisu(      magma_z_vector x, 
                    magma_int_t offset, 
                    magma_int_t displaylength ){

    printf("visualize entries %d - %d of vector ", 
                    (int) offset, (int) (offset+displaylength) );
    fflush(stdout);  
    if( x.memory_location == Magma_CPU ){
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset+displaylength; i++ )
            printf("%f\n", MAGMA_Z_REAL(x.val[i]));
    return MAGMA_SUCCESS;
    }
    else if( x.memory_location == Magma_DEV ){
        printf("located on DEV:\n");
        magma_z_vector y;
        magma_z_vtransfer( x, &y, Magma_DEV, Magma_CPU);
        for( magma_int_t i=offset; i<offset+displaylength; i++ )
            printf("%f\n", MAGMA_Z_REAL(y.val[i]));
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

    Reads in a double vector of length "length".

    Arguments
    =========

    magma_z_vector x                     vector to read in
    magma_int_t length                   length of vector
    char filename                        file where vector is stored

    ========================================================================  */


// small helper function
extern "C"
double magma_zstring_to_double( const std::string& s )
{
    std::istringstream i(s);
    double x;
    if (!(i >> x))
        return 0;
    return x;
} 



extern "C"
magma_int_t
magma_z_vread(      magma_z_vector *x, 
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
        if( magma_zstring_to_double(line) != 0 )
            nnz++;
        x->val[i] = MAGMA_Z_MAKE(magma_zstring_to_double(line), 0.0);
        i++;
    }
    fin.close();
    x->nnz = nnz;
    return MAGMA_SUCCESS;
}   



