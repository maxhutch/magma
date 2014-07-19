/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_z_vio.cpp normal z -> c, Fri Jul 18 17:34:30 2014
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


using namespace std;


/**
    Purpose
    -------

    Visualizes part of a vector of type magma_c_vector.
    With input vector x , offset, visulen, the entries 
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param
    x           magma_c_vector
                vector to visualize

    @param
    offset      magma_int_t
                start inex of visualization

    @param
    visulen     magma_int_t
                number of entries to visualize       


    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_c_vvisu(      magma_c_vector x, 
                    magma_int_t offset, 
                    magma_int_t  visulen ){

    printf("visualize entries %d - %d of vector ", 
                    (int) offset, (int) (offset + visulen) );
    fflush(stdout);  
    if( x.memory_location == Magma_CPU ){
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset + visulen; i++ )
            printf("%f\n", MAGMA_C_REAL(x.val[i]));
    return MAGMA_SUCCESS;
    }
    else if( x.memory_location == Magma_DEV ){
        printf("located on DEV:\n");
        magma_c_vector y;
        magma_c_vtransfer( x, &y, Magma_DEV, Magma_CPU);
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            printf("%f\n", MAGMA_C_REAL(y.val[i]));
    free(y.val);
    return MAGMA_SUCCESS;
    }
    return MAGMA_SUCCESS; 
}   




// small helper function
extern "C"
float magma_cstring_to_float( const std::string& s )
{
    std::istringstream i(s);
    float x;
    if (!(i >> x))
        return 0;
    return x;
} 



/**
    Purpose
    -------

    Reads in a float vector of length "length".

    Arguments
    ---------

    @param
    x           magma_c_vector
                vector to read in

    @param
    length      magma_int_t
                length of vector
    @param
    filename    char*
                file where vector is stored

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_c_vread(      magma_c_vector *x, 
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
        if( magma_cstring_to_float(line) != 0 )
            nnz++;
        x->val[i] = MAGMA_C_MAKE(magma_cstring_to_float(line), 0.0);
        i++;
    }
    fin.close();
    x->nnz = nnz;
    return MAGMA_SUCCESS;
}   



