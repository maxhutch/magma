/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_z_vio.cpp normal z -> c, Sat Nov 15 19:54:23 2014
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

    @param[in]
    x           magma_c_vector
                vector to visualize

    @param[in]
    offset      magma_int_t
                start inex of visualization

    @param[in]
    visulen     magma_int_t
                number of entries to visualize       

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_c_vvisu(
    magma_c_vector x, 
    magma_int_t offset, 
    magma_int_t  visulen,
    magma_queue_t queue )
{
    printf("visualize entries %d - %d of vector ", 
                    (int) offset, (int) (offset + visulen) );
    fflush(stdout);  
    if ( x.memory_location == Magma_CPU ) {
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset + visulen; i++ )
            printf("%5.2f\n", MAGMA_C_REAL(x.val[i]));
    return MAGMA_SUCCESS;
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        magma_c_vector y;
        magma_c_vtransfer( x, &y, Magma_DEV, Magma_CPU, queue );
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            printf("%5.2f\n", MAGMA_C_REAL(y.val[i]));
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

    @param[out]
    x           magma_c_vector *
                vector to read in

    @param[in]
    length      magma_int_t
                length of vector
    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_c_vread(
    magma_c_vector *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue )
{
    x->memory_location = Magma_CPU;
    x->num_rows = length;
    x->num_cols = 1;
    x->major = MagmaColMajor;
    magma_cmalloc_cpu( &x->val, length );
    magma_int_t nnz=0, i=0;
    string line;
    ifstream fin(filename);  
    getline(fin, line, '\n');  
    while( i<length )  // eof() is 'true' at the end of data
    {
        getline(fin, line, '\n');
        if ( magma_cstring_to_float(line) != 0 )
            nnz++;
        x->val[i] = MAGMA_C_MAKE(magma_cstring_to_float(line), 0.0);
        i++;
    }
    fin.close();
    x->nnz = nnz;
    return MAGMA_SUCCESS;
}   




/**
    Purpose
    -------

    Reads in a sparse vector-block stored in COO format.

    Arguments
    ---------

    @param[out]
    x           magma_c_vector *
                vector to read in

    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_c_vspread(
    magma_c_vector *x, 
    const char * filename,
    magma_queue_t queue )
{
    magma_c_sparse_matrix A,B;
    magma_int_t entry=0;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    magma_c_csr_mtx( &A,  filename, queue  ); 
    magma_c_mconvert( A, &B, Magma_CSR, Magma_DENSE, queue );
    magma_c_vinit( x, Magma_CPU, A.num_cols*A.num_rows, MAGMA_C_ZERO, queue );
    x->major = MagmaRowMajor;
    for(magma_int_t i=0; i<A.num_cols; i++) {
        for(magma_int_t j=0; j<A.num_rows; j++) {
            x->val[i*A.num_rows+j] = B.val[ i+j*A.num_cols ];
            entry++;     
        }
    }
    x->num_rows = A.num_rows;
    x->num_cols = A.num_cols;

    magma_c_mfree( &A, queue );
    magma_c_mfree( &B, queue );

    return MAGMA_SUCCESS;
}   


