/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_z_vio.cpp normal z -> d, Sat Nov 15 19:54:23 2014
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

#include "magmasparse_d.h"
#include "magma.h"
#include "mmio.h"


using namespace std;


/**
    Purpose
    -------

    Visualizes part of a vector of type magma_d_vector.
    With input vector x , offset, visulen, the entries 
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param[in]
    x           magma_d_vector
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

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_d_vvisu(
    magma_d_vector x, 
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
            printf("%5.2f\n", MAGMA_D_REAL(x.val[i]));
    return MAGMA_SUCCESS;
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        magma_d_vector y;
        magma_d_vtransfer( x, &y, Magma_DEV, Magma_CPU, queue );
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            printf("%5.2f\n", MAGMA_D_REAL(y.val[i]));
    free(y.val);
    return MAGMA_SUCCESS;
    }
    return MAGMA_SUCCESS;
}   




// small helper function
extern "C"
double magma_dstring_to_double( const std::string& s )
{
    std::istringstream i(s);
    double x;
    if (!(i >> x))
        return 0;
    return x;
} 



/**
    Purpose
    -------

    Reads in a double vector of length "length".

    Arguments
    ---------

    @param[out]
    x           magma_d_vector *
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

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_d_vread(
    magma_d_vector *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue )
{
    x->memory_location = Magma_CPU;
    x->num_rows = length;
    x->num_cols = 1;
    x->major = MagmaColMajor;
    magma_dmalloc_cpu( &x->val, length );
    magma_int_t nnz=0, i=0;
    string line;
    ifstream fin(filename);  
    getline(fin, line, '\n');  
    while( i<length )  // eof() is 'true' at the end of data
    {
        getline(fin, line, '\n');
        if ( magma_dstring_to_double(line) != 0 )
            nnz++;
        x->val[i] = MAGMA_D_MAKE(magma_dstring_to_double(line), 0.0);
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
    x           magma_d_vector *
                vector to read in

    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_d_vspread(
    magma_d_vector *x, 
    const char * filename,
    magma_queue_t queue )
{
    magma_d_sparse_matrix A,B;
    magma_int_t entry=0;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    magma_d_csr_mtx( &A,  filename, queue  ); 
    magma_d_mconvert( A, &B, Magma_CSR, Magma_DENSE, queue );
    magma_d_vinit( x, Magma_CPU, A.num_cols*A.num_rows, MAGMA_D_ZERO, queue );
    x->major = MagmaRowMajor;
    for(magma_int_t i=0; i<A.num_cols; i++) {
        for(magma_int_t j=0; j<A.num_rows; j++) {
            x->val[i*A.num_rows+j] = B.val[ i+j*A.num_cols ];
            entry++;     
        }
    }
    x->num_rows = A.num_rows;
    x->num_cols = A.num_cols;

    magma_d_mfree( &A, queue );
    magma_d_mfree( &B, queue );

    return MAGMA_SUCCESS;
}   


