/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_z_vio.cpp normal z -> s, Fri Jan 30 19:00:32 2015
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


/**
    Purpose
    -------

    Visualizes part of a vector of type magma_s_vector.
    With input vector x , offset, visulen, the entries 
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param[in]
    x           magma_s_vector
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

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_s_vvisu(
    magma_s_vector x, 
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
            printf("%5.2f\n", MAGMA_S_REAL(x.val[i]));
    return MAGMA_SUCCESS;
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        magma_s_vector y;
        magma_s_vtransfer( x, &y, Magma_DEV, Magma_CPU, queue );
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            printf("%5.2f\n", MAGMA_S_REAL(y.val[i]));
    magma_free_cpu(y.val);
    return MAGMA_SUCCESS;
    }
    return MAGMA_SUCCESS;
}   




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



/**
    Purpose
    -------

    Reads in a float vector of length "length".

    Arguments
    ---------

    @param[out]
    x           magma_s_vector *
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

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_s_vread(
    magma_s_vector *x, 
    magma_int_t length,
    char * filename,
    magma_queue_t queue )
{
    x->memory_location = Magma_CPU;
    x->num_rows = length;
    x->num_cols = 1;
    x->major = MagmaColMajor;
    magma_smalloc_cpu( &x->val, length );
    magma_int_t nnz=0, i=0;
    string line;
    ifstream fin(filename);  
    getline(fin, line, '\n');  
    while( i<length )  // eof() is 'true' at the end of data
    {
        getline(fin, line, '\n');
        if ( magma_sstring_to_float(line) != 0 )
            nnz++;
        x->val[i] = MAGMA_S_MAKE(magma_sstring_to_float(line), 0.0);
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
    x           magma_s_vector *
                vector to read in

    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_s_vspread(
    magma_s_vector *x, 
    const char * filename,
    magma_queue_t queue )
{
    magma_s_sparse_matrix A,B;
    magma_int_t entry=0;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    magma_s_csr_mtx( &A,  filename, queue  ); 
    magma_s_mconvert( A, &B, Magma_CSR, Magma_DENSE, queue );
    magma_s_vinit( x, Magma_CPU, A.num_cols*A.num_rows, MAGMA_S_ZERO, queue );
    x->major = MagmaRowMajor;
    for(magma_int_t i=0; i<A.num_cols; i++) {
        for(magma_int_t j=0; j<A.num_rows; j++) {
            x->val[i*A.num_rows+j] = B.val[ i+j*A.num_cols ];
            entry++;     
        }
    }
    x->num_rows = A.num_rows;
    x->num_cols = A.num_cols;

    magma_s_mfree( &A, queue );
    magma_s_mfree( &B, queue );

    return MAGMA_SUCCESS;
}   


