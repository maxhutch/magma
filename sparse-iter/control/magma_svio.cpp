/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/control/magma_zvio.cpp normal z -> s, Mon May  2 23:30:54 2016
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define REAL
#define PRECISION_s

/**
    Purpose
    -------

    Visualizes part of a vector of type magma_s_matrix.
    With input vector x , offset, visulen, the entries
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param[in]
    x           magma_s_matrix
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
magma_sprint_vector(
    magma_s_matrix x,
    magma_int_t offset,
    magma_int_t  visulen,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_s_matrix y={Magma_CSR};
    
    //**************************************************************
    float c_zero = MAGMA_S_ZERO;
    
    #ifdef COMPLEX
    #define magma_sprintval( tmp )       {                                  \
        if ( MAGMA_S_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.              \n" );                                \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f+%8.4fi\n",                                        \
                    MAGMA_S_REAL( tmp ), MAGMA_S_IMAG( tmp ));              \
        }                                                                   \
    }
    #else
    #define magma_sprintval( tmp )       {                                  \
        if ( MAGMA_S_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.    \n" );                                          \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f\n", MAGMA_S_REAL( tmp ));                         \
        }                                                                   \
    }
    #endif
    //**************************************************************
    
    printf("visualize entries %d - %d of vector ",
                    int(offset), int(offset + visulen) );
    fflush(stdout);
    if ( x.memory_location == Magma_CPU ) {
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset + visulen; i++ )
            magma_sprintval(x.val[i]);
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        CHECK( magma_smtransfer( x, &y, Magma_DEV, Magma_CPU, queue ));
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            magma_sprintval(y.val[i]);
    }

cleanup:
    magma_free_cpu(y.val);
    return info;
}




/**
    Purpose
    -------

    Reads in a float vector of length "length".

    Arguments
    ---------

    @param[out]
    x           magma_s_matrix *
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
magma_svread(
    magma_s_matrix *x,
    magma_int_t length,
    char * filename,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t nnz=0, i=0;
    FILE *fid;
    char buff[BUFSIZ]={0};
    int count=0;
    char *p;
    
    x->memory_location = Magma_CPU;
    x->storage_type = Magma_DENSE;
    x->num_rows = length;
    x->num_cols = 1;
    x->major = MagmaColMajor;
    CHECK( magma_smalloc_cpu( &x->val, length ));
    
    fid = fopen(filename, "r");

    if(NULL==fgets(buff, BUFSIZ, fid))
        return -1;
    rewind(fid);
    for( p=buff; NULL != strtok(p, " \t\n"); p=NULL)
        count++;
    
    while( i<length )  // eof() is 'true' at the end of data
    {
        float VAL1;

        float VAL;
        
        #if defined(PRECISION_z) || defined(PRECISION_d)
            float VAL2;
            if( count == 2 ){
                fscanf(fid, "%lg %lg\n", &VAL1, &VAL2);
                VAL = MAGMA_S_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%lg\n", &VAL1);
                VAL = MAGMA_S_MAKE(VAL1, 0.0);  
            }
        #else // single-real or single
            float VAL2;
            if( count == 2 ){
                fscanf(fid, "%g %g\n", &VAL1, &VAL2);
                VAL = MAGMA_S_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%g\n", &VAL1);
                VAL = MAGMA_S_MAKE(VAL1, 0.0);  
            }
        #endif
        
        if ( VAL != MAGMA_S_ZERO )
            nnz++;
        x->val[i] = VAL;
        i++;
    }
    fclose(fid);
    
    x->nnz = nnz;
    
cleanup:
    return info;
}




/**
    Purpose
    -------

    Reads in a sparse vector-block stored in COO format.

    Arguments
    ---------

    @param[out]
    x           magma_s_matrix *
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
magma_svspread(
    magma_s_matrix *x,
    const char * filename,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_s_matrix A={Magma_CSR}, B={Magma_CSR};
    magma_int_t entry=0;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    CHECK( magma_s_csr_mtx( &A,  filename, queue  ));
    CHECK( magma_smconvert( A, &B, Magma_CSR, Magma_DENSE, queue ));
    CHECK( magma_svinit( x, Magma_CPU, A.num_cols, A.num_rows, MAGMA_S_ZERO, queue ));
    x->major = MagmaRowMajor;
    for(magma_int_t i=0; i<A.num_cols; i++) {
        for(magma_int_t j=0; j<A.num_rows; j++) {
            x->val[i*A.num_rows+j] = B.val[ i+j*A.num_cols ];
            entry++;
        }
    }
    x->num_rows = A.num_rows;
    x->num_cols = A.num_cols;
    
cleanup:
    magma_smfree( &A, queue );
    magma_smfree( &B, queue );
    return info;
}
