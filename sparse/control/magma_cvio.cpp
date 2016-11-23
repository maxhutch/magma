/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/control/magma_zvio.cpp, normal z -> c, Sun Nov 20 20:20:43 2016
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define COMPLEX
#define PRECISION_c

/**
    Purpose
    -------

    Visualizes part of a vector of type magma_c_matrix.
    With input vector x , offset, visulen, the entries
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param[in]
    x           magma_c_matrix
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
magma_cprint_vector(
    magma_c_matrix x,
    magma_int_t offset,
    magma_int_t  visulen,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix y={Magma_CSR};
    
    //**************************************************************
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    
    #ifdef COMPLEX
    #define magma_cprintval( tmp )       {                                  \
        if ( MAGMA_C_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.              \n" );                                \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f+%8.4fi\n",                                        \
                    MAGMA_C_REAL( tmp ), MAGMA_C_IMAG( tmp ));              \
        }                                                                   \
    }
    #else
    #define magma_cprintval( tmp )       {                                  \
        if ( MAGMA_C_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.    \n" );                                          \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f\n", MAGMA_C_REAL( tmp ));                         \
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
            magma_cprintval(x.val[i]);
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        CHECK( magma_cmtransfer( x, &y, Magma_DEV, Magma_CPU, queue ));
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            magma_cprintval(y.val[i]);
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
    x           magma_c_matrix *
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
magma_cvread(
    magma_c_matrix *x,
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
    CHECK( magma_cmalloc_cpu( &x->val, length ));
    
    fid = fopen(filename, "r");

    if(NULL==fgets(buff, BUFSIZ, fid))
        return -1;
    rewind(fid);
    for( p=buff; NULL != strtok(p, " \t\n"); p=NULL)
        count++;
    
    while( i<length )  // eof() is 'true' at the end of data
    {
        float VAL1;

        magmaFloatComplex VAL;
        
        #if defined(PRECISION_z) || defined(PRECISION_d)
            float VAL2;
            if( count == 2 ){
                fscanf(fid, "%lg %lg\n", &VAL1, &VAL2);
                VAL = MAGMA_C_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%lg\n", &VAL1);
                VAL = MAGMA_C_MAKE(VAL1, 0.0);  
            }
        #else // single-complex or single
            float VAL2;
            if( count == 2 ){
                fscanf(fid, "%g %g\n", &VAL1, &VAL2);
                VAL = MAGMA_C_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%g\n", &VAL1);
                VAL = MAGMA_C_MAKE(VAL1, 0.0);  
            }
        #endif
        
        if ( VAL != MAGMA_C_ZERO )
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
    x           magma_c_matrix *
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
magma_cvspread(
    magma_c_matrix *x,
    const char * filename,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix A={Magma_CSR}, B={Magma_CSR};
    magma_int_t entry=0;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    CHECK( magma_c_csr_mtx( &A,  filename, queue  ));
    CHECK( magma_cmconvert( A, &B, Magma_CSR, Magma_DENSE, queue ));
    CHECK( magma_cvinit( x, Magma_CPU, A.num_cols, A.num_rows, MAGMA_C_ZERO, queue ));
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
    magma_cmfree( &A, queue );
    magma_cmfree( &B, queue );
    return info;
}

/**
    Purpose
    -------

    Writes a vector to a file.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                matrix to write out

    @param[in]
    filename    const char*
                output-filname of the mtx matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C"
magma_int_t
magma_cwrite_vector(
    magma_c_matrix A,
    const char *filename,
    magma_queue_t queue )
{
    magma_int_t i, info = 0;
    
    FILE *fp;
    
    fp = fopen(filename, "w");
    if ( fp == NULL ){
        printf("\n%% error writing vector: file exists or missing write permission\n");
        info = -1;
        goto cleanup;
    }
            
    #define COMPLEX

    #ifdef COMPLEX
    // complex case
    for(i=0; i < A.num_rows; i++) {
        fprintf( fp, "%.16g %.16g\n",
            MAGMA_C_REAL((A.val)[i]),
            MAGMA_C_IMAG((A.val)[i]) );
    }
    #else
    for(i=0; i < A.num_rows; i++) {
        fprintf( fp, "%.16g\n",
            MAGMA_C_REAL((A.val)[i]) );
    }
    #endif
    
    if (fclose(fp) != 0)
        printf("\n%% error: writing matrix failed\n");
    else
        info = 0;

cleanup:
    return info;
}
