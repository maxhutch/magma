/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Scales a matrix.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix*
                input/output matrix 

    @param
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)


    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmscale( magma_z_sparse_matrix *A, magma_scale_t scaling ){

    if( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ){
        if( scaling == Magma_NOSCALE ){
            // no scale
            ;
        }
        else if( scaling == Magma_UNITROW ){
            // scale to unit rownorm
            magmaDoubleComplex *tmp;
            magma_zmalloc_cpu( &tmp, A->num_rows );
            for( magma_int_t z=0; z<A->num_rows; z++ ){
                magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                    s+= MAGMA_Z_REAL(A->val[f])*MAGMA_Z_REAL(A->val[f]);
                tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );                   
            }
            for( magma_int_t z=0; z<A->nnz; z++ ){
                A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
            }
            magma_free_cpu( tmp );
        }
        else if (scaling == Magma_UNITDIAG ){
            // scale to unit diagonal
            magmaDoubleComplex *tmp;
            magma_zmalloc_cpu( &tmp, A->num_rows );
            for( magma_int_t z=0; z<A->num_rows; z++ ){
                magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ){
                    if( A->col[f]== z ){
                        // add some identity matrix
                        //A->val[f] = A->val[f] +  MAGMA_Z_MAKE( 100000.0, 0.0 );
                        s = A->val[f];
                    }
                }
                if( s == MAGMA_Z_MAKE( 0.0, 0.0 ) )
                    printf("error: zero diagonal element.\n");
                tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );    
                   
            }
            for( magma_int_t z=0; z<A->nnz; z++ ){
                A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
            }
            magma_free_cpu( tmp );
        }
        else
            printf( "error: scaling not supported\n" );
        return MAGMA_SUCCESS; 
    }
    else{

        magma_z_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_z_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_zmscale( &CSRA, scaling );

        magma_z_mfree( &hA );
        magma_z_mfree( A );
        magma_z_mconvert( CSRA, &hA, Magma_CSRCOO, A_storage );
        magma_z_mtransfer( hA, A, Magma_CPU, A_location );
        magma_z_mfree( &hA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}


/**
    Purpose
    -------

    Adds a multiple of the Identity matrix to a matrix: A = A+add * I

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix*
                input/output matrix 

    @param
    add         magmaDoubleComplex
                scaling for the identity matrix

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdiagadd( magma_z_sparse_matrix *A, magmaDoubleComplex add ){

    if( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ){
        for( magma_int_t z=0; z<A->nnz; z++ ){
            if( A->col[z]== A->rowidx[z] ){
                // add some identity matrix
                A->val[z] = A->val[z] +  add;
            }
        }
        return MAGMA_SUCCESS; 
    }
    else{

        magma_z_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_z_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_zmdiagadd( &CSRA, add );

        magma_z_mfree( &hA );
        magma_z_mfree( A );
        magma_z_mconvert( CSRA, &hA, Magma_CSRCOO, A_storage );
        magma_z_mtransfer( hA, A, Magma_CPU, A_location );
        magma_z_mfree( &hA );
        magma_z_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}



