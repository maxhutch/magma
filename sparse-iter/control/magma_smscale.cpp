/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zmscale.cpp normal z -> s, Fri Jan 30 19:00:32 2015
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

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Scales a matrix.

    Arguments
    ---------

    @param[in,out]
    A           magma_s_sparse_matrix*
                input/output matrix 

    @param[in]
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smscale(
    magma_s_sparse_matrix *A, 
    magma_scale_t scaling,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        if ( scaling == Magma_NOSCALE ) {
            // no scale
            ;
        }
        else if ( scaling == Magma_UNITROW ) {
            // scale to unit rownorm
            float *tmp;
            magma_smalloc_cpu( &tmp, A->num_rows );
            for( magma_int_t z=0; z<A->num_rows; z++ ) {
                float s = MAGMA_S_MAKE( 0.0, 0.0 );
                for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                    s+= MAGMA_S_REAL(A->val[f])*MAGMA_S_REAL(A->val[f]);
                tmp[z] = MAGMA_S_MAKE( 1.0/sqrt(  MAGMA_S_REAL( s )  ), 0.0 );                   
            }
            for( magma_int_t z=0; z<A->nnz; z++ ) {
                A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
            }
            magma_free_cpu( tmp );
        }
        else if (scaling == Magma_UNITDIAG ) {
            // scale to unit diagonal
            float *tmp;
            magma_smalloc_cpu( &tmp, A->num_rows );
            for( magma_int_t z=0; z<A->num_rows; z++ ) {
                float s = MAGMA_S_MAKE( 0.0, 0.0 );
                for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                    if ( A->col[f]== z ) {
                        // add some identity matrix
                        //A->val[f] = A->val[f] +  MAGMA_S_MAKE( 100000.0, 0.0 );
                        s = A->val[f];
                    }
                }
                if ( s == MAGMA_S_MAKE( 0.0, 0.0 ) )
                    printf("error: zero diagonal element.\n");
                tmp[z] = MAGMA_S_MAKE( 1.0/sqrt(  MAGMA_S_REAL( s )  ), 0.0 );    
                   
            }
            for( magma_int_t z=0; z<A->nnz; z++ ) {
                A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
            }
            magma_free_cpu( tmp );
        }
        else
            printf( "error: scaling not supported\n" );
        return MAGMA_SUCCESS; 
    }
    else {

        magma_s_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_s_mtransfer( *A, &hA, A->memory_location, Magma_CPU, queue );
        magma_s_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue );

        magma_smscale( &CSRA, scaling, queue );

        magma_s_mfree( &hA, queue );
        magma_s_mfree( A, queue );
        magma_s_mconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue );
        magma_s_mtransfer( hA, A, Magma_CPU, A_location, queue );
        magma_s_mfree( &hA, queue );
        magma_s_mfree( &CSRA, queue );    

        return MAGMA_SUCCESS; 
    }
}


/**
    Purpose
    -------

    Adds a multiple of the Identity matrix to a matrix: A = A+add * I

    Arguments
    ---------

    @param[in,out]
    A           magma_s_sparse_matrix*
                input/output matrix 

    @param[in]
    add         float
                scaling for the identity matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smdiagadd(
    magma_s_sparse_matrix *A, 
    float add,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        for( magma_int_t z=0; z<A->nnz; z++ ) {
            if ( A->col[z]== A->rowidx[z] ) {
                // add some identity matrix
                A->val[z] = A->val[z] +  add;
            }
        }
        return MAGMA_SUCCESS; 
    }
    else {

        magma_s_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_s_mtransfer( *A, &hA, A->memory_location, Magma_CPU, queue );
        magma_s_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue );

        magma_smdiagadd( &CSRA, add, queue );

        magma_s_mfree( &hA, queue );
        magma_s_mfree( A, queue );
        magma_s_mconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue );
        magma_s_mtransfer( hA, A, Magma_CPU, A_location, queue );
        magma_s_mfree( &hA, queue );
        magma_s_mfree( &CSRA, queue );    

        return MAGMA_SUCCESS; 
    }
}



