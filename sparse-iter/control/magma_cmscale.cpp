/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_zmscale.cpp normal z -> c, Sat Nov 15 19:54:23 2014
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
    A           magma_c_sparse_matrix*
                input/output matrix 

    @param[in]
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmscale(
    magma_c_sparse_matrix *A, 
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
            magmaFloatComplex *tmp;
            magma_cmalloc_cpu( &tmp, A->num_rows );
            for( magma_int_t z=0; z<A->num_rows; z++ ) {
                magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                    s+= MAGMA_C_REAL(A->val[f])*MAGMA_C_REAL(A->val[f]);
                tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );                   
            }
            for( magma_int_t z=0; z<A->nnz; z++ ) {
                A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
            }
            magma_free_cpu( tmp );
        }
        else if (scaling == Magma_UNITDIAG ) {
            // scale to unit diagonal
            magmaFloatComplex *tmp;
            magma_cmalloc_cpu( &tmp, A->num_rows );
            for( magma_int_t z=0; z<A->num_rows; z++ ) {
                magmaFloatComplex s = MAGMA_C_MAKE( 0.0, 0.0 );
                for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                    if ( A->col[f]== z ) {
                        // add some identity matrix
                        //A->val[f] = A->val[f] +  MAGMA_C_MAKE( 100000.0, 0.0 );
                        s = A->val[f];
                    }
                }
                if ( s == MAGMA_C_MAKE( 0.0, 0.0 ) )
                    printf("error: zero diagonal element.\n");
                tmp[z] = MAGMA_C_MAKE( 1.0/sqrt(  MAGMA_C_REAL( s )  ), 0.0 );    
                   
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

        magma_c_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_c_mtransfer( *A, &hA, A->memory_location, Magma_CPU, queue );
        magma_c_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue );

        magma_cmscale( &CSRA, scaling, queue );

        magma_c_mfree( &hA, queue );
        magma_c_mfree( A, queue );
        magma_c_mconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue );
        magma_c_mtransfer( hA, A, Magma_CPU, A_location, queue );
        magma_c_mfree( &hA, queue );
        magma_c_mfree( &CSRA, queue );    

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
    A           magma_c_sparse_matrix*
                input/output matrix 

    @param[in]
    add         magmaFloatComplex
                scaling for the identity matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmdiagadd(
    magma_c_sparse_matrix *A, 
    magmaFloatComplex add,
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

        magma_c_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_c_mtransfer( *A, &hA, A->memory_location, Magma_CPU, queue );
        magma_c_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue );

        magma_cmdiagadd( &CSRA, add, queue );

        magma_c_mfree( &hA, queue );
        magma_c_mfree( A, queue );
        magma_c_mconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue );
        magma_c_mtransfer( hA, A, Magma_CPU, A_location, queue );
        magma_c_mfree( &hA, queue );
        magma_c_mfree( &CSRA, queue );    

        return MAGMA_SUCCESS; 
    }
}



