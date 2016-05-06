/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/control/magma_zdummy.cpp normal z -> c, Mon May  2 23:30:53 2016
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>




extern "C" magma_int_t
magma_cmdynamicic_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_c_matrix *LU_new,
    magma_c_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    



extern "C" magma_int_t
magma_cmdynamicilu_rm_thrs(
    magmaFloatComplex *thrs,
    magma_int_t *num_rm,
    magma_c_matrix *LU,
    magma_c_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    



extern "C" magma_int_t
magma_cmdynamicilu_set_thrs(
    magma_int_t num_rm,
    magma_c_matrix *LU,
    magmaFloatComplex *thrs,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    


extern "C" magma_int_t
magma_cmdynamicic_sweep(
    magma_c_matrix A,
    magma_c_matrix *LU,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    

 



/**
    Purpose
    -------
    This function computes the residuals for the candidates.

    Arguments
    ---------
    
    @param[in]
    A           magma_c_matrix
                System matrix A.
    
    @param[in]
    LU          magma_c_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_c_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmdynamicic_residuals(
    magma_c_matrix A,
    magma_c_matrix LU,
    magma_c_matrix *LU_new,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    




/**
    Purpose
    -------
    This function identifies the candidates.

    Arguments
    ---------
    
    @param[in]
    LU          magma_c_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_c_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmdynamicic_candidates(
    magma_c_matrix LU,
    magma_c_matrix *LU_new,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    


#endif

