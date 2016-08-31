/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from sparse-iter/control/magma_zdummy.cpp, normal z -> s, Tue Aug 30 09:38:51 2016
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>

extern "C" magma_int_t
magma_sparilutsetup(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}

extern "C" magma_int_t
magma_sparictsetup(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}


extern "C" magma_int_t
magma_sparict_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_s_matrix *LU_new,
    magma_s_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    



extern "C" magma_int_t
magma_sparilut_rm_thrs(
    float *thrs,
    magma_int_t *num_rm,
    magma_s_matrix *LU,
    magma_s_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    



extern "C" magma_int_t
magma_sparilut_set_thrs(
    magma_int_t num_rm,
    magma_s_matrix *LU,
    magma_int_t order,
    float *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    


extern "C" magma_int_t
magma_sparict_sweep(
    magma_s_matrix *A,
    magma_s_matrix *LU,
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
    A           magma_s_matrix
                System matrix A.
    
    @param[in]
    LU          magma_s_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_s_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_sparict_residuals(
    magma_s_matrix A,
    magma_s_matrix LU,
    magma_s_matrix *LU_new,
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
    LU          magma_s_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_s_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_sparict_candidates(
    magma_s_matrix LU,
    magma_s_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}

#endif


/**
    Purpose
    -------

    Prepares Incomplete LU preconditioner using a sparse approximate inverse 
    instead of sparse triangular solves.
    

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A
                
    @param[in]
    b           magma_s_matrix
                input RHS b

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_siluisaisetup(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    




/**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner.
    

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                input RHS b
                
    @param[in,out]
    x           magma_s_matrix
                solution x

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_sisai_l(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
extern "C"
magma_int_t
magma_sisai_l_t(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}


/**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner.
    

    Arguments
    ---------

    @param[in]
    b           magma_s_matrix
                input RHS b
                
    @param[in,out]
    x           magma_s_matrix
                solution x

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_sisai_r(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
extern "C"
magma_int_t
magma_sisai_r_t(
    magma_s_matrix b,
    magma_s_matrix *x,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}

    

/**
    Purpose
    -------

    Prepares Incomplete Cholesky preconditioner using a sparse approximate 
    inverse instead of sparse triangular solves. This is the symmetric variant 
    of sgeisai.cpp. 
    

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A
                
    @param[in]
    b           magma_s_matrix
                input RHS b

    @param[in,out]
    precond     magma_s_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_sicisaisetup(
    magma_s_matrix A,
    magma_s_matrix b,
    magma_s_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
