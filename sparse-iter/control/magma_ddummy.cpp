/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from sparse-iter/control/magma_zdummy.cpp, normal z -> d, Tue Aug 30 09:38:51 2016
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>

extern "C" magma_int_t
magma_dparilutsetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}

extern "C" magma_int_t
magma_dparictsetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}


extern "C" magma_int_t
magma_dparict_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_d_matrix *LU_new,
    magma_d_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    



extern "C" magma_int_t
magma_dparilut_rm_thrs(
    double *thrs,
    magma_int_t *num_rm,
    magma_d_matrix *LU,
    magma_d_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    



extern "C" magma_int_t
magma_dparilut_set_thrs(
    magma_int_t num_rm,
    magma_d_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    


extern "C" magma_int_t
magma_dparict_sweep(
    magma_d_matrix *A,
    magma_d_matrix *LU,
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
    A           magma_d_matrix
                System matrix A.
    
    @param[in]
    LU          magma_d_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_d_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dparict_residuals(
    magma_d_matrix A,
    magma_d_matrix LU,
    magma_d_matrix *LU_new,
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
    LU          magma_d_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_d_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dparict_candidates(
    magma_d_matrix LU,
    magma_d_matrix *LU_new,
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
    A           magma_d_matrix
                input matrix A
                
    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_diluisaisetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
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
    b           magma_d_matrix
                input RHS b
                
    @param[in,out]
    x           magma_d_matrix
                solution x

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_disai_l(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
extern "C"
magma_int_t
magma_disai_l_t(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
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
    b           magma_d_matrix
                input RHS b
                
    @param[in,out]
    x           magma_d_matrix
                solution x

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_disai_r(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
extern "C"
magma_int_t
magma_disai_r_t(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
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
    of dgeisai.cpp. 
    

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A
                
    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dicisaisetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
