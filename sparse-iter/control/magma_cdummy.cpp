/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from sparse-iter/control/magma_zdummy.cpp, normal z -> c, Tue Aug 30 09:38:51 2016
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>

extern "C" magma_int_t
magma_cparilutsetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}

extern "C" magma_int_t
magma_cparictsetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}


extern "C" magma_int_t
magma_cparict_insert(
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
magma_cparilut_rm_thrs(
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
magma_cparilut_set_thrs(
    magma_int_t num_rm,
    magma_c_matrix *LU,
    magma_int_t order,
    magmaFloatComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    


extern "C" magma_int_t
magma_cparict_sweep(
    magma_c_matrix *A,
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
magma_cparict_residuals(
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
magma_cparict_candidates(
    magma_c_matrix LU,
    magma_c_matrix *LU_new,
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
    A           magma_c_matrix
                input matrix A
                
    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_ciluisaisetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
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
    b           magma_c_matrix
                input RHS b
                
    @param[in,out]
    x           magma_c_matrix
                solution x

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_cisai_l(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
extern "C"
magma_int_t
magma_cisai_l_t(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
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
    b           magma_c_matrix
                input RHS b
                
    @param[in,out]
    x           magma_c_matrix
                solution x

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_cisai_r(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
extern "C"
magma_int_t
magma_cisai_r_t(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
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
    of cgeisai.cpp. 
    

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A
                
    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_cicisaisetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
