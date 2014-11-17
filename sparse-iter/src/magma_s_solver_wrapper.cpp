/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_z_solver_wrapper.cpp normal z -> s, Sat Nov 15 19:54:22 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"




/**
    Purpose
    -------

    ALlows the user to choose a solver.

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_s_vector
                input vector b     

    @param[in]
    x           magma_s_vector*
                output vector x        

    @param[in]
    zopts     magma_sopts
              options for solver and preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_s_solver(
    magma_s_sparse_matrix A, magma_s_vector b, 
    magma_s_vector *x, magma_sopts *zopts,
    magma_queue_t queue )
{
    // preconditioner
        if ( zopts->solver_par.solver != Magma_ITERREF ) {
            magma_s_precondsetup( A, b, &zopts->precond_par, queue );
        }
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    magma_scg_res( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_CGMERGE:
                    magma_scg_merge( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_PCG:
                    magma_spcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_BICGSTAB:
                    magma_sbicgstab( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_BICGSTABMERGE: 
                    magma_sbicgstab_merge( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_PBICGSTAB: 
                    magma_spbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_GMRES: 
                    magma_sgmres( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_PGMRES: 
                    magma_spgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_LOBPCG: 
                    magma_slobpcg( A, &zopts->solver_par, queue );break;
            case  Magma_ITERREF:
                    magma_siterref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_JACOBI: 
                    magma_sjacobi( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_BAITER: 
                    magma_sbaiter( A, b, x, &zopts->solver_par, queue );break;
            
        }
    return MAGMA_SUCCESS;
}


