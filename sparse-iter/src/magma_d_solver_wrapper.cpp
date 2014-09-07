/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from magma_z_solver_wrapper.cpp normal z -> d, Tue Sep  2 12:38:35 2014
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

    @param
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param
    b           magma_d_vector
                input vector b     

    @param
    x           magma_d_vector*
                output vector x        

    @param
    zopts     magma_dopts
              options for solver and preconditioner

    @ingroup magmasparse_daux
    ********************************************************************/

magma_int_t
magma_d_solver( magma_d_sparse_matrix A, magma_d_vector b, 
                 magma_d_vector *x, magma_dopts *zopts ){


        // preconditioner
        if( zopts->solver_par.solver != Magma_ITERREF )
            magma_d_precondsetup( A, b, &zopts->precond_par );

        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    magma_dcg_res( A, b, x, &zopts->solver_par );break;
            case  Magma_CGMERGE:
                    magma_dcg_merge( A, b, x, &zopts->solver_par );break;
            case  Magma_PCG:
                    magma_dpcg( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_BICGSTAB:
                    magma_dbicgstab( A, b, x, &zopts->solver_par );break;
            case  Magma_BICGSTABMERGE: 
                    magma_dbicgstab_merge( A, b, x, &zopts->solver_par );break;
            case  Magma_PBICGSTAB: 
                    magma_dpbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_GMRES: 
                    magma_dgmres( A, b, x, &zopts->solver_par );break;
            case  Magma_PGMRES: 
                    magma_dpgmres( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_LOBPCG: 
                    magma_dlobpcg( A, &zopts->solver_par );break;
            case  Magma_ITERREF: 
                    magma_diterref( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_JACOBI: 
                    magma_djacobi( A, b, x, &zopts->solver_par );break;
            case  Magma_BAITER: 
                    magma_dbaiter( A, b, x, &zopts->solver_par );break;
        }
    return MAGMA_SUCCESS;
}


