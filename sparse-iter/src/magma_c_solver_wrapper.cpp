/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from magma_z_solver_wrapper.cpp normal z -> c, Tue Sep  2 12:38:35 2014
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
    A           magma_c_sparse_matrix
                sparse matrix A    

    @param
    b           magma_c_vector
                input vector b     

    @param
    x           magma_c_vector*
                output vector x        

    @param
    zopts     magma_copts
              options for solver and preconditioner

    @ingroup magmasparse_caux
    ********************************************************************/

magma_int_t
magma_c_solver( magma_c_sparse_matrix A, magma_c_vector b, 
                 magma_c_vector *x, magma_copts *zopts ){


        // preconditioner
        if( zopts->solver_par.solver != Magma_ITERREF )
            magma_c_precondsetup( A, b, &zopts->precond_par );

        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    magma_ccg_res( A, b, x, &zopts->solver_par );break;
            case  Magma_CGMERGE:
                    magma_ccg_merge( A, b, x, &zopts->solver_par );break;
            case  Magma_PCG:
                    magma_cpcg( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_BICGSTAB:
                    magma_cbicgstab( A, b, x, &zopts->solver_par );break;
            case  Magma_BICGSTABMERGE: 
                    magma_cbicgstab_merge( A, b, x, &zopts->solver_par );break;
            case  Magma_PBICGSTAB: 
                    magma_cpbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_GMRES: 
                    magma_cgmres( A, b, x, &zopts->solver_par );break;
            case  Magma_PGMRES: 
                    magma_cpgmres( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_LOBPCG: 
                    magma_clobpcg( A, &zopts->solver_par );break;
            case  Magma_ITERREF: 
                    magma_citerref( A, b, x, &zopts->solver_par, &zopts->precond_par );break;
            case  Magma_JACOBI: 
                    magma_cjacobi( A, b, x, &zopts->solver_par );break;
            case  Magma_BAITER: 
                    magma_cbaiter( A, b, x, &zopts->solver_par );break;
        }
    return MAGMA_SUCCESS;
}


