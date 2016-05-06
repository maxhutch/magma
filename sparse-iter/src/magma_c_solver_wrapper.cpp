/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/magma_z_solver_wrapper.cpp normal z -> c, Mon May  2 23:31:04 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"




/**
    Purpose
    -------

    Allows the user to choose a solver.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                sparse matrix A

    @param[in]
    b           magma_c_matrix
                input vector b

    @param[in]
    x           magma_c_matrix*
                output vector x

    @param[in]
    zopts     magma_copts
              options for solver and preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_c_solver(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_matrix *x, magma_copts *zopts,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // make sure RHS is a dense matrix
    if ( b.storage_type != Magma_DENSE ) {
        printf( "error: sparse RHS not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    if( b.num_cols == 1 ){
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    CHECK( magma_ccg_res( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICG:
                    CHECK( magma_cbicg( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICG:
                    CHECK( magma_cpbicg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGMERGE:
                    CHECK( magma_ccg_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_cpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCGMERGE:
                    CHECK( magma_cpcg_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_BICGSTAB:
                    CHECK( magma_cbicgstab( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICGSTABMERGE: 
                    CHECK( magma_cbicgstab_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICGSTABMERGE:
                    CHECK( magma_cpbicgstab_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PBICGSTAB:
                    CHECK( magma_cpbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_GMRES:
                    CHECK( magma_cfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PGMRES:
                    CHECK( magma_cfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_IDR:
                    CHECK( magma_cidr( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_IDRMERGE:
                    CHECK( magma_cidr_merge( A, b, x, &zopts->solver_par, queue )); break;
                    //CHECK( magma_cidr_strms( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PIDR:
                    CHECK( magma_cpidr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PIDRMERGE:
                    //CHECK( magma_cpidr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
                    CHECK( magma_cpidr_strms( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_clobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_ITERREF:
                    CHECK( magma_citerref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_JACOBI:
                    CHECK( magma_cjacobi( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BAITER:
                    CHECK( magma_cbaiter( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_BAITERO:
                    CHECK( magma_cbaiter_overlap( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGS:
                    CHECK( magma_ccgs( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_CGSMERGE:
                    CHECK( magma_ccgs_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PCGS:
                    CHECK( magma_cpcgs( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_PCGSMERGE:
                    CHECK( magma_cpcgs_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMR:
                    CHECK( magma_ctfqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMR:
                    CHECK( magma_cptfqmr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMRMERGE:
                    CHECK( magma_ctfqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMRMERGE:
                    CHECK( magma_cptfqmr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMR:
                    CHECK( magma_cqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_LSQR:
                    CHECK( magma_clsqr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMRMERGE:
                    CHECK( magma_cqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARD:
                    CHECK( magma_cbombard( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARDMERGE:
                    CHECK( magma_cbombard_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            default:
                    printf("error: solver class not supported.\n"); break;
        }
    }
    else {
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    CHECK( magma_cbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_cbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_clobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            default:
                    printf("error: only 1 RHS supported for this solver class.\n"); break;
        }
    }
cleanup:
    return info; 
}
