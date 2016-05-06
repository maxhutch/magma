/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/magma_z_solver_wrapper.cpp normal z -> d, Mon May  2 23:31:04 2016
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
    A           magma_d_matrix
                sparse matrix A

    @param[in]
    b           magma_d_matrix
                input vector b

    @param[in]
    x           magma_d_matrix*
                output vector x

    @param[in]
    zopts     magma_dopts
              options for solver and preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_d_solver(
    magma_d_matrix A, magma_d_matrix b,
    magma_d_matrix *x, magma_dopts *zopts,
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
                    CHECK( magma_dcg_res( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICG:
                    CHECK( magma_dbicg( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICG:
                    CHECK( magma_dpbicg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGMERGE:
                    CHECK( magma_dcg_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_dpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCGMERGE:
                    CHECK( magma_dpcg_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_BICGSTAB:
                    CHECK( magma_dbicgstab( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICGSTABMERGE: 
                    CHECK( magma_dbicgstab_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICGSTABMERGE:
                    CHECK( magma_dpbicgstab_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PBICGSTAB:
                    CHECK( magma_dpbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_GMRES:
                    CHECK( magma_dfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PGMRES:
                    CHECK( magma_dfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_IDR:
                    CHECK( magma_didr( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_IDRMERGE:
                    CHECK( magma_didr_merge( A, b, x, &zopts->solver_par, queue )); break;
                    //CHECK( magma_didr_strms( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PIDR:
                    CHECK( magma_dpidr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PIDRMERGE:
                    //CHECK( magma_dpidr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
                    CHECK( magma_dpidr_strms( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_dlobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_ITERREF:
                    CHECK( magma_diterref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_JACOBI:
                    CHECK( magma_djacobi( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BAITER:
                    CHECK( magma_dbaiter( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_BAITERO:
                    CHECK( magma_dbaiter_overlap( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGS:
                    CHECK( magma_dcgs( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_CGSMERGE:
                    CHECK( magma_dcgs_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PCGS:
                    CHECK( magma_dpcgs( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_PCGSMERGE:
                    CHECK( magma_dpcgs_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMR:
                    CHECK( magma_dtfqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMR:
                    CHECK( magma_dptfqmr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMRMERGE:
                    CHECK( magma_dtfqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMRMERGE:
                    CHECK( magma_dptfqmr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMR:
                    CHECK( magma_dqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_LSQR:
                    CHECK( magma_dlsqr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMRMERGE:
                    CHECK( magma_dqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARD:
                    CHECK( magma_dbombard( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARDMERGE:
                    CHECK( magma_dbombard_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            default:
                    printf("error: solver class not supported.\n"); break;
        }
    }
    else {
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    CHECK( magma_dbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_dbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_dlobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            default:
                    printf("error: only 1 RHS supported for this solver class.\n"); break;
        }
    }
cleanup:
    return info; 
}
