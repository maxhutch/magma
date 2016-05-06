/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/magma_z_solver_wrapper.cpp normal z -> s, Mon May  2 23:31:04 2016
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
    A           magma_s_matrix
                sparse matrix A

    @param[in]
    b           magma_s_matrix
                input vector b

    @param[in]
    x           magma_s_matrix*
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
    magma_s_matrix A, magma_s_matrix b,
    magma_s_matrix *x, magma_sopts *zopts,
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
                    CHECK( magma_scg_res( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICG:
                    CHECK( magma_sbicg( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICG:
                    CHECK( magma_spbicg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGMERGE:
                    CHECK( magma_scg_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_spcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCGMERGE:
                    CHECK( magma_spcg_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_BICGSTAB:
                    CHECK( magma_sbicgstab( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICGSTABMERGE: 
                    CHECK( magma_sbicgstab_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICGSTABMERGE:
                    CHECK( magma_spbicgstab_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PBICGSTAB:
                    CHECK( magma_spbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_GMRES:
                    CHECK( magma_sfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PGMRES:
                    CHECK( magma_sfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_IDR:
                    CHECK( magma_sidr( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_IDRMERGE:
                    CHECK( magma_sidr_merge( A, b, x, &zopts->solver_par, queue )); break;
                    //CHECK( magma_sidr_strms( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PIDR:
                    CHECK( magma_spidr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PIDRMERGE:
                    //CHECK( magma_spidr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
                    CHECK( magma_spidr_strms( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_slobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_ITERREF:
                    CHECK( magma_siterref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_JACOBI:
                    CHECK( magma_sjacobi( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BAITER:
                    CHECK( magma_sbaiter( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_BAITERO:
                    CHECK( magma_sbaiter_overlap( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGS:
                    CHECK( magma_scgs( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_CGSMERGE:
                    CHECK( magma_scgs_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PCGS:
                    CHECK( magma_spcgs( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_PCGSMERGE:
                    CHECK( magma_spcgs_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMR:
                    CHECK( magma_stfqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMR:
                    CHECK( magma_sptfqmr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMRMERGE:
                    CHECK( magma_stfqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMRMERGE:
                    CHECK( magma_sptfqmr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMR:
                    CHECK( magma_sqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_LSQR:
                    CHECK( magma_slsqr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMRMERGE:
                    CHECK( magma_sqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARD:
                    CHECK( magma_sbombard( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARDMERGE:
                    CHECK( magma_sbombard_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            default:
                    printf("error: solver class not supported.\n"); break;
        }
    }
    else {
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    CHECK( magma_sbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_sbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_slobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            default:
                    printf("error: only 1 RHS supported for this solver class.\n"); break;
        }
    }
cleanup:
    return info; 
}
