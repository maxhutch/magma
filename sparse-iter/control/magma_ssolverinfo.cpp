/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zsolverinfo.cpp normal z -> s, Sun May  3 11:23:01 2015
       @author Hartwig Anzt

*/
#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

/**
    Purpose
    -------

    Prints information about a previously called solver.

    Arguments
    ---------

    @param[in]
    solver_par  magma_s_solver_par*
                structure containing all solver information
    @param[in,out]
    precond_par magma_s_preconditioner*
                structure containing all preconditioner information
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_ssolverinfo(
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    if( solver_par->verbose > 0 ){
        magma_int_t k = solver_par->verbose;
        printf("%%======================================================="
            "======%%\n");
        switch( solver_par->solver ) {
            case  Magma_CG:
                    printf("%%   CG performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_PCG:
                    printf("%%   CG performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_CGMERGE:
                    printf("%%   CG (merged) performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_BICGSTAB:
                    printf("%%   BiCGSTAB performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_PBICGSTAB:
                    printf("%%   BiCGSTAB performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_BICGSTABMERGE:
                    printf("%%   BiCGSTAB (merged) performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_BICGSTABMERGE2:
                    printf("%%   BiCGSTAB (merged) performance analysis every %d iteration\n",
                                                                        (int) k); break;
            case  Magma_GMRES:
                    printf("%%   GMRES(%d) performance analysis every %d iteration\n",
                                                (int) solver_par->restart, (int) k); break;
            case  Magma_PGMRES:
                    printf("%%   GMRES(%d) performance analysis every %d iteration\n",
                                                (int) solver_par->restart, (int) k); break;
            case  Magma_ITERREF:
                    printf("%%   Iterative refinement performance analysis every %d iteration\n",
                                                                        (int) k); break;
            default:
                    printf("%%   Detailed performance analysis not supported.\n"); break;
    
        }
        
        switch( precond_par->solver ) {
            case  Magma_CG:
                    printf("%%   Preconditioner used: CG.\n"); break;
            case  Magma_BICGSTAB:
                    printf("%%   Preconditioner used: BiCGSTAB.\n"); break;
            case  Magma_GMRES:
                    printf("%%   Preconditioner used: GMRES.\n"); break;
            case  Magma_JACOBI:
                    printf("%%   Preconditioner used: Jacobi.\n"); break;
            case  Magma_BAITER:
                    printf("%%   Preconditioner used: Block-asynchronous iteration.\n"); break;
            case  Magma_ILU:
                    printf("%%   Preconditioner used: ILU(%d).\n", precond_par->levels); break;
            case  Magma_AILU:
                    printf("%%   Preconditioner used: iterative ILU(%d).\n", precond_par->levels); break;
            case  Magma_ICC:
                    printf("%%   Preconditioner used: IC(%d).\n", precond_par->levels); break;
            case  Magma_AICC:
                    printf("%%   Preconditioner used: iterative IC(%d).\n", precond_par->levels); break;
            default:
                  break;
    
        }
        
            printf("%%======================================================="
            "======%%\n");
        switch( solver_par->solver ) {
            case  Magma_CG:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_PCG:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_CGMERGE:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_BICGSTAB:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_PBICGSTAB:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_BICGSTABMERGE:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_BICGSTABMERGE2:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_GMRES:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_PGMRES:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            case  Magma_ITERREF:
                printf("%%   iter   ||   residual-nrm2    ||   runtime \n");
                printf("%%======================================================="
                        "======%%\n");
                for( int j=0; j<(solver_par->numiter)/k+1; j++ ) {
                    printf("   %4d    ||    %e    ||    %f\n",
                       (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
                }
                printf("%%======================================================="
                        "======%%\n"); break;
            default:
                printf("%%======================================================="
                        "======%%\n"); break;
        }
    }
                
    printf("\n%%======================================================="
        "======%%\n");
    switch( solver_par->solver ) {
        case  Magma_CG:
            printf("%% CG solver summary:\n"); break;
        case  Magma_PCG:
            printf("%% PCG solver summary:\n"); break;
        case  Magma_CGMERGE:
            printf("%% CG solver summary:\n"); break;
        case  Magma_BICGSTAB:
            printf("%% BiCGSTAB solver summary:\n"); break;
        case  Magma_PBICGSTAB:
            printf("%% PBiCGSTAB solver summary:\n"); break;
        case  Magma_BICGSTABMERGE:
            printf("%% BiCGSTAB solver summary:\n"); break;
        case  Magma_BICGSTABMERGE2:
            printf("%% BiCGSTAB solver summary:\n"); break;
        case  Magma_GMRES:
            printf("%% GMRES(%d) solver summary:\n", solver_par->restart); break;
        case  Magma_PGMRES:
            printf("%% PGMRES(%d) solver summary:\n", solver_par->restart); break;
        case  Magma_ITERREF:
            printf("%% Iterative refinement solver summary:\n"); break;
        case  Magma_JACOBI:
            printf("%% CG solver summary:\n"); break;
        case  Magma_BAITER:
            printf("%% Block-asynchronous iteration solver summary:\n"); break;
        case  Magma_LOBPCG:
            printf("%% LOBPCG iteration solver summary:\n"); break;
        default:
            printf("%%   Solver info not supported.\n"); goto cleanup;
    }
    printf("%%    initial residual: %e\n", solver_par->init_res );
    printf("%%    iterations: %4d\n", (int) (solver_par->numiter) );
    printf("%%    exact final residual: %e\n%%    runtime: %.4f sec\n",
        solver_par->final_res, solver_par->runtime);

cleanup:
    printf("%%======================================================="
        "======%%\n");
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Frees any memory assocoiated with the verbose mode of solver_par. The
    other values are set to default.

    Arguments
    ---------

    @param[in,out]
    solver_par  magma_s_solver_par*
                structure containing all solver information
    @param[in,out]
    precond_par magma_s_preconditioner*
                structure containing all preconditioner information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_ssolverinfo_free(
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    solver_par->init_res = 0.0;
    solver_par->iter_res = 0.0;
    solver_par->final_res = 0.0;

    if ( solver_par->res_vec != NULL ) {
        magma_free_cpu( solver_par->res_vec );
        solver_par->res_vec = NULL;
    }
    if ( solver_par->timing != NULL ) {
        magma_free_cpu( solver_par->timing );
        solver_par->timing = NULL;
    }
    if ( solver_par->eigenvectors != NULL ) {
        magma_free( solver_par->eigenvectors );
        solver_par->eigenvectors = NULL;
    }
    if ( solver_par->eigenvalues != NULL ) {
        magma_free_cpu( solver_par->eigenvalues );
        solver_par->eigenvalues = NULL;
    }
    if ( precond_par->d.val != NULL ) {
        magma_free( precond_par->d.val );
        precond_par->d.val = NULL;
    }
    if ( precond_par->d2.val != NULL ) {
        magma_free( precond_par->d2.val );
        precond_par->d2.val = NULL;
    }
    if ( precond_par->work1.val != NULL ) {
        magma_free( precond_par->work1.val );
        precond_par->work1.val = NULL;
    }
    if ( precond_par->work2.val != NULL ) {
        magma_free( precond_par->work2.val );
        precond_par->work2.val = NULL;
    }
    if ( precond_par->M.val != NULL ) {
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.dval );
        else
            magma_free_cpu( precond_par->M.val );
        precond_par->M.val = NULL;
    }
    if ( precond_par->M.col != NULL ) {
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.dcol );
        else
            magma_free_cpu( precond_par->M.col );
        precond_par->M.col = NULL;
    }
    if ( precond_par->M.row != NULL ) {
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.drow );
        else
            magma_free_cpu( precond_par->M.row );
        precond_par->M.row = NULL;
    }
    if ( precond_par->M.blockinfo != NULL ) {
        magma_free_cpu( precond_par->M.blockinfo );
        precond_par->M.blockinfo = NULL;
    }
    if ( precond_par->L.val != NULL ) {
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.dval );
        else
            magma_free_cpu( precond_par->L.val );
        precond_par->L.val = NULL;
    }
    if ( precond_par->L.col != NULL ) {
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.col );
        else
            magma_free_cpu( precond_par->L.dcol );
        precond_par->L.col = NULL;
    }
    if ( precond_par->L.row != NULL ) {
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.drow );
        else
            magma_free_cpu( precond_par->L.row );
        precond_par->L.row = NULL;
    }
    if ( precond_par->L.blockinfo != NULL ) {
        magma_free_cpu( precond_par->L.blockinfo );
        precond_par->L.blockinfo = NULL;
    }
    if ( precond_par->U.val != NULL ) {
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.dval );
        else
            magma_free_cpu( precond_par->U.val );
        precond_par->U.val = NULL;
    }
    if ( precond_par->U.col != NULL ) {
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.dcol );
        else
            magma_free_cpu( precond_par->U.col );
        precond_par->U.col = NULL;
    }
    if ( precond_par->U.row != NULL ) {
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.drow );
        else
            magma_free_cpu( precond_par->U.row );
        precond_par->U.row = NULL;
    }
    if ( precond_par->U.blockinfo != NULL ) {
        magma_free_cpu( precond_par->U.blockinfo );
        precond_par->U.blockinfo = NULL;
    }
    if ( precond_par->solver == Magma_ILU ||
        precond_par->solver == Magma_AILU ||
        precond_par->solver == Magma_ICC||
        precond_par->solver == Magma_AICC ) {
        cusparseDestroySolveAnalysisInfo( precond_par->cuinfoL ); 
        cusparseDestroySolveAnalysisInfo( precond_par->cuinfoU ); 
        precond_par->cuinfoL = NULL;
        precond_par->cuinfoU = NULL;

    }
    if ( precond_par->LD.val != NULL ) {
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.dval );
        else
            magma_free_cpu( precond_par->LD.val );
        precond_par->LD.val = NULL;
    }
    if ( precond_par->LD.col != NULL ) {
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.dcol );
        else
            magma_free_cpu( precond_par->LD.col );
        precond_par->LD.col = NULL;
    }
    if ( precond_par->LD.row != NULL ) {
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.drow );
        else
            magma_free_cpu( precond_par->LD.row );
        precond_par->LD.row = NULL;
    }
    if ( precond_par->LD.blockinfo != NULL ) {
        magma_free_cpu( precond_par->LD.blockinfo );
        precond_par->LD.blockinfo = NULL;
    }
    if ( precond_par->UD.val != NULL ) {
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.dval );
        else
            magma_free_cpu( precond_par->UD.val );
        precond_par->UD.val = NULL;
    }
    if ( precond_par->UD.col != NULL ) {
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.dcol );
        else
            magma_free_cpu( precond_par->UD.col );
        precond_par->UD.col = NULL;
    }
    if ( precond_par->UD.row != NULL ) {
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.drow );
        else
            magma_free_cpu( precond_par->UD.row );
        precond_par->UD.row = NULL;
    }
    if ( precond_par->UD.blockinfo != NULL ) {
        magma_free_cpu( precond_par->UD.blockinfo );
        precond_par->UD.blockinfo = NULL;
    }

    precond_par->solver = Magma_NONE;
    return MAGMA_SUCCESS;
}

/**
    Purpose
    -------

    Initializes all solver and preconditioner parameters.

    Arguments
    ---------

    @param[in,out]
    solver_par  magma_s_solver_par*
                structure containing all solver information
    @param[in,out]
    precond_par magma_s_preconditioner*
                structure containing all preconditioner information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_ssolverinfo_init(
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;
    solver_par->eigenvectors = NULL;
    solver_par->eigenvalues = NULL;

    if( solver_par->maxiter == 0 )
        solver_par->maxiter = 1000;
    if( solver_par->version == 0 )
        solver_par->version = 0;
    if( solver_par->restart == 0 )
        solver_par->restart = 30;
    if( solver_par->solver == 0 )
        solver_par->solver = Magma_CG;

    if ( solver_par->verbose > 0 ) {
        CHECK( magma_malloc_cpu( (void **)&solver_par->res_vec, sizeof(real_Double_t)
                * ( (solver_par->maxiter)/(solver_par->verbose)+1) ));
        CHECK( magma_malloc_cpu( (void **)&solver_par->timing, sizeof(real_Double_t)
                *( (solver_par->maxiter)/(solver_par->verbose)+1) ));
    } else {
        solver_par->res_vec = NULL;
        solver_par->timing = NULL;
    }

    precond_par->d.val = NULL;
    precond_par->d2.val = NULL;
    precond_par->work1.val = NULL;
    precond_par->work2.val = NULL;

    precond_par->M.val = NULL;
    precond_par->M.col = NULL;
    precond_par->M.row = NULL;
    precond_par->M.blockinfo = NULL;

    precond_par->L.val = NULL;
    precond_par->L.col = NULL;
    precond_par->L.row = NULL;
    precond_par->L.blockinfo = NULL;

    precond_par->U.val = NULL;
    precond_par->U.col = NULL;
    precond_par->U.row = NULL;
    precond_par->U.blockinfo = NULL;

    precond_par->LD.val = NULL;
    precond_par->LD.col = NULL;
    precond_par->LD.row = NULL;
    precond_par->LD.blockinfo = NULL;

    precond_par->UD.val = NULL;
    precond_par->UD.col = NULL;
    precond_par->UD.row = NULL;
    precond_par->UD.blockinfo = NULL;
    
    precond_par->cuinfoL = NULL;
    precond_par->cuinfoU = NULL;

cleanup:
    if( info != 0 ){
        magma_free( solver_par->timing );
        magma_free( solver_par->res_vec );
    }
    return info;
}


/**
    Purpose
    -------

    Initializes space for eigensolvers.

    Arguments
    ---------

    @param[in,out]
    solver_par  magma_s_solver_par*
                structure containing all solver information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_seigensolverinfo_init(
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    float *initial_guess=NULL;
    solver_par->eigenvectors = NULL;
    solver_par->eigenvalues = NULL;
    if ( solver_par->solver == Magma_LOBPCG ) {
        CHECK( magma_smalloc_cpu( &solver_par->eigenvalues ,
                                3*solver_par->num_eigenvalues ));
        // setup initial guess EV using lapack
        // then copy to GPU
        magma_int_t ev = solver_par->num_eigenvalues * solver_par->ev_length;

        CHECK( magma_smalloc_cpu( &initial_guess, ev ));
        CHECK( magma_smalloc( &solver_par->eigenvectors, ev ));
        magma_int_t ISEED[4] = {0,0,0,1}, ione = 1;
        lapackf77_slarnv( &ione, ISEED, &ev, initial_guess );

        magma_ssetmatrix( solver_par->ev_length, solver_par->num_eigenvalues,
            initial_guess, solver_par->ev_length, solver_par->eigenvectors,
                                                    solver_par->ev_length );
    } else {
        solver_par->eigenvectors = NULL;
        solver_par->eigenvalues = NULL;
    }

cleanup:
    if( info != 0 ){
        magma_free( solver_par->eigenvectors );
        magma_free( solver_par->eigenvalues );
    }
    magma_free_cpu( initial_guess );
    return info;
}


