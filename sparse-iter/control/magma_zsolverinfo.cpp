/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

/**
    Purpose
    -------

    Prints information about a previously called solver.

    Arguments
    ---------

    @param
    solver_par  magma_z_solver_par*
                structure containing all solver information
    @param
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zsolverinfo( magma_z_solver_par *solver_par, 
                    magma_z_preconditioner *precond_par ){

    if( (solver_par->solver == Magma_CG) || (solver_par->solver == Magma_PCG) ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            if( solver_par->solver == Magma_CG )
                printf("#   CG performance analysis every %d iteration\n", 
                                                                    (int) k);
            else if( solver_par->solver == Magma_PCG ){
                if( precond_par->solver == Magma_JACOBI )
                        printf("#   Jacobi-CG performance analysis"
                                " every %d iteration\n", (int) k);
                if( precond_par->solver == Magma_ICC )
                        printf("#   IC-CG performance analysis"
                                " every %d iteration\n", (int) k);

            }
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                   (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# CG solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int)(solver_par->numiter) );
        printf("#    iterative residual: %e\n", solver_par->iter_res );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                     solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_CGMERGE ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   CG (merged) performance analysis every %d iteration\n",
                                                                       (int) k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                   (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# CG (merged) solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int)(solver_par->numiter) );
        printf("#    iterative residual: %e\n", solver_par->iter_res );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BICGSTAB || 
                        solver_par->solver == Magma_PBICGSTAB ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            if( solver_par->solver == Magma_BICGSTAB )
                printf("#   BiCGStab performance analysis every %d iteration\n", 
                                                             (int) k);
            else if( solver_par->solver == Magma_PBICGSTAB ){
                if( precond_par->solver == Magma_JACOBI )
                        printf("#   Jacobi-BiCGStab performance analysis"
                                " every %d iteration\n", (int) k);
                if( precond_par->solver == Magma_ILU )
                        printf("#   ILU-BiCGStab performance analysis"
                                " every %d iteration\n", (int) k);
            }
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                  (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# BiCGStab solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    iterative residual: %e\n", solver_par->iter_res );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BICGSTABMERGE ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   BiCGStab (merged) performance analysis"
                   " every %d iteration\n", (int) k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                  (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# BiCGStab (merged) solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    iterative residual: %e\n", solver_par->iter_res );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BICGSTABMERGE2 ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   BiCGStab (merged2) performance analysis"
                   " every %d iteration\n", (int) k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                  (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# BiCGStab (merged2) solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    iterative residual: %e\n", solver_par->iter_res );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_GMRES || 
                        solver_par->solver == Magma_PGMRES ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            if( solver_par->solver == Magma_GMRES )
                printf("#   GMRES-(%d) performance analysis\n", 
                                                 (int) solver_par->restart);
            else if( solver_par->solver == Magma_PGMRES ){
                if( precond_par->solver == Magma_JACOBI )
                        printf("#   Jacobi-GMRES-(%d) performance analysis\n",
                                               (int) solver_par->restart);
                if( precond_par->solver == Magma_ILU )
                        printf("#   ILU-GMRES-(%d) performance analysis\n",
                                               (int) solver_par->restart);
            }
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                 (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# GMRES-(%d) solver summary:\n", (int) solver_par->restart);
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    iterative residual: %e\n", solver_par->iter_res );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_ITERREF ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("# Iterative Refinement performance analysis"
                   " every %d iteration\n", (int) k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                   (int) (j*k), solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# Iterative Refinement solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_JACOBI ){
        printf("#======================================================="
                "======#\n");
        printf("# Jacobi solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BAITER ){
        printf("#======================================================="
                "======#\n");
        printf("# Block-asynchronous iteration solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (int) (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BCSRLU ){
        printf("#======================================================="
                "======#\n");
        printf("# BCSRLU solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    exact final residual: %e\n", solver_par->final_res );
        printf("#    runtime factorization: %4f sec\n",
                    solver_par->timing[0] );
        printf("#    runtime triangular solve: %.4f sec\n", 
                    solver_par->timing[1] );
        printf("#======================================================="
                "======#\n");
    }else{
        printf("error: solver info not supported.\n");
    }

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Frees any memory assocoiated with the verbose mode of solver_par. The
    other values are set to default.

    Arguments
    ---------

    @param
    solver_par  magma_z_solver_par*
                structure containing all solver information
    @param
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information                
                

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zsolverinfo_free( magma_z_solver_par *solver_par, 
                        magma_z_preconditioner *precond_par ){

    solver_par->init_res = 0.0;
    solver_par->iter_res = 0.0;
    solver_par->final_res = 0.0;

    if( solver_par->res_vec != NULL ){
        magma_free_cpu( solver_par->res_vec );
        solver_par->res_vec = NULL;
    }
    if( solver_par->timing != NULL ){
        magma_free_cpu( solver_par->timing );
        solver_par->timing = NULL;
    }
    if( solver_par->eigenvectors != NULL ){
        magma_free( solver_par->eigenvectors );
        solver_par->eigenvectors = NULL;
    }
    if( solver_par->eigenvalues != NULL ){
        magma_free_cpu( solver_par->eigenvalues );
        solver_par->eigenvalues = NULL;
    }
    if( precond_par->d.val != NULL ){
        magma_free( precond_par->d.val );
        precond_par->d.val = NULL;
    }
    if( precond_par->work1.val != NULL ){
        magma_free( precond_par->work1.val );
        precond_par->work1.val = NULL;
    }
    if( precond_par->work2.val != NULL ){
        magma_free( precond_par->work2.val );
        precond_par->work2.val = NULL;
    }
    if( precond_par->M.val != NULL ){
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.val );
        else
            magma_free_cpu( precond_par->M.val );
        precond_par->M.val = NULL;
    }
    if( precond_par->M.col != NULL ){
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.col );
        else
            magma_free_cpu( precond_par->M.col );
        precond_par->M.col = NULL;
    }
    if( precond_par->M.row != NULL ){
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.row );
        else
            magma_free_cpu( precond_par->M.row );
        precond_par->M.row = NULL;
    }
    if( precond_par->M.blockinfo != NULL ){
        magma_free_cpu( precond_par->M.blockinfo );
        precond_par->M.blockinfo = NULL;
    }
    if( precond_par->L.val != NULL ){
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.val );
        else
            magma_free_cpu( precond_par->L.val );
        precond_par->L.val = NULL;
    }
    if( precond_par->L.col != NULL ){
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.col );
        else
            magma_free_cpu( precond_par->L.col );
        precond_par->L.col = NULL;
    }
    if( precond_par->L.row != NULL ){
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.row );
        else
            magma_free_cpu( precond_par->L.row );
        precond_par->L.row = NULL;
    }
    if( precond_par->L.blockinfo != NULL ){
        magma_free_cpu( precond_par->L.blockinfo );
        precond_par->L.blockinfo = NULL;
    }
    if( precond_par->U.val != NULL ){
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.val );
        else
            magma_free_cpu( precond_par->U.val );
        precond_par->U.val = NULL;
    }
    if( precond_par->U.col != NULL ){
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.col );
        else
            magma_free_cpu( precond_par->U.col );
        precond_par->U.col = NULL;
    }
    if( precond_par->U.row != NULL ){
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.row );
        else
            magma_free_cpu( precond_par->U.row );
        precond_par->U.row = NULL;
    }
    if( precond_par->U.blockinfo != NULL ){
        magma_free_cpu( precond_par->U.blockinfo );
        precond_par->U.blockinfo = NULL;
    }
    if( precond_par->solver == Magma_ILU ||
        precond_par->solver == Magma_AILU ||
        precond_par->solver == Magma_ICC||
        precond_par->solver == Magma_AICC ){
        cusparseStatus_t cusparseStatus;
        cusparseStatus =
        cusparseDestroySolveAnalysisInfo( precond_par->cuinfoL );
         if(cusparseStatus != 0)    printf("error in info-free.\n");
        cusparseStatus =
        cusparseDestroySolveAnalysisInfo( precond_par->cuinfoU );
         if(cusparseStatus != 0)    printf("error in info-free.\n");

    }
    if( precond_par->LD.val != NULL ){
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.val );
        else
            magma_free_cpu( precond_par->LD.val );
        precond_par->LD.val = NULL;
    }
    if( precond_par->LD.col != NULL ){
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.col );
        else
            magma_free_cpu( precond_par->LD.col );
        precond_par->LD.col = NULL;
    }
    if( precond_par->LD.row != NULL ){
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.row );
        else
            magma_free_cpu( precond_par->LD.row );
        precond_par->LD.row = NULL;
    }
    if( precond_par->LD.blockinfo != NULL ){
        magma_free_cpu( precond_par->LD.blockinfo );
        precond_par->LD.blockinfo = NULL;
    }
    if( precond_par->UD.val != NULL ){
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.val );
        else
            magma_free_cpu( precond_par->UD.val );
        precond_par->UD.val = NULL;
    }
    if( precond_par->UD.col != NULL ){
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.col );
        else
            magma_free_cpu( precond_par->UD.col );
        precond_par->UD.col = NULL;
    }
    if( precond_par->UD.row != NULL ){
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.row );
        else
            magma_free_cpu( precond_par->UD.row );
        precond_par->UD.row = NULL;
    }
    if( precond_par->UD.blockinfo != NULL ){
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

    @param
    solver_par  magma_z_solver_par*
                structure containing all solver information
    @param
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information                
                

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zsolverinfo_init( magma_z_solver_par *solver_par, 
                        magma_z_preconditioner *precond_par ){

/*
    solver_par->solver = Magma_CG;
    solver_par->maxiter = 1000;
    solver_par->numiter = 0;
    solver_par->ortho = Magma_CGS;
    solver_par->epsilon = RTOLERANCE;
    solver_par->restart = 30;
    solver_par->init_res = 0.;
    solver_par->final_res = 0.;
    solver_par->runtime = 0.;
    solver_par->verbose = 0;
    solver_par->info = 0;
*/
    if( solver_par->verbose > 0 ){
        magma_malloc_cpu( (void **)&solver_par->res_vec, sizeof(real_Double_t) 
                * ( (solver_par->maxiter)/(solver_par->verbose)+1) );
        magma_malloc_cpu( (void **)&solver_par->timing, sizeof(real_Double_t) 
                *( (solver_par->maxiter)/(solver_par->verbose)+1) );
    }else{
        solver_par->res_vec = NULL;
        solver_par->timing = NULL;
    }  

    if( solver_par->solver == Magma_LOBPCG ){
        magma_dmalloc_cpu( &solver_par->eigenvalues , 
                                3*solver_par->num_eigenvalues );

        // setup initial guess EV using lapack
        // then copy to GPU
        magma_int_t ev = solver_par->num_eigenvalues * solver_par->ev_length;
        magmaDoubleComplex *initial_guess;
        magma_zmalloc_cpu( &initial_guess, ev );
        magma_zmalloc( &solver_par->eigenvectors, ev );
        magma_int_t ISEED[4] = {0,0,0,1}, ione = 1;
        lapackf77_zlarnv( &ione, ISEED, &ev, initial_guess );
        magma_zsetmatrix( solver_par->ev_length, solver_par->num_eigenvalues, 
            initial_guess, solver_par->ev_length, solver_par->eigenvectors, 
                                                    solver_par->ev_length );

        magma_free_cpu( initial_guess );
    }else{
        solver_par->eigenvectors = NULL;
        solver_par->eigenvalues = NULL;
    }  

    precond_par->d.val = NULL;
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


    return MAGMA_SUCCESS;
}


