/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from magma_zsolverinfo.cpp normal z -> s, Fri May 30 10:41:46 2014
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Prints information about a previously called solver.

    Arguments
    =========

    magma_s_solver_par *solver_par    structure containing all information

    ========================================================================  */


magma_int_t
magma_ssolverinfo( magma_s_solver_par *solver_par, 
                    magma_s_preconditioner *precond_par ){

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


/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from magma_zsolverinfo.cpp normal z -> s, Fri May 30 10:41:46 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Frees any memory assocoiated with the verbose mode of solver_par. The
    other values are set to default.

    Arguments
    =========

    magma_s_solver_par *solver_par    structure containing all information

    ========================================================================  */


magma_int_t
magma_ssolverinfo_free( magma_s_solver_par *solver_par, 
                        magma_s_preconditioner *precond ){

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

    if( precond->d.val != NULL ){
        magma_free( precond->d.val );
        precond->d.val = NULL;
    }
    if( precond->M.val != NULL ){
        if ( precond->M.memory_location == Magma_DEV )
            magma_free( precond->M.val );
        else
            magma_free_cpu( precond->M.val );
        precond->M.val = NULL;
    }
    if( precond->M.col != NULL ){
        if ( precond->M.memory_location == Magma_DEV )
            magma_free( precond->M.col );
        else
            magma_free_cpu( precond->M.col );
        precond->M.col = NULL;
    }
    if( precond->M.row != NULL ){
        if ( precond->M.memory_location == Magma_DEV )
            magma_free( precond->M.row );
        else
            magma_free_cpu( precond->M.row );
        precond->M.row = NULL;
    }
    if( precond->M.blockinfo != NULL ){
        magma_free_cpu( precond->M.blockinfo );
        precond->M.blockinfo = NULL;
    }
    if( precond->L.val != NULL ){
        if ( precond->L.memory_location == Magma_DEV )
            magma_free( precond->L.val );
        else
            magma_free_cpu( precond->L.val );
        precond->L.val = NULL;
    }
    if( precond->L.col != NULL ){
        if ( precond->L.memory_location == Magma_DEV )
            magma_free( precond->L.col );
        else
            magma_free_cpu( precond->L.col );
        precond->L.col = NULL;
    }
    if( precond->L.row != NULL ){
        if ( precond->L.memory_location == Magma_DEV )
            magma_free( precond->L.row );
        else
            magma_free_cpu( precond->L.row );
        precond->L.row = NULL;
    }
    if( precond->L.blockinfo != NULL ){
        magma_free_cpu( precond->L.blockinfo );
        precond->L.blockinfo = NULL;
    }
    if( precond->U.val != NULL ){
        if ( precond->U.memory_location == Magma_DEV )
            magma_free( precond->U.val );
        else
            magma_free_cpu( precond->U.val );
        precond->U.val = NULL;
    }
    if( precond->U.col != NULL ){
        if ( precond->U.memory_location == Magma_DEV )
            magma_free( precond->U.col );
        else
            magma_free_cpu( precond->U.col );
        precond->U.col = NULL;
    }
    if( precond->U.row != NULL ){
        if ( precond->U.memory_location == Magma_DEV )
            magma_free( precond->U.row );
        else
            magma_free_cpu( precond->U.row );
        precond->U.row = NULL;
    }
    if( precond->U.blockinfo != NULL ){
        magma_free_cpu( precond->U.blockinfo );
        precond->U.blockinfo = NULL;
    }

    if( precond->solver == Magma_ILU ){
        cusparseStatus_t cusparseStatus;
        cusparseStatus =
        cusparseDestroySolveAnalysisInfo( precond->cuinfo );
         if(cusparseStatus != 0)    printf("error in info-free.\n");
        cusparseStatus =
        cusparseDestroySolveAnalysisInfo( precond->cuinfoL );
         if(cusparseStatus != 0)    printf("error in info-free.\n");
        cusparseStatus =
        cusparseDestroySolveAnalysisInfo( precond->cuinfoU );
         if(cusparseStatus != 0)    printf("error in info-free.\n");

    }
    if( precond->LD.val != NULL ){
        if ( precond->LD.memory_location == Magma_DEV )
            magma_free( precond->LD.val );
        else
            magma_free_cpu( precond->LD.val );
        precond->LD.val = NULL;
    }
    if( precond->LD.col != NULL ){
        if ( precond->LD.memory_location == Magma_DEV )
            magma_free( precond->LD.col );
        else
            magma_free_cpu( precond->LD.col );
        precond->LD.col = NULL;
    }
    if( precond->LD.row != NULL ){
        if ( precond->LD.memory_location == Magma_DEV )
            magma_free( precond->LD.row );
        else
            magma_free_cpu( precond->LD.row );
        precond->LD.row = NULL;
    }
    if( precond->LD.blockinfo != NULL ){
        magma_free_cpu( precond->LD.blockinfo );
        precond->LD.blockinfo = NULL;
    }
    if( precond->UD.val != NULL ){
        if ( precond->UD.memory_location == Magma_DEV )
            magma_free( precond->UD.val );
        else
            magma_free_cpu( precond->UD.val );
        precond->UD.val = NULL;
    }
    if( precond->UD.col != NULL ){
        if ( precond->UD.memory_location == Magma_DEV )
            magma_free( precond->UD.col );
        else
            magma_free_cpu( precond->UD.col );
        precond->UD.col = NULL;
    }
    if( precond->UD.row != NULL ){
        if ( precond->UD.memory_location == Magma_DEV )
            magma_free( precond->UD.row );
        else
            magma_free_cpu( precond->UD.row );
        precond->UD.row = NULL;
    }
    if( precond->UD.blockinfo != NULL ){
        magma_free_cpu( precond->UD.blockinfo );
        precond->UD.blockinfo = NULL;
    }
    return MAGMA_SUCCESS;
}


magma_int_t
magma_ssolverinfo_init( magma_s_solver_par *solver_par, 
                        magma_s_preconditioner *precond ){

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

    if( solver_par->num_eigenvalues > 0 ){
        magma_smalloc_cpu( &solver_par->eigenvalues , 
                                3*solver_par->num_eigenvalues );

        // setup initial guess EV using lapack
        // then copy to GPU
        magma_int_t ev = solver_par->num_eigenvalues * solver_par->ev_length;
        float *initial_guess;
        magma_smalloc_cpu( &initial_guess, ev );
        magma_smalloc( &solver_par->eigenvectors, ev );
        magma_int_t ISEED[4] = {0,0,0,1}, ione = 1;
        lapackf77_slarnv( &ione, ISEED, &ev, initial_guess );
        magma_ssetmatrix( solver_par->ev_length, solver_par->num_eigenvalues, 
            initial_guess, solver_par->ev_length, solver_par->eigenvectors, 
                                                    solver_par->ev_length );

        magma_free_cpu( initial_guess );
    }else{
        solver_par->eigenvectors = NULL;
        solver_par->eigenvalues = NULL;
    }  

    precond->d.val = NULL;
    precond->M.val = NULL;
    precond->M.col = NULL;
    precond->M.row = NULL;
    precond->M.blockinfo = NULL;

    precond->L.val = NULL;
    precond->L.col = NULL;
    precond->L.row = NULL;
    precond->L.blockinfo = NULL;

    precond->U.val = NULL;
    precond->U.col = NULL;
    precond->U.row = NULL;
    precond->U.blockinfo = NULL;

    precond->LD.val = NULL;
    precond->LD.col = NULL;
    precond->LD.row = NULL;
    precond->LD.blockinfo = NULL;


    precond->UD.val = NULL;
    precond->UD.col = NULL;
    precond->UD.row = NULL;
    precond->UD.blockinfo = NULL;


    return MAGMA_SUCCESS;
}


