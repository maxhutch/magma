/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s

       @author Hartwig Anzt

       Utilities for testing MAGMA-sparse.
*/
#include <cuda_runtime_api.h>

#include "magmasparse_internal.h"

#define PRECISION_z

// --------------------
static const char *usage_sparse_short =
"%% Usage: %s [options] [-h|--help]  matrices\n\n";

static const char *usage_sparse =
"Options are:\n"
" --format      Possibility to choose a format for the sparse matrix:\n"
"               CSR, ELL, SELLP, CUSPARSECSR\n"
" --blocksize x Set a specific blocksize for SELL-P format.\n"
" --alignment x Set a specific alignment for SELL-P format.\n"
" --mscale      Possibility to scale the original matrix:\n"
"               0   no scaling\n"
"               1   symmetric scaling to unit diagonal\n"
" --solver      Possibility to choose a solver:\n"
"               CG, PCG, BICGSTAB, PBICGSTAB, GMRES, PGMRES, LOBPCG, JACOBI,\n"
"               BAITER, IDR, PIDR, CGS, PCGS, TFQMR, PTFQMR, QMR, BICG\n"
"               BOMBARDMENT, ITERREF.\n"
" --basic       Use non-optimized version\n"
" --restart     For GMRES: possibility to choose the restart.\n"
"               For IDR: Number of distinct subspaces (1,2,4,8).\n"
" --precond x   Possibility to choose a preconditioner:\n"
"               CG, BICGSTAB, GMRES, LOBPCG, JACOBI,\n"
"               BAITER, IDR, CGS, TFQMR, QMR, BICG\n"
"               BOMBARDMENT, ITERREF, ILU, AILU, NONE.\n"
"                   --patol atol  Absolute residual stopping criterion for preconditioner.\n"
"                   --prtol rtol  Relative residual stopping criterion for preconditioner.\n"
"                   --psweeps k   Iteration count for iterative incomplete factorizations.\n"
"                   --piter k     Iteration count for iterative preconditioner.\n"
"                   --plevels k   Number of ILU levels.\n"
"                   --psweeps x   Number of iterative ILU sweeps.\n"
" --ev x        For eigensolvers, set number of eigenvalues/eigenvectors to compute.\n"
" --verbose x   Possibility to print intermediate residuals every x iteration.\n"
" --maxiter x   Set an upper limit for the iteration count.\n"
" --atol atol   Set an absolute residual stopping criterion.\n"
" --rtol rtol   Set a relative residual stopping criterion.\n";



/**
    Purpose
    -------

    Parses input options for a solver

    Arguments
    ---------

    @param[in]
    argc            int
                    command line input
                
    @param[in]
    argv            char**
                    command line input

    @param[in,out]
    opts            magma_zopts *
                    magma solver options

    @param[out]
    matrices        int
                    counter how many linear systems to process

    @param[in]
    queue           magma_queue_t
                    Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zparse_opts(
    int argc,
    char** argv,
    magma_zopts *opts,
    int *matrices,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_SUCCESS;
    
    // fill in default values
    opts->input_format = Magma_CSR;
    opts->blocksize = 32;
    opts->alignment = 1;
    opts->output_format = Magma_CUCSR;
    opts->input_location = Magma_CPU;
    opts->output_location = Magma_CPU;
    opts->scaling = Magma_NOSCALE;
    #if defined(PRECISION_z) | defined(PRECISION_d)
        opts->solver_par.atol = 1e-16;
        opts->solver_par.rtol = 1e-10;
    #else
        opts->solver_par.atol = 1e-8;
        opts->solver_par.rtol = 1e-5;
    #endif
    opts->solver_par.maxiter = 1000;
    opts->solver_par.verbose = 0;
    opts->solver_par.version = 0;
    opts->solver_par.restart = 50;
    opts->solver_par.num_eigenvalues = 0;
    opts->precond_par.solver = Magma_NONE;
    #if defined(PRECISION_z) | defined(PRECISION_d)
        opts->precond_par.atol = 1e-16;
        opts->precond_par.rtol = 1e-10;
    #else
        opts->precond_par.atol = 1e-8;
        opts->precond_par.rtol = 1e-5;
    #endif
    opts->precond_par.maxiter = 100;
    opts->precond_par.restart = 10;
    opts->precond_par.levels = 0;
    opts->precond_par.sweeps = 5;
    opts->solver_par.solver = Magma_CGMERGE;
    
    printf( usage_sparse_short, argv[0] );
    
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    
    int basic = 0;

    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--format", argv[i]) == 0 && i+1 < argc ) {
            i++;
            if ( strcmp("CSR", argv[i]) == 0 ) {
                opts->output_format = Magma_CUCSR;
            } else if ( strcmp("ELL", argv[i]) == 0 ) {
                opts->output_format = Magma_ELL;
            } else if ( strcmp("SELLP", argv[i]) == 0 ) {
                opts->output_format = Magma_SELLP;
            } else if ( strcmp("CUSPARSECSR", argv[i]) == 0 ) {
                opts->output_format = Magma_CUCSR;
            } else {
                printf( "error: invalid format, use default (CSR).\n" );
            }
        } else if ( strcmp("--mscale", argv[i]) == 0 && i+1 < argc ) {
            i++;
            if ( strcmp("NOSCALE", argv[i]) == 0 ) {
                opts->scaling = Magma_NOSCALE;
            }
            else if ( strcmp("UNITDIAG", argv[i]) == 0 ) {
                opts->scaling = Magma_UNITDIAG;
            }
            else if ( strcmp("UNITROW", argv[i]) == 0 ) {
                opts->scaling = Magma_UNITROW;
            }
            else {
                printf( "error: invalid scaling, use default.\n" );
            }
        } else if ( strcmp("--solver", argv[i]) == 0 && i+1 < argc ) {
            i++;
            if ( strcmp("CG", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_CGMERGE;
            }
            else if ( strcmp("PCG", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PCGMERGE;
            }
            else if ( strcmp("BICG", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_BICG;
            }
            else if ( strcmp("PBICG", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PBICG;
            }
            else if ( strcmp("BICGSTAB", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_BICGSTABMERGE;
            }
            else if ( strcmp("PBICGSTAB", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PBICGSTABMERGE;
            }
            else if ( strcmp("QMR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_QMRMERGE;
            }
            else if ( strcmp("PQMR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PQMRMERGE;
            }
            else if ( strcmp("TFQMR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_TFQMRMERGE;
            }
            else if ( strcmp("PTFQMR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PTFQMRMERGE;
            }
            else if ( strcmp("GMRES", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_GMRES;
            }
            else if ( strcmp("PGMRES", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PGMRES;
            }
            else if ( strcmp("LOBPCG", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_LOBPCG;
            }
            else if ( strcmp("LSQR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_LSQR;
            }
            else if ( strcmp("JACOBI", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_JACOBI;
            }
            else if ( strcmp("BA", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_BAITER;
            }
            else if ( strcmp("BAO", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_BAITERO;
            }
            else if ( strcmp("IDR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_IDRMERGE;
            }
            else if ( strcmp("PIDR", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PIDRMERGE;
            }
            else if ( strcmp("CGS", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_CGSMERGE;
            }
            else if ( strcmp("PCGS", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_PCGSMERGE;
            }
            else if ( strcmp("BOMBARDMENT", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_BOMBARDMERGE;
            }
            else if ( strcmp("ITERREF", argv[i]) == 0 ) {
                opts->solver_par.solver = Magma_ITERREF;
            }
            else {
                printf( "error: invalid solver.\n" );
            }
        } else if ( strcmp("--restart", argv[i]) == 0 && i+1 < argc ) {
            opts->solver_par.restart = atoi( argv[++i] );
        } else if ( strcmp("--precond", argv[i]) == 0 && i+1 < argc ) {
            i++;
            if ( strcmp("CG", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_CGMERGE;
            }
            else if ( strcmp("PCG", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_PCG;
            }
            else if ( strcmp("BICGSTAB", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_BICGSTABMERGE;
            }
            else if ( strcmp("QMR", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_QMRMERGE;
            }
            else if ( strcmp("TFQMR", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_TFQMRMERGE;
            }
            else if ( strcmp("PTFQMR", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_PTFQMRMERGE;
            }
            else if ( strcmp("GMRES", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_GMRES;
            }
            else if ( strcmp("PGMRES", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_PGMRES;
            }
            else if ( strcmp("LOBPCG", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_LOBPCG;
            }
            else if ( strcmp("JACOBI", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_JACOBI;
            }
            else if ( strcmp("BA", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_BAITER;
            }
            else if ( strcmp("BAO", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_BAITERO;
            }
            else if ( strcmp("IDR", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_IDRMERGE;
            }
            else if ( strcmp("PIDR", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_PIDRMERGE;
            }
            else if ( strcmp("CGS", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_CGSMERGE;
            }
            else if ( strcmp("PCGS", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_PCGSMERGE;
            }
            else if ( strcmp("BOMBARDMENT", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_BOMBARD;
            }
            else if ( strcmp("ITERREF", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_ITERREF;
            }
            else if ( strcmp("ILU", argv[i]) == 0 || strcmp("IC", argv[i]) == 0 )  {
                opts->precond_par.solver = Magma_ILU;
            }
            else if ( strcmp("AILU", argv[i]) == 0 || strcmp("AIC", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_AILU;
            }
            else if ( strcmp("AICT", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_AICT;
            }
            else if ( strcmp("CUSTOMIC", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_CUSTOMIC;
            }
            else if ( strcmp("CUSTOMILU", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_CUSTOMILU;
            }
            else if ( strcmp("NONE", argv[i]) == 0 ) {
                opts->precond_par.solver = Magma_NONE;
            }
            else {
                printf( "error: invalid preconditioner.\n" );
            }
        } else if ( strcmp("--basic", argv[i]) == 0 && i+1 < argc ) {
            basic = 1;
        } else if ( strcmp("--patol", argv[i]) == 0 && i+1 < argc ) {
            sscanf( argv[++i], "%lf", &opts->precond_par.atol );
        }else if ( strcmp("--prtol", argv[i]) == 0 && i+1 < argc ) {
            sscanf( argv[++i], "%lf", &opts->precond_par.rtol );
        } else if ( strcmp("--piter", argv[i]) == 0 && i+1 < argc ) {
            opts->precond_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--psweeps", argv[i]) == 0 && i+1 < argc ) {
            opts->precond_par.sweeps = atoi( argv[++i] );
        } else if ( strcmp("--plevels", argv[i]) == 0 && i+1 < argc ) {
            opts->precond_par.levels = atoi( argv[++i] );
        } else if ( strcmp("--blocksize", argv[i]) == 0 && i+1 < argc ) {
            opts->blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 && i+1 < argc ) {
            opts->alignment = atoi( argv[++i] );
        } else if ( strcmp("--verbose", argv[i]) == 0 && i+1 < argc ) {
            opts->solver_par.verbose = atoi( argv[++i] );
        }  else if ( strcmp("--maxiter", argv[i]) == 0 && i+1 < argc ) {
            opts->solver_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--atol", argv[i]) == 0 && i+1 < argc ) {
            sscanf( argv[++i], "%lf", &opts->solver_par.atol );
        } else if ( strcmp("--rtol", argv[i]) == 0 && i+1 < argc ) {
            sscanf( argv[++i], "%lf", &opts->solver_par.rtol );
        } else if ( strcmp("--ev", argv[i]) == 0 && i+1 < argc ) {
            opts->solver_par.num_eigenvalues = atoi( argv[++i] );
        } else if ( strcmp("--version", argv[i]) == 0 && i+1 < argc ) {
            opts->solver_par.version = atoi( argv[++i] );
        }
        // ----- usage
        else if ( strcmp("-h",     argv[i]) == 0 ||
                  strcmp("--help", argv[i]) == 0 ) {
            fprintf( stderr, usage_sparse, argv[0] );
        } else {
            *matrices = i;
            break;
        }
    }
    if( basic == 1 ){
        if ( opts->solver_par.solver == Magma_CGMERGE ) {
            opts->solver_par.solver = Magma_CG;
        }
        else if ( opts->solver_par.solver == Magma_PCGMERGE) {
            opts->solver_par.solver = Magma_PCG;
        }
        else if ( opts->solver_par.solver == Magma_BICGSTABMERGE ) {
            opts->solver_par.solver = Magma_BICGSTAB;
        }
        else if ( opts->solver_par.solver == Magma_PBICGSTAB ) {
            opts->solver_par.solver = Magma_PBICGSTAB;
        }
        else if ( opts->solver_par.solver == Magma_TFQMRMERGE ) {
            opts->solver_par.solver = Magma_TFQMR;
        }
        else if ( opts->solver_par.solver == Magma_PTFQMRMERGE) {
            opts->solver_par.solver = Magma_PTFQMR;
        }
        else if ( opts->solver_par.solver == Magma_CGSMERGE ) {
            opts->solver_par.solver = Magma_CGS;
        }
        else if ( opts->solver_par.solver == Magma_PCGSMERGE) {
            opts->solver_par.solver = Magma_PCGS;
        }
        else if ( opts->solver_par.solver == Magma_QMRMERGE ) {
            opts->solver_par.solver = Magma_QMR;
        }
        else if ( opts->solver_par.solver == Magma_PQMRMERGE) {
            opts->solver_par.solver = Magma_PQMR;
        }
        else if ( opts->solver_par.solver == Magma_QMRMERGE ) {
            opts->solver_par.solver = Magma_QMR;
        }
        else if ( opts->solver_par.solver == Magma_PCGMERGE) {
            opts->solver_par.solver = Magma_PCG;
        }
        else if ( opts->solver_par.solver == Magma_IDRMERGE) {
            opts->solver_par.solver = Magma_IDR;
        }
        else if ( opts->solver_par.solver == Magma_PIDRMERGE) {
            opts->solver_par.solver = Magma_PIDR;
        }
        else if ( opts->solver_par.solver == Magma_BOMBARDMERGE) {
            opts->solver_par.solver = Magma_BOMBARD;
        }
    }
    
    // make sure preconditioner is NONE for unpreconditioned systems
    if ( opts->solver_par.solver != Magma_PCG &&
         opts->solver_par.solver != Magma_PCGMERGE &&
         opts->solver_par.solver != Magma_PGMRES &&
         opts->solver_par.solver != Magma_PBICGSTAB &&
         opts->solver_par.solver != Magma_PBICGSTABMERGE &&
         opts->solver_par.solver != Magma_ITERREF  &&
         opts->solver_par.solver != Magma_PIDR  &&
         opts->solver_par.solver != Magma_PIDRMERGE  &&
         opts->solver_par.solver != Magma_PCGS  &&
         opts->solver_par.solver != Magma_PCGSMERGE &&
         opts->solver_par.solver != Magma_PTFQMR &&
         opts->solver_par.solver != Magma_PTFQMRMERGE &&
         opts->solver_par.solver != Magma_PQMR &&
         opts->solver_par.solver != Magma_PBICG &&
         opts->solver_par.solver != Magma_LSQR &&
         opts->solver_par.solver != Magma_LOBPCG ){
                    opts->precond_par.solver = Magma_NONE;
         }
    
    // ensure to take a symmetric preconditioner for the PCG
    if ( ( opts->solver_par.solver == Magma_PCG || opts->solver_par.solver == Magma_PCGMERGE )
        && opts->precond_par.solver == Magma_ILU )
            opts->precond_par.solver = Magma_ICC;
    if ( ( opts->solver_par.solver == Magma_PCG || opts->solver_par.solver == Magma_PCGMERGE )
        && opts->precond_par.solver == Magma_AILU )
            opts->precond_par.solver = Magma_AICC;
            
            
    return info;
}
