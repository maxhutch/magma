/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @author Mark Gates
       
       Utilities for testing.
*/

#include <string.h>
#include <assert.h>
#include <errno.h>

// flock exists only on Unix
#ifdef USE_FLOCK
#include <sys/file.h>  // flock
#include <sys/stat.h>  // fchmod
#endif

#if defined(HAVE_CUBLAS)
#include <cuda_runtime_api.h>
#endif

#include "testings.h"
#include "magma.h"

// --------------------
// global variable
#if   defined(HAVE_CUBLAS)
    const char* g_platform_str = "cuBLAS";

#elif defined(HAVE_clBLAS)
    const char* g_platform_str = "clBLAS";

#elif defined(HAVE_MIC)
    const char* g_platform_str = "Xeon Phi";

#else
    #error "unknown platform"
#endif


// --------------------
// If condition is false, print error message and exit.
// Error message is formatted using printf, using any additional arguments.
extern "C"
void magma_assert( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        printf( "Assert failed: " );
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        printf( "\n" );
        exit(1);
    }
}

// --------------------
// If condition is false, print warning message; does not exit.
// Warning message is formatted using printf, using any additional arguments.
extern "C"
void magma_assert_warn( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        printf( "Assert failed: " );
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        printf( "\n" );
    }
}


// --------------------
// Acquire lock file.
// operation should be LOCK_SH (for shared access) or LOCK_EX (for exclusive access).
// Returns open file descriptor.
// Exits program on error.
// Lock is released by simply closing the file descriptor with close(),
// or when program exits or crashes.

int open_lockfile( const char* file, int operation )
{
    int fd = -1;
#ifdef USE_FLOCK
    int err;

    if ( file == NULL )
        return -1;
    else if ( operation != LOCK_SH && operation != LOCK_EX )
        return -2;
    
    fd = open( file, O_RDONLY|O_CREAT, 0666 );
    if ( fd < 0 ) {
        fprintf( stderr, "Error: Can't read file %s: %s (%d)\n",
                 file, strerror(errno), errno );
        exit(1);
    }

    // make it world-writable so anyone can rm the lockfile later on if needed
    // Ignore error -- occurs when someone else created the file.
    err = fchmod( fd, 0666 );
    //if ( err < 0 ) {
    //    fprintf( stderr, "Warning: Can't chmod file %s 0666: %s (%d)\n",
    //             file, strerror(errno), errno );
    //}
    
    // first try nonblocking lock;
    // if that fails (e.g., someone has exclusive lock) let user know and try blocking lock.
    err = flock( fd, operation|LOCK_NB );
    if ( err < 0 ) {
        fprintf( stderr, "Waiting for lock on %s...\n", file );
        err = flock( fd, operation );
        if ( err < 0 ) {
            fprintf( stderr, "Error: Can't lock file %s (operation %d): %s (%d)\n",
                     file, operation, strerror(errno), errno );
            exit(1);
        }
    }
#endif
    return fd;
}

// filename to use for lock file
const char* lockfile = "/tmp/icl-lock";


// --------------------
const char *usage_short =
"%% Usage: %s [options] [-h|--help]\n\n";

const char *usage =
"Options are:\n"
"  --range m0:m1:mstep[,n0:n1:nstep[,k0:k1:kstep]]\n"
"                   Adds test cases with range m = m0, m0+mstep, ..., m1;\n"
"                   similarly for n, k. Can be repeated.\n"
"  -N m[,n[,k]]     Adds one test case with sizes m,n,k. Can be repeated.\n"
"                   If only m,n given then k=n. If only m given then n=k=m.\n"
"  -m m             Sets m for all tests, overriding -N and --range.\n"
"  -n n             Sets n for all tests, overriding -N and --range.\n"
"  -k k             Sets k for all tests, overriding -N and --range.\n"
"  Default test sizes are the range 1088 : 10304 : 1024, that is, 1K+64 : 10K+64 : 1K.\n"
"  For batched, default sizes are     32 :   512 :   32.\n"
"\n"
"  -c  --[no]check  Whether to check results. Some tests always check.\n"
"                   Also set with $MAGMA_TESTINGS_CHECK.\n"
"  -c2 --check2     For getrf, check residual |Ax-b| instead of |PA-LU|.\n"
"  -l  --[no]lapack Whether to run lapack. Some tests always run lapack.\n"
"                   Also set with $MAGMA_RUN_LAPACK.\n"
"      --[no]warmup Whether to warmup. Not yet implemented in most cases.\n"
"                   Also set with $MAGMA_WARMUP.\n"
"      --[not]all   Whether to test all combinations of flags, e.g., jobu.\n"
"  --dev x          GPU device to use, default 0.\n"
"  --align n        Round up LDDA on GPU to multiple of align, default 32.\n"
"  --verbose        Verbose output.\n"
"  -x  --exclusive  Lock file for exclusive use (internal ICL functionality).\n"
"\n"
"The following options apply to only some routines.\n"
"  --batch x        number of matrices for the batched routines, default 1000.\n"
"  --nb x           Block size, default set automatically.\n"
"  --nrhs x         Number of right hand sides, default 1.\n"
"  --nstream x      Number of CUDA streams, default 1.\n"
"  --ngpu x         Number of GPUs, default 1. Also set with $MAGMA_NUM_GPUS.\n"
"  --nsub x         Number of submatrices, default 1.\n"
"  --niter x        Number of iterations to repeat each test, default 1.\n"
"  --nthread x      Number of CPU threads, default 1.\n"
"  --offset x       Offset from beginning of matrix, default 0.\n"
"  --itype [123]    Generalized Hermitian-definite eigenproblem type, default 1.\n"
"  --svd_work [0123] SVD workspace size, from min (1) to optimal (3), or query (0), default 0.\n"
"  --version x      version of routine, e.g., during development, default 1.\n"
"  --fraction x     fraction of eigenvectors to compute, default 1.\n"
"  --tolerance x    accuracy tolerance, multiplied by machine epsilon, default 30.\n"
"  --tol x          same.\n"
"  --panel_nthread x Number of threads in the first dimension if the panel is decomposed into a 2D layout, default 1.\n"
"  --fraction_dcpu x Percentage of the workload to schedule on the cpu. Used in magma_amc algorithms only, default 0.\n"
"  -L -U -F         uplo   = Lower*, Upper, or Full.\n"
"  -[NTC][NTC]      transA = NoTrans*, Trans, or ConjTrans (first letter) and\n"
"                   transB = NoTrans*, Trans, or ConjTrans (second letter).\n"
"  -[TC]            transA = Trans or ConjTrans. Default is NoTrans. Doesn't change transB.\n"
"  -S[LR]           side   = Left*, Right.\n"
"  -D[NU]           diag   = NonUnit*, Unit.\n"
"  -U[NASO]         jobu   = No*, All, Some, or Overwrite; compute left  singular vectors. gesdd uses this for jobz.\n"
"  -V[NASO]         jobvt  = No*, All, Some, or Overwrite; compute right singular vectors.\n"
"  -J[NV]           jobz   = No* or Vectors; compute eigenvectors (symmetric).\n"
"  -L[NV]           jobvl  = No* or Vectors; compute left  eigenvectors (non-symmetric).\n"
"  -R[NV]           jobvr  = No* or Vectors; compute right eigenvectors (non-symmetric).\n"
"                   * default values\n";


// constructor fills in default values
magma_opts::magma_opts( magma_opts_t flag )
{
    // fill in default values
    this->batchcount = 300;
    this->device   = 0;
    this->align    = 32;
    this->nb       = 0;  // auto
    this->nrhs     = 1;
    this->nstream  = 1;
    this->ngpu     = magma_num_gpus();
    this->nsub     = 1;
    this->niter    = 1;
    this->nthread  = 1;
    this->offset   = 0;
    this->itype    = 1;
    this->svd_work = 0;
    this->version  = 1;
    this->fraction = 1.;
    this->tolerance = 30.;
    this->panel_nthread = 1;
    this->fraction_dcpu = 0.0;
    this->check     = (getenv("MAGMA_TESTINGS_CHECK") != NULL);
    this->lapack    = (getenv("MAGMA_RUN_LAPACK")     != NULL);
    this->warmup    = (getenv("MAGMA_WARMUP")         != NULL);
    this->all       = (getenv("MAGMA_RUN_ALL")        != NULL);
    this->verbose   = false;
    
    this->uplo      = MagmaLower;      // potrf, etc.
    this->transA    = MagmaNoTrans;    // gemm, etc.
    this->transB    = MagmaNoTrans;    // gemm
    this->side      = MagmaLeft;       // trsm, etc.
    this->diag      = MagmaNonUnit;    // trsm, etc.
    this->jobu      = MagmaNoVec;  // gesvd: no left  singular vectors
    this->jobvt     = MagmaNoVec;  // gesvd: no right singular vectors
    this->jobz      = MagmaNoVec;  // heev:  no eigen vectors
    this->jobvr     = MagmaNoVec;  // geev:  no right eigen vectors
    this->jobvl     = MagmaNoVec;  // geev:  no left  eigen vectors
    
    #ifdef USE_FLOCK
    this->flock_op = LOCK_SH;  // default shared lock
    #endif
    
    if ( flag == MagmaOptsBatched ) {
        // 32, 64, ..., 512
        this->default_nstart = 32;
        this->default_nstep  = 32;
        this->default_nend   = 512;
    }
    else {
        // 1K + 64, 2K + 64, ..., 10K + 64
        this->default_nstart = 1024 + 64;
        this->default_nstep  = 1024;
        this->default_nend   = 10304;
    }
}


// parse values from command line
void magma_opts::parse_opts( int argc, char** argv )
{
    printf( usage_short, argv[0] );
    
    // negative flag indicating -m, -n, -k not given
    int m = -1;
    int n = -1;
    int k = -1;
    
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    
    int info;
    this->ntest = 0;
    for( int i = 1; i < argc; ++i ) {
        // ----- matrix size
        // each -N fills in next entry of msize, nsize, ksize and increments ntest
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( this->ntest < MAX_NTEST, "error: -N %s, max number of tests exceeded, ntest=%d.\n",
                          argv[i], this->ntest );
            i++;
            int m2, n2, k2;
            info = sscanf( argv[i], "%d,%d,%d", &m2, &n2, &k2 );
            if ( info == 3 && m2 >= 0 && n2 >= 0 && k2 >= 0 ) {
                this->msize[ this->ntest ] = m2;
                this->nsize[ this->ntest ] = n2;
                this->ksize[ this->ntest ] = k2;
            }
            else if ( info == 2 && m2 >= 0 && n2 >= 0 ) {
                this->msize[ this->ntest ] = m2;
                this->nsize[ this->ntest ] = n2;
                this->ksize[ this->ntest ] = n2;  // implicitly
            }
            else if ( info == 1 && m2 >= 0 ) {
                this->msize[ this->ntest ] = m2;
                this->nsize[ this->ntest ] = m2;  // implicitly
                this->ksize[ this->ntest ] = m2;  // implicitly
            }
            else {
                fprintf( stderr, "error: -N %s is invalid; ensure m >= 0, n >= 0, k >= 0.\n",
                         argv[i] );
                exit(1);
            }
            this->ntest++;
        }
        // --range start:stop:step fills in msize[ntest:], nsize[ntest:], ksize[ntest:]
        // with given range and updates ntest
        else if ( strcmp("--range", argv[i]) == 0 && i+1 < argc ) {
            i++;
            int start_m, stop_m, step_m;
            int start_n, stop_n, step_n;
            int start_k, stop_k, step_k;
            
            info = sscanf( argv[i], "%d:%d:%d,%d:%d:%d,%d:%d:%d",
                           &start_m, &stop_m, &step_m,
                           &start_n, &stop_n, &step_n,
                           &start_k, &stop_k, &step_k );
            if ( info == 9 ) {
                // matched --range m1:m2:mstep,n1:n2:nstep,k1:k2:kstep
                magma_assert( start_m >= 0 && stop_m >= 0 &&
                              start_n >= 0 && stop_n >= 0 &&
                              start_k >= 0 && stop_k >= 0 &&
                              (step_m != 0 || step_n != 0 || step_k != 0),
                              "error: --range %s is invalid; ensure start >= 0, stop >= 0, at least one step != 0.\n", argv[i] );
                for( int lm = start_m, ln = start_n, lk = start_k;
                     (step_m >= 0 ? lm <= stop_m : lm >= stop_m) &&
                     (step_n >= 0 ? ln <= stop_n : ln >= stop_n) &&
                     (step_k >= 0 ? lk <= stop_k : lk >= stop_k);
                     lm += step_m, ln += step_n, lk += step_k )
                {
                    magma_assert( this->ntest < MAX_NTEST, "error: --range %s exceeded maximum number of tests (%d).\n",
                                  argv[1], MAX_NTEST );
                    this->msize[ this->ntest ] = lm;
                    this->nsize[ this->ntest ] = ln;
                    this->ksize[ this->ntest ] = lk;
                    this->ntest++;
                }
                continue;
            }
            else if ( info == 6 ) {
                // matched --range m1:m2:mstep,n1:n2:nstep
                magma_assert( start_m >= 0 && stop_m >= 0 &&
                              start_n >= 0 && stop_n >= 0 &&
                              (step_m != 0 || step_n != 0),
                              "error: --range %s is invalid; ensure start >= 0, stop >= 0, at least one step != 0.\n", argv[i] );
                for( int lm = start_m, ln = start_n;
                     (step_m >= 0 ? lm <= stop_m : lm >= stop_m) &&
                     (step_n >= 0 ? ln <= stop_n : ln >= stop_n);
                     lm += step_m, ln += step_n )
                {
                    magma_assert( this->ntest < MAX_NTEST, "error: --range %s exceeded maximum number of tests (%d).\n",
                                  argv[1], MAX_NTEST );
                    this->msize[ this->ntest ] = lm;
                    this->nsize[ this->ntest ] = ln;
                    this->ksize[ this->ntest ] = ln;
                    this->ntest++;
                }
                continue;
            }
            else if ( info == 3 ) {
                // matched --range n1:n2:nstep
                magma_assert( start_m >= 0 && stop_m >= 0 && step_m != 0,
                              "error: --range %s is invalid; ensure start >= 0, stop >= 0, step != 0.\n", argv[i] );
                for( int lm = start_m;
                     (step_m >= 0 ? lm <= stop_m : lm >= stop_m);
                     lm += step_m )
                {
                    magma_assert( this->ntest < MAX_NTEST, "error: --range %s exceeded maximum number of tests (%d).\n",
                                  argv[1], MAX_NTEST );
                    this->msize[ this->ntest ] = lm;
                    this->nsize[ this->ntest ] = lm;
                    this->ksize[ this->ntest ] = lm;
                    this->ntest++;
                }
                continue;
            }
            else {
                // didn't match above cases: invalid
                magma_assert( false, "error: --range %s is invalid; expect --range m0:m1:mstep[,n0:n1:nstep[,k0:k1:kstep]].\n", argv[i] );
            }
        }
        // save m, n, k if -m, -n, -k is given; applied after loop
        else if ( strcmp("-m", argv[i]) == 0 && i+1 < argc ) {
            m = atoi( argv[++i] );
            magma_assert( m >= 0, "error: -m %s is invalid; ensure m >= 0.\n", argv[i] );
        }
        else if ( strcmp("-n", argv[i]) == 0 && i+1 < argc ) {
            n = atoi( argv[++i] );
            magma_assert( n >= 0, "error: -n %s is invalid; ensure n >= 0.\n", argv[i] );
        }
        else if ( strcmp("-k", argv[i]) == 0 && i+1 < argc ) {
            k = atoi( argv[++i] );
            magma_assert( k >= 0, "error: -k %s is invalid; ensure k >= 0.\n", argv[i] );
        }
        
        // ----- scalar arguments
        else if ( strcmp("--dev", argv[i]) == 0 && i+1 < argc ) {
            this->device = atoi( argv[++i] );
            magma_assert( this->device >= 0 && this->device < ndevices,
                          "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices-1 );
        }
        else if ( strcmp("--align", argv[i]) == 0 && i+1 < argc ) {
            this->align = atoi( argv[++i] );
            magma_assert( this->align >= 1 && this->align <= 4096,
                          "error: --align %s is invalid; ensure align in [1,4096].\n", argv[i] );
        }
        else if ( strcmp("--nrhs",    argv[i]) == 0 && i+1 < argc ) {
            this->nrhs = atoi( argv[++i] );
            magma_assert( this->nrhs >= 0,
                          "error: --nrhs %s is invalid; ensure nrhs >= 0.\n", argv[i] );
        }
        else if ( strcmp("--nb",      argv[i]) == 0 && i+1 < argc ) {
            this->nb = atoi( argv[++i] );
            magma_assert( this->nb > 0,
                          "error: --nb %s is invalid; ensure nb > 0.\n", argv[i] );
        }
        else if ( strcmp("--ngpu",    argv[i]) == 0 && i+1 < argc ) {
            this->ngpu = atoi( argv[++i] );
            magma_assert( this->ngpu <= MagmaMaxGPUs,
                          "error: --ngpu %s exceeds MagmaMaxGPUs, %d.\n", argv[i], MagmaMaxGPUs );
            magma_assert( this->ngpu <= ndevices,
                          "error: --ngpu %s exceeds number of CUDA devices, %d.\n", argv[i], ndevices );
            magma_assert( this->ngpu > 0,
                          "error: --ngpu %s is invalid; ensure ngpu > 0.\n", argv[i] );
            // save in environment variable, so magma_num_gpus() picks it up
            #if defined( _WIN32 ) || defined( _WIN64 )
                char env_num_gpus[20] = "MAGMA_NUM_GPUS=";  // space for 4 digits & nil
                strncat( env_num_gpus, argv[i], sizeof(env_num_gpus) - strlen(env_num_gpus) - 1 );
                putenv( env_num_gpus );
            #else
                setenv( "MAGMA_NUM_GPUS", argv[i], true );
            #endif
        }
        else if ( strcmp("--nsub", argv[i]) == 0 && i+1 < argc ) {
            this->nsub = atoi( argv[++i] );
            magma_assert( this->nsub > 0,
                          "error: --nsub %s is invalid; ensure nsub > 0.\n", argv[i] );
        }
        else if ( strcmp("--nstream", argv[i]) == 0 && i+1 < argc ) {
            this->nstream = atoi( argv[++i] );
            magma_assert( this->nstream > 0,
                          "error: --nstream %s is invalid; ensure nstream > 0.\n", argv[i] );
        }
        else if ( strcmp("--niter",   argv[i]) == 0 && i+1 < argc ) {
            this->niter = atoi( argv[++i] );
            magma_assert( this->niter > 0,
                          "error: --niter %s is invalid; ensure niter > 0.\n", argv[i] );
        }
        else if ( strcmp("--nthread", argv[i]) == 0 && i+1 < argc ) {
            this->nthread = atoi( argv[++i] );
            magma_assert( this->nthread > 0,
                          "error: --nthread %s is invalid; ensure nthread > 0.\n", argv[i] );
        }
        else if ( strcmp("--offset", argv[i]) == 0 && i+1 < argc ) {
            this->offset = atoi( argv[++i] );
            magma_assert( this->offset >= 0,
                          "error: --offset %s is invalid; ensure offset >= 0.\n", argv[i] );
        }
        else if ( strcmp("--itype",   argv[i]) == 0 && i+1 < argc ) {
            this->itype = atoi( argv[++i] );
            magma_assert( this->itype >= 1 && this->itype <= 3,
                          "error: --itype %s is invalid; ensure itype in [1,2,3].\n", argv[i] );
        }
        else if ( strcmp("--svd_work", argv[i]) == 0 && i+1 < argc ) {
            this->svd_work = atoi( argv[++i] );
            magma_assert( this->svd_work >= 0 && this->svd_work <= 3,
                          "error: --svd_work %s is invalid; ensure svd_work in [0,1,2,3].\n", argv[i] );
        }
        else if ( strcmp("--version", argv[i]) == 0 && i+1 < argc ) {
            this->version = atoi( argv[++i] );
            magma_assert( this->version >= 1,
                          "error: --version %s is invalid; ensure version > 0.\n", argv[i] );
        }
        else if ( strcmp("--fraction", argv[i]) == 0 && i+1 < argc ) {
            this->fraction = atof( argv[++i] );
            magma_assert( this->fraction >= 0 && this->fraction <= 1,
                          "error: --fraction %s is invalid; ensure fraction in [0,1].\n", argv[i] );
        }
        else if ( (strcmp("--tol",       argv[i]) == 0 ||
                   strcmp("--tolerance", argv[i]) == 0) && i+1 < argc ) {
            this->tolerance = atof( argv[++i] );
            magma_assert( this->tolerance >= 0 && this->tolerance <= 1000,
                          "error: --tolerance %s is invalid; ensure tolerance in [0,1000].\n", argv[i] );
        }
        else if ( strcmp("--panel_nthread", argv[i]) == 0 && i+1 < argc ) {
            this->panel_nthread = atoi( argv[++i] );
            magma_assert( this->panel_nthread > 0,
                          "error: --panel_nthread %s is invalid; ensure panel_nthread > 0.\n", argv[i] );
        }
        else if ( strcmp("--fraction_dcpu", argv[i]) == 0 && i+1 < argc ) {
            this->fraction_dcpu = atof( argv[++i] );
            magma_assert( this->fraction_dcpu > 0 && this->fraction_dcpu <= 1,
                          "error: --fraction_dcpu %s is invalid; ensure fraction_dcpu in [0, 1]\n", argv[i] );
        }
        else if ( strcmp("--batch", argv[i]) == 0 && i+1 < argc ) {
            this->batchcount = atoi( argv[++i] );
            magma_assert( this->batchcount > 0,
                          "error: --batch %s is invalid; ensure batch > 0.\n", argv[i] );
        }
        // ----- boolean arguments
        // check results
        else if ( strcmp("-c",         argv[i]) == 0 ||
                  strcmp("--check",    argv[i]) == 0 ) { this->check  = 1; }
        else if ( strcmp("-c2",        argv[i]) == 0 ||
                  strcmp("--check2",   argv[i]) == 0 ) { this->check  = 2; }
        else if ( strcmp("--nocheck",  argv[i]) == 0 ) { this->check  = 0; }
        else if ( strcmp("-l",         argv[i]) == 0 ||
                  strcmp("--lapack",   argv[i]) == 0 ) { this->lapack = true;  }
        else if ( strcmp("--nolapack", argv[i]) == 0 ) { this->lapack = false; }
        else if ( strcmp("--warmup",   argv[i]) == 0 ) { this->warmup = true;  }
        else if ( strcmp("--nowarmup", argv[i]) == 0 ) { this->warmup = false; }
        else if ( strcmp("--all",      argv[i]) == 0 ) { this->all    = true;  }
        else if ( strcmp("--notall",   argv[i]) == 0 ) { this->all    = false; }
        else if ( strcmp("--verbose",  argv[i]) == 0 ) { this->verbose= true;  }
        
        // ----- lapack flag arguments
        else if ( strcmp("-L",  argv[i]) == 0 ) { this->uplo = MagmaLower; }
        else if ( strcmp("-U",  argv[i]) == 0 ) { this->uplo = MagmaUpper; }
        else if ( strcmp("-F",  argv[i]) == 0 ) { this->uplo = MagmaUpperLower; }
        
        else if ( strcmp("-NN", argv[i]) == 0 ) { this->transA = MagmaNoTrans;   this->transB = MagmaNoTrans;   }
        else if ( strcmp("-NT", argv[i]) == 0 ) { this->transA = MagmaNoTrans;   this->transB = MagmaTrans;     }
        else if ( strcmp("-NC", argv[i]) == 0 ) { this->transA = MagmaNoTrans;   this->transB = MagmaConjTrans; }
        else if ( strcmp("-TN", argv[i]) == 0 ) { this->transA = MagmaTrans;     this->transB = MagmaNoTrans;   }
        else if ( strcmp("-TT", argv[i]) == 0 ) { this->transA = MagmaTrans;     this->transB = MagmaTrans;     }
        else if ( strcmp("-TC", argv[i]) == 0 ) { this->transA = MagmaTrans;     this->transB = MagmaConjTrans; }
        else if ( strcmp("-CN", argv[i]) == 0 ) { this->transA = MagmaConjTrans; this->transB = MagmaNoTrans;   }
        else if ( strcmp("-CT", argv[i]) == 0 ) { this->transA = MagmaConjTrans; this->transB = MagmaTrans;     }
        else if ( strcmp("-CC", argv[i]) == 0 ) { this->transA = MagmaConjTrans; this->transB = MagmaConjTrans; }
        else if ( strcmp("-T",  argv[i]) == 0 ) { this->transA = MagmaTrans;     }
        else if ( strcmp("-C",  argv[i]) == 0 ) { this->transA = MagmaConjTrans; }
        
        else if ( strcmp("-SL", argv[i]) == 0 ) { this->side  = MagmaLeft;  }
        else if ( strcmp("-SR", argv[i]) == 0 ) { this->side  = MagmaRight; }
        
        else if ( strcmp("-DN", argv[i]) == 0 ) { this->diag  = MagmaNonUnit; }
        else if ( strcmp("-DU", argv[i]) == 0 ) { this->diag  = MagmaUnit;    }
        
        else if ( strcmp("-UA", argv[i]) == 0 ) { this->jobu  = MagmaAllVec;       }
        else if ( strcmp("-US", argv[i]) == 0 ) { this->jobu  = MagmaSomeVec;      }
        else if ( strcmp("-UO", argv[i]) == 0 ) { this->jobu  = MagmaOverwriteVec; }
        else if ( strcmp("-UN", argv[i]) == 0 ) { this->jobu  = MagmaNoVec;        }
        
        else if ( strcmp("-VA", argv[i]) == 0 ) { this->jobvt = MagmaAllVec;       }
        else if ( strcmp("-VS", argv[i]) == 0 ) { this->jobvt = MagmaSomeVec;      }
        else if ( strcmp("-VO", argv[i]) == 0 ) { this->jobvt = MagmaOverwriteVec; }
        else if ( strcmp("-VN", argv[i]) == 0 ) { this->jobvt = MagmaNoVec;        }
        
        else if ( strcmp("-JN", argv[i]) == 0 ) { this->jobz  = MagmaNoVec; }
        else if ( strcmp("-JV", argv[i]) == 0 ) { this->jobz  = MagmaVec;   }
        
        else if ( strcmp("-LN", argv[i]) == 0 ) { this->jobvl = MagmaNoVec; }
        else if ( strcmp("-LV", argv[i]) == 0 ) { this->jobvl = MagmaVec;   }
        
        else if ( strcmp("-RN", argv[i]) == 0 ) { this->jobvr = MagmaNoVec; }
        else if ( strcmp("-RV", argv[i]) == 0 ) { this->jobvr = MagmaVec;   }
        
        // ----- misc
        else if ( strcmp("-x",          argv[i]) == 0 ||
                  strcmp("--exclusive", argv[i]) == 0 ) {
            #ifdef USE_FLOCK
            this->flock_op = LOCK_EX;
            #else
            fprintf( stderr, "ignoring %s: USE_FLOCK not defined; flock not supported.\n", argv[i] );
            #endif
        }
        
        // ----- usage
        else if ( strcmp("-h",     argv[i]) == 0 ||
                  strcmp("--help", argv[i]) == 0 ) {
            fprintf( stderr, usage, argv[0], MAX_NTEST );
            exit(0);
        }
        else {
            fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
            exit(1);
        }
    }
    
    // if -N or --range not given, use default range
    if ( this->ntest == 0 ) {
        int n2 = this->default_nstart;  //1024 + 64;
        while( n2 <= this->default_nend && this->ntest < MAX_NTEST ) {
            this->msize[ this->ntest ] = n2;
            this->nsize[ this->ntest ] = n2;
            this->ksize[ this->ntest ] = n2;
            n2 += this->default_nstep;  //1024;
            this->ntest++;
        }
    }
    assert( this->ntest <= MAX_NTEST );
    
    // fill in msize[:], nsize[:], ksize[:] if -m, -n, -k were given
    if ( m >= 0 ) {
        for( int j = 0; j < this->ntest; ++j ) {
            this->msize[j] = m;
        }
    }
    if ( n >= 0 ) {
        for( int j = 0; j < this->ntest; ++j ) {
            this->nsize[j] = n;
        }
    }
    if ( k >= 0 ) {
        for( int j = 0; j < this->ntest; ++j ) {
            this->ksize[j] = k;
        }
    }
    
    // find max dimensions
    this->mmax = 0;
    this->nmax = 0;
    this->kmax = 0;
    for( int i = 0; i < this->ntest; ++i ) {
        this->mmax = max( this->mmax, this->msize[i] );
        this->nmax = max( this->nmax, this->nsize[i] );
        this->kmax = max( this->kmax, this->ksize[i] );
    }

    // disallow jobu=O, jobvt=O
    if ( this->jobu == MagmaOverwriteVec && this->jobvt == MagmaOverwriteVec ) {
        printf( "jobu and jobvt cannot both be Overwrite.\n" );
        exit(1);
    }
    
    // lock file
    #ifdef USE_FLOCK
    this->flock_fd = open_lockfile( lockfile, this->flock_op );
    #endif

    #ifdef HAVE_CUBLAS
    magma_setdevice( this->device );
    #endif
    
    // create queues on this device
    magma_int_t num;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices( devices, MagmaMaxGPUs, &num );
    
    // 2 queues + 1 extra NULL entry to catch errors
    magma_queue_create( &this->queues2[ 0 ] );
    magma_queue_create( &this->queues2[ 1 ] );
    this->queues2[ 2 ] = NULL;
    
    this->queue = this->queues2[ 0 ];
    
    #ifdef HAVE_CUBLAS
    // handle for directly calling cublas
    cublasCreate( &this->handle );
    cublasSetStream( this->handle, this->queue );
    #endif
}
// end parse_opts


// ------------------------------------------------------------
// Initialize PAPI events set to measure flops.
// Note flops counters are inaccurate on Sandy Bridge, and don't exist on Haswell.
// See http://icl.cs.utk.edu/projects/papi/wiki/PAPITopics:SandyFlops
#ifdef HAVE_PAPI
#include <papi.h>
#include <string.h>  // memset
#endif  // HAVE_PAPI

int gPAPI_flops_set = -1;  /* i.e., PAPI_NULL */

extern "C"
void flops_init()
{
    #ifdef HAVE_PAPI
    int err = PAPI_library_init( PAPI_VER_CURRENT );
    if ( err != PAPI_VER_CURRENT ) {
        fprintf( stderr, "Error: PAPI couldn't initialize: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    // read flops
    err = PAPI_create_eventset( &gPAPI_flops_set );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_create_eventset failed\n" );
    }
    
    err = PAPI_assign_eventset_component( gPAPI_flops_set, 0 );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_assign_eventset_component failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    PAPI_option_t opt;
    memset( &opt, 0, sizeof(PAPI_option_t) );
    opt.inherit.inherit  = PAPI_INHERIT_ALL;
    opt.inherit.eventset = gPAPI_flops_set;
    err = PAPI_set_opt( PAPI_INHERIT, &opt );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_set_opt failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    err = PAPI_add_event( gPAPI_flops_set, PAPI_FP_OPS );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_add_event failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    err = PAPI_start( gPAPI_flops_set );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_start failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    #endif  // HAVE_PAPI
}
