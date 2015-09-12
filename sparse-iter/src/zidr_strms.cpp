/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @author Hartwig Anzt
       @author Eduardo Ponce

       @precisions normal z -> s d c
*/

#include "common_magmasparse.h"
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Induced Dimension Reduction method.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in,out]
    x           magma_z_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zposv
    ********************************************************************/

// -1 = no print but write iniital p
//  0 = no printing
//  1 = print scalars
//  2 = print all (few iters)
// >2 = prints all (all iters)
#define MYDEBUG 0 

#if MYDEBUG <= 0
#define printD(...)
#define printMatrix(s,m,q)
#elif MYDEBUG == 1
#define printD(...) printf("%% "__VA_ARGS__)
#define printMatrix(s,m,q)
#elif MYDEBUG == 2
#define printD(...) printf("%% " __VA_ARGS__)
#define printMatrix(s,m,q) magma_zmatrixInfo_strms(s,m,q)
#else
#define printD(...) printf("%% " __VA_ARGS__)
#define printMatrix(s,m,q) magma_zmatrixInfo_strms(s,m,q)
#endif


// Comments:
// 1. queues - when receiving queue in magma_zidr function, one does not knows if queue is stream 0 (synchronous) or stream # (asynchronous). For this reason, I create the necessary queues needed and ignore the queue parameter. In the serial version of magma_zidr I set the queue parameter to the same queue as cuBLAS library. Also, queues will be used in OpenCL so we should use 0 instead of NULL for portability.
// 2. For pointer to array element, use &A[i] or A+i?
// 3. Synchronous functions:
//      magma_zvinit (does not uses queues, calls zmalloc)
//      magma_zmtransfer(calls magma_zsetvector, magma_zcopyvector)
//      magma_zqr_wrapper(calls magma_get_zgeqrf_nb, magma_zgeqrf_gpu, magmablas_zlacpy, magma_zgetvector, magma_zcopyvector, magma_zungqr_gpu)
//

extern "C" void
magma_zmatrixInfo_strms(
    const char *s,
    magma_z_matrix A,
    magma_queue_t queue )
{
    magma_queue_sync( queue );

    printD(" %s dims = %d x %d\n", s, A.num_rows, A.num_cols);
    printD(" %s location = %d = %s\n", s, A.memory_location, (A.memory_location == Magma_CPU) ? "CPU" : "DEV");
    printD(" %s storage = %d = %s\n", s, A.storage_type, (A.storage_type == Magma_CSR) ? "CSR" : "DENSE");
    printD(" %s major = %d = %s\n", s, A.major, (A.major == MagmaRowMajor) ? "row" : "column");
    printD(" %s nnz = %d\n", s, A.nnz);
    
    magma_int_t ldd = magma_roundup( A.num_rows, 32 );
    if (A.ld != ldd) {
        A.ld = A.num_rows;
    }
    if (A.memory_location == Magma_DEV)
        magma_zprint_gpu( A.num_rows, A.num_cols, A.dval, A.ld );
    else
        magma_zprint( A.num_rows, A.num_cols, A.val, A.ld );
}


extern "C" magma_int_t
magma_zidr_strm(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue = NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_IDR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // constants
    const magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magmaDoubleComplex c_n_one = MAGMA_Z_NEG_ONE;

    // internal user options
    const magma_int_t smoothing = 1;   // 1 = enable, 0 = disabled, -1 = disabled with delayed x update
    const double angle = 0.7;          // [0-1]

    // local variables
    magma_int_t info = 0;
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dofx = x->num_rows * x->num_cols;
    magma_int_t dofb = b.num_rows * b.num_cols;
    magma_int_t dofr = A.num_rows * b.num_cols;
    magma_int_t dofM; 
    magma_int_t dofP;
    magma_int_t doft;
    magma_int_t inc = 1;
    magma_int_t s;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t innerflag;
    magma_int_t ldd;
    double residual;
    double nrm;
    double nrmb;
    double nrmr;
    double nrmt;
    double rho;
    magmaDoubleComplex *om = NULL;
    magmaDoubleComplex *gamma = NULL;
    magmaDoubleComplex *mkk = NULL;
    magmaDoubleComplex *fk = NULL;

    // matrices and vectors
    magma_z_matrix dxs = {Magma_CSR};
    magma_z_matrix dP1 = {Magma_CSR}, dP = {Magma_CSR};
    magma_z_matrix dr = {Magma_CSR}, drs = {Magma_CSR};
    magma_z_matrix dg = {Magma_CSR}, dG = {Magma_CSR};
    magma_z_matrix du = {Magma_CSR}, dU = {Magma_CSR};
    magma_z_matrix dM1 = {Magma_CSR}, dM = {Magma_CSR};
    magma_z_matrix df = {Magma_CSR};
    magma_z_matrix dt = {Magma_CSR}, dtt = {Magma_CSR};
    magma_z_matrix dc = {Magma_CSR};
    magma_z_matrix dv1 = {Magma_CSR}, dv = {Magma_CSR};
    magma_z_matrix dskp = {Magma_CSR};
    magma_z_matrix dalpha = {Magma_CSR};
    magma_z_matrix dbeta = {Magma_CSR};
    magmaDoubleComplex *skp = NULL;
    magmaDoubleComplex *alpha = NULL;
    magmaDoubleComplex *beta = NULL;
    magmaDoubleComplex *d1 = NULL, *d2 = NULL;
    
    // queue variables
    const magma_queue_t squeue = 0;    // synchronous kernel queues
    const magma_int_t nqueues = 3;     // number of queues
    magma_queue_t queues[nqueues];    
    magma_int_t q1flag = 0;

    // performance variables
    long long int gpumem = 0;

    // chronometry
    real_Double_t tempo1, tempo2;

    // set asynchronous kernel queues
    printD("Kernel queues: (orig, queue) = (%p, %p)\n", (void *)orig_queue, (void *)queue);
    cudaStreamCreateWithFlags( &(queues[0]), cudaStreamNonBlocking );
    if ( queue != squeue ) {
        queues[1] = queue;
        q1flag = 0;
    } else {
        magma_queue_create( &(queues[1]) );
        queue = queues[1];
        q1flag = 1;
    }
    magma_queue_create( &(queues[2]) );
    for ( i = 0; i < nqueues; ++i ) {
        printD("Kernel queue #%d = %p\n", i, (void *)queues[i]);
    }

    // set to Q1
    magmablasSetKernelStream( queue );

    // initial s space
    // hack --> use "--restart" option as the shadow space number.
    s = 1;
    if ( solver_par->restart != 30 ) {
        if ( solver_par->restart > A.num_cols )
            s = A.num_cols;
        else
            s = solver_par->restart;
    }
    solver_par->restart = s;

    // set max iterations
    solver_par->maxiter = min( 2 * A.num_cols, solver_par->maxiter );

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        printD("Error! matrix must be square.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }
    gpumem += (A.nnz * sizeof(magmaDoubleComplex)) + (A.nnz * sizeof(magma_index_t)) + ((A.num_rows + 1) * sizeof(magma_index_t));

    // initial solution vector
    // x = 0
    //magmablas_zlaset( MagmaFull, x->num_rows, x->num_cols, c_zero, c_zero, x->dval, x->num_rows );
    printMatrix("X", *x, queue);
    gpumem += x->nnz * sizeof(magmaDoubleComplex);

    // initial RHS
    // b = 1
    //magmablas_zlaset( MagmaFull, b.num_rows, b.num_cols, c_one, c_one, b.dval, b.num_rows );
    printMatrix("B", b, queue);
    gpumem += b.nnz * sizeof(magmaDoubleComplex);

    // |b|
    nrmb = magma_dznrm2( b.num_rows, b.dval, inc );
    printD("init norm(b) ..........%lg\n", nrmb);

    // check for |b| == 0
    if ( nrmb == 0.0 ) {
        printD("RHS is zero, exiting...\n");
        magma_zscal( dofx, MAGMA_Z_ZERO, x->dval, 1 );
        solver_par->init_res = 0.0;
        solver_par->final_res = 0.0;
        solver_par->iter_res = 0.0;
        solver_par->runtime = 0.0;
        goto cleanup;
    }

    // t = 0
    // make t twice as large to contain both, dt and dr
    ldd = magma_roundup( b.num_rows, 32 );
    CHECK( magma_zvinit( &dt, Magma_DEV, ldd, 2 * b.num_cols, c_zero, queue ));
    gpumem += dt.nnz * sizeof(magmaDoubleComplex);
    dt.num_rows = b.num_rows;
    dt.num_cols = b.num_cols;
    dt.nnz = dt.num_rows * dt.num_cols;
    doft = dt.ld * dt.num_cols;

    // redirect the dr.dval to the second part of dt
    CHECK( magma_zvinit( &dr, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    magma_free( dr.dval );
    dr.dval = dt.dval + ldd * b.num_cols;

    // r = b - A x
    CHECK( magma_zresidualvec( A, b, *x, &dr, &nrmr, queue ));
    printMatrix("R", dr, queue);
    
    // |r|
    solver_par->init_res = nrmr;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nrmr;
    }

    // check if initial is guess good enough
    if ( nrmr <= solver_par->atol ||
        nrmr/nrmb <= solver_par->rtol ) {
        solver_par->final_res = solver_par->init_res;
        solver_par->iter_res = solver_par->init_res;
        goto cleanup;
    }

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P = 0.0
    CHECK( magma_zvinit( &dP, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dofP = dP.num_rows * dP.num_cols;
    lapackf77_zlarnv( &distr, iseed, &dofP, dP.val );
    printMatrix("P1", dP, queue);

    // transfer P to device
    CHECK( magma_zmtransfer( dP, &dP1, Magma_CPU, Magma_DEV, queue ));
    dP1.major = dP.major;
    magma_zmfree( &dP, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_zqr(P1), QR factorization
        CHECK( magma_zqr( dP1.num_rows, dP1.num_cols, dP1, dP1.ld, &dP, NULL, queue ));
    } else {
        // P = P1 / |P1|
        nrm = magma_dznrm2( dofP, dP1.dval, inc );
        nrm = 1.0 / nrm;
        magma_zdscal( dofP, nrm, dP1.dval, 1 );
        CHECK( magma_zmtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
        dP.major = dP1.major;
    }
    magma_zmfree(&dP1, queue );
//---------------------------------------
    printMatrix("P", dP, queue);
    gpumem += dP.nnz * sizeof(magmaDoubleComplex);

#if MYDEBUG == -1
    // Note: write P matrix to file to use in MATLAB for validation
    printf("P = ");
    magma_zprint_gpu( dP.num_rows, dP.num_cols, dP.dval, dP.ld );
#endif

    // allocate memory for the scalar products
    CHECK( magma_zmalloc_pinned( &skp, 4 ));
    CHECK( magma_zvinit( &dskp, Magma_DEV, 4, 1, c_zero, queue ));
    gpumem += dskp.nnz * sizeof(magmaDoubleComplex);

    CHECK( magma_zmalloc_pinned( &alpha, s ));
    CHECK( magma_zvinit( &dalpha, Magma_DEV, s, 1, c_zero, queue ));
    gpumem += dalpha.nnz * sizeof(magmaDoubleComplex);

    CHECK( magma_zmalloc_pinned( &beta, s ));
    CHECK( magma_zvinit( &dbeta, Magma_DEV, s, 1, c_zero, queue ));
    gpumem += dbeta.nnz * sizeof(magmaDoubleComplex);
    
    // workspace for merged dot product
    CHECK( magma_zmalloc( &d1, 2 * dofb ));
    CHECK( magma_zmalloc( &d2, 2 * dofb ));
    gpumem += 4 * dofb * sizeof(magmaDoubleComplex);

    if ( smoothing > 0 ) {
        // set smoothing solution vector
        CHECK( magma_zmtransfer( *x, &dxs, Magma_DEV, Magma_DEV, queue ));
        dxs.major = x->major;
        gpumem += dxs.nnz * sizeof(magmaDoubleComplex);

        // tt = 0
        // make tt twice as large to contain both, dtt and drs
        ldd = magma_roundup( b.num_rows, 32 );
        CHECK( magma_zvinit( &dtt, Magma_DEV, ldd, 2 * b.num_cols, c_zero, queue ));
        gpumem += dtt.nnz * sizeof(magmaDoubleComplex);
        dtt.num_rows = b.num_rows;
        dtt.num_cols = b.num_cols;
        dtt.nnz = dtt.num_rows * dtt.num_cols;

        // redirect the drs.dval to the second part of dtt
        CHECK( magma_zvinit( &drs, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
        magma_free( drs.dval );
        drs.dval = dtt.dval + ldd * b.num_cols;

        // set smoothing residual vector
        magma_zcopy( dofr, dr.dval, 1, drs.dval, 1 );
    }

    // G(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_rows, 32 );
        CHECK( magma_zvinit( &dG, Magma_DEV, ldd, s, c_zero, queue ));
        dG.num_rows = A.num_rows;
    } else {
        CHECK( magma_zvinit( &dG, Magma_DEV, A.num_rows, s, c_zero, queue ));
    }
    gpumem += dG.nnz * sizeof(magmaDoubleComplex);

    // dg represents a single column of dG, array pointer is set inside loop
    CHECK( magma_zvinit( &dg, Magma_DEV, dG.num_rows, 1, c_zero, queue ));
    magma_free( dg.dval );

    // U(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_cols, 32 );
        CHECK( magma_zvinit( &dU, Magma_DEV, ldd, s, c_zero, queue ));
        dU.num_rows = A.num_cols;
    } else {
        CHECK( magma_zvinit( &dU, Magma_DEV, A.num_cols, s, c_zero, queue ));
    }
    gpumem += dU.nnz * sizeof(magmaDoubleComplex);

    // du represents a single column of dU, array pointer is set inside loop
    CHECK( magma_zvinit( &du, Magma_DEV, dU.num_rows, 1, c_zero, queue ));
    magma_free( du.dval );

    // M1 = 0
    // M(s,s) = I
    CHECK( magma_zvinit( &dM1, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_zvinit( &dM, Magma_DEV, s, s, c_zero, queue ));
    dofM = dM.num_rows * dM.num_cols;
    magmablas_zlaset( MagmaFull, dM.num_rows, dM.num_cols, c_zero, c_one, dM.dval, dM.ld );
    magma_zcopy( dofM, dM.dval, 1, dM1.dval, 1 );
    gpumem += (dM1.nnz + dM.nnz) * sizeof(magmaDoubleComplex);

    // M(k,k)
    CHECK( magma_zmalloc_pinned( &mkk, s ));

    // fk
    CHECK( magma_zmalloc_pinned( &fk, 1 ));

    // gamma 
    CHECK( magma_zmalloc_pinned( &gamma, 1 ));

    // om 
    CHECK( magma_zmalloc_pinned( &om, 1 ));

    // f = 0
    CHECK( magma_zvinit( &df, Magma_DEV, dP.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += df.nnz * sizeof(magmaDoubleComplex);

    // c = 0
    CHECK( magma_zvinit( &dc, Magma_DEV, dM.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += dc.nnz * sizeof(magmaDoubleComplex);

    // v = 0
    CHECK( magma_zvinit( &dv, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    magma_zcopy( dofr, dr.dval, 1, dv.dval, 1 );
    gpumem += dv.nnz * sizeof(magmaDoubleComplex);

    // v1 = 0
    CHECK( magma_zvinit( &dv1, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += dv1.nnz * sizeof(magmaDoubleComplex);

    // print local stats
    printD("GPU memory = %f MB\n", (real_Double_t)gpumem / (1<<20));
    
    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }
    
cudaProfilerStart();

    om[0] = MAGMA_Z_ONE;
    innerflag = 0;
    solver_par->numiter = 0;

    // new RHS for small systems
    // f = P' r
    // Q1
    magmablas_zgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.ld, dr.dval, 1, c_zero, df.dval, 1 );
    printMatrix("F", df, queues[1]);

    // c = f
    // Q1
    magma_zcopy( s, df.dval, 1, dc.dval, 1 );

    // solve small system and make v orthogonal to P
    // f(k:s) = M(k:s,k:s) c(k:s)
    // c(k:s) = M1(k:s,k:s) c(k:s)
    // Q1
    magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, s, dM1.dval, dM1.ld, dc.dval, inc );
    printMatrix("C", dc, queues[1]);

    // start iteration
    do
    {
        solver_par->numiter++;
    
        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
            dg.dval = dG.dval + k * dG.ld;
            du.dval = dU.dval + k * dU.ld;
    
            // v = r - G(:,k:s) c(k:s)
            // v = v - G(:,k:s) c(k:s)
            // Q1
            magmablas_zgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, dg.dval, dG.ld, &dc.dval[k], 1, c_one, dv.dval, 1 );
            printMatrix("V", dv, queues[1]);

            // compute new U
            // U(:,k) = om * v + U(:,k:s) c(k:s)
            // v = om * v + U(:,k:s) c(k:s)
            // Q1
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, sk, c_one, du.dval, dU.ld, &dc.dval[k], 1, om[0], dv.dval, 1 );
            printMatrix("U", dv, queues[1]);

            // G(:,k) = A U(:,k)
            // G(:,k) = A v
            // Q1
            CHECK( magma_z_spmv( c_one, A, dv, c_zero, dg, queues[1] ));
            printMatrix("G", dG, queues[1]);

            if ( k == 0 ) {
                // fk = f(k)
                // Q
                magma_zgetvector( 1, &df.dval[k], 1, fk, 1 );
// implicit sync Q1, Q2

                // new column of M = P'G, first k-1 entries are zero
                // M(k:s,k) = P(:,k:s)' G(:,k)
                // Q1
                magmablas_zgemv( MagmaConjTrans, dP.num_rows, s, c_one, dP.dval, dP.ld, dg.dval, 1, c_zero, dM.dval, 1 );
                printMatrix("M", dM, queues[1]);
            }
            else {
                if ( smoothing >= 0 ) {
                    // x = x + beta * U(:,k)
                    // Q0
                    magmablasSetKernelStream( queues[0] );
                    magma_zaxpy( x->num_rows, beta[k-1], &dU.dval[(k-1)*dU.ld], 1, x->dval, 1 );
                    printMatrix("X", *x, queues[0]);
                    magmablasSetKernelStream( queues[1] );
                }

                if ( smoothing > 0 ) {
                    // xs = xs - gamma * (xs - x) 
                    // Q0
                    magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma[0], x->dval, dxs.dval, queues[0] );
                    printMatrix("XS", dxs, queues[0]);

/*
                    // update f
                    magmablasSetKernelStream( queues[2] );

                    // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                    // Q2
                    magma_zaxpy( sk, -beta[k-1], &dM.dval[(k-1)*dM.ld+k], 1, &df.dval[k], 1 );
                    printMatrix("F", df, queues[2]);

                    // c(k:s) = f(k:s)
                    // Q2
                    magma_zcopy( sk + 1, &df.dval[k-1], 1, &dc.dval[k-1], 1 );

                    // solve small system and make v orthogonal to P
                    // f(k:s) = M(k:s,k:s) c(k:s)
                    // c(k:s) = M1(k:s,k:s) c(k:s)
                    // Q2
                    magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk, &dM1.dval[k*dM1.ld+k], dM1.ld, &dc.dval[k], inc );
                    printMatrix("C", dc, queues[2]);

                    magmablasSetKernelStream( queues[1] );
*/
                }
            
                // bi-orthogonalize the new basis vectors
                // comment: merge zaxpy with next dot product into single kernel
                for ( i = 0; i < k; ++i ) {
                    // alpha = P(:,i)' G(:,k) / M(i,i)
                    // Q1
                    alpha[i] = magma_zdotc( dP.num_rows, &dP.dval[i*dP.ld], 1, dg.dval, 1 ); 
// implicit sync Q1, Q2
                    alpha[i] = alpha[i] / mkk[i];
                    printD("bi-ortho: i, k, alpha ...................%d, %d, (%lg, %lg)\n", i, k, MAGMA_Z_REAL(alpha[i]), MAGMA_Z_IMAG(alpha[i]));
                    
                    // G(:,k) = G(:,k) - alpha * G(:,i)
                    // Q1
                    magma_zaxpy( dG.num_rows, -alpha[i], &dG.dval[i*dG.ld], 1, dg.dval, 1 );
                    printMatrix("G", dG, queues[1]);
                }

                // new column of M = P'G, first k-1 entries are zero
                // M(k:s,k) = P(:,k:s)' G(:,k)
                // Q1
                magmablas_zgemv( MagmaConjTrans, dP.num_rows, sk, c_one, &dP.dval[k*dP.ld], dP.ld, dg.dval, 1, c_zero, &dM.dval[k*dM.ld+k], 1 );
                printMatrix("M", dM, queues[1]);

                // fk = f(k)
                // Q2
                magma_zgetvector_async( 1, &df.dval[k], 1, fk, 1, queues[2] );

                // alpha = dalpha
                // Q0
                magma_zsetvector_async( k, alpha, 1, dalpha.dval, 1, queues[0] );

                // outside the loop using GEMV
                // U(:,k) = U(:,k) - U(:,0:k-1) alpha(0:k-1)
                // v = v - U(:,0:k-1) alpha(0:k-1)
                // Q0
                magmablasSetKernelStream( queues[0] );
                magmablas_zgemv( MagmaNoTrans, dU.num_rows, k, c_n_one, dU.dval, dU.ld, dalpha.dval, 1, c_one, dv.dval, 1 );
                printMatrix("U", dU, queues[0]);
                magmablasSetKernelStream( queues[1] );
            }

            // U(:,k) = v
            // Q0
            magma_zcopyvector_async( dU.num_rows, dv.dval, 1, du.dval, 1, queues[0]);

            // mkk = M(k,k)
            // Q
            magma_zgetvector( 1, &dM.dval[k*dM.ld+k], 1, &mkk[k], 1 );
// implicit sync Q1, Q2

            // check M(k,k) == 0
            if ( MAGMA_Z_EQUAL(mkk[k], MAGMA_Z_ZERO) ) {
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            // beta = f(k) / M(k,k)
            beta[k] = fk[0] / mkk[k];
            printD("beta: k ...................%d, (%lg, %lg)\n", k, MAGMA_Z_REAL(beta[k]), MAGMA_Z_IMAG(beta[k]));

            if ( smoothing < 0 ) {
                // make r orthogonal to q_i, i = 1..k
                // r = r - beta * G(:,k)
                // Q1
                magma_zaxpy( dr.num_rows, -beta[k], dg.dval, 1, dr.dval, 1 );
                printMatrix("R", dr, queues[1]);
            
                // M1 = M
                // Q0
                //magma_zcopyvector_async( dofM, dM.dval, 1, dM1.dval, 1, queues[0] );
                magma_zcopyvector_async( sk, &dM.dval[k*dM.ld+k], 1, &dM1.dval[k*dM1.ld+k], 1, queues[0] );

                // update f
                if ( (k + 1) < s ) {
                    magmablasSetKernelStream( queues[0] );

                    // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                    // Q0
                    magma_zaxpy( sk - 1, -beta[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1 );
                    printMatrix("F", df, queues[0]);

                    // c(k:s) = f(k:s)
                    // Q0
                    magma_zcopy( sk, &df.dval[k], 1, &dc.dval[k], 1 );

                    // solve small system and make v orthogonal to P
                    // f(k:s) = M(k:s,k:s) c(k:s)
                    // c(k:s) = M1(k:s,k:s) c(k:s)
                    // Q0
                    magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk - 1, &dM1.dval[(k+1)*dM1.ld+(k+1)], dM1.ld, &dc.dval[k+1], inc );
                    printMatrix("C", dc, queues[0]);

                    magmablasSetKernelStream( queues[1] );
                }

                // |r|
                // Q1
                nrmr = magma_dznrm2( dofr, dr.dval, inc );           
// implicit sync Q1, Q2
                printD("norm(r): k ...................%d, %lg\n", k, nrmr);

/*            
                // v = r
                // Q1
                magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
*/

                if ( (k + 1) == s ) {
                    // v1 = r
                    // Q0
                    magma_zcopyvector_async( dofr, dr.dval, 1, dv1.dval, 1, queues[0] );
                }
                else {
                    // v = r
                    // Q1
                    magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
                }
            }
            else if ( smoothing == 0 ) {
                // make r orthogonal to q_i, i = 1..k
                // r = r - beta * G(:,k)
                // Q1
                magma_zaxpy( dr.num_rows, -beta[k], dg.dval, 1, dr.dval, 1 );
                printMatrix("R", dr, queues[1]);
            
                // M1 = M
                // Q0
                magma_zcopyvector_async( dofM, dM.dval, 1, dM1.dval, 1, queues[0] );
                //magma_zcopyvector_async( sk, &dM.dval[k*dM.ld+k], 1, &dM1.dval[k*dM1.ld+k], 1, queues[0] );

                if ( s == 1 || (k + 1) == s ) {
                    // x = x + beta * U(:,k)
                    // Q0
                    magmablasSetKernelStream( queues[0] );
                    magma_zaxpy( x->num_rows, beta[k], du.dval, 1, x->dval, 1 );
                    printMatrix("X", *x, queues[0]);
                    magmablasSetKernelStream( queues[1] );
                }

                // update f
                if ( (k + 1) < s ) {
                    magmablasSetKernelStream( queues[0] );

                    // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                    // Q0
                    magma_zaxpy( sk - 1, -beta[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1 );
                    printMatrix("F", df, queues[0]);

                    // c(k:s) = f(k:s)
                    // Q0
                    magma_zcopy( sk, &df.dval[k], 1, &dc.dval[k], 1 );

                    // solve small system and make v orthogonal to P
                    // f(k:s) = M(k:s,k:s) c(k:s)
                    // c(k:s) = M1(k:s,k:s) c(k:s)
                    // Q0
                    magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk - 1, &dM1.dval[(k+1)*dM1.ld+(k+1)], dM1.ld, &dc.dval[k+1], inc );
                    printMatrix("C", dc, queues[0]);

                    magmablasSetKernelStream( queues[1] );
                }

                // |r|
                // Q1
                nrmr = magma_dznrm2( dofr, dr.dval, inc );           
// implicit sync Q1, Q2
                printD("norm(r): k ...................%d, %lg\n", k, nrmr);

/*            
                // v = r
                // Q1
                magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
*/

                if ( (k + 1) == s ) {
                    // v1 = r
                    // Q0
                    magma_zcopyvector_async( dofr, dr.dval, 1, dv1.dval, 1, queues[0] );
                }
                else {
                    // v = r
                    // Q1
                    magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
                }
            }
            else {
                // make r orthogonal to q_i, i = 1..k
                // r = r - beta * G(:,k)
                // Q1
                magma_zaxpy( dr.num_rows, -beta[k], dg.dval, 1, dr.dval, 1 );
                printMatrix("R", dr, queues[1]);
            
                // M1 = M
                // M1(k:s,k) = M(k:s,k)
                // Q0
                magma_zcopyvector_async( dofM, dM.dval, 1, dM1.dval, 1, queues[0] );
                //magma_zcopyvector_async( sk, &dM.dval[k*dM.ld+k], 1, &dM1.dval[k*dM1.ld+k], 1, queues[0] );
                
                // smoothing operation
                // t = rs - r
                // Q1
                magma_zidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queues[1] );

                // t't 
                // t'rs
                // Q1
                CHECK( magma_zmdotc( doft, 2, dtt.dval, dtt.dval, d1, d2, &dskp.dval[2], queues[1] ));

                // skp = dskp
                // Q
                magma_zgetvector( 2 , &dskp.dval[2], 1, &skp[2], 1 );
// implicit sync Q1, Q2

                // gamma = t'rs / t't
                // comment: need to check this approach
                gamma[0] = skp[3] / skp[2];
                
                // rs = rs - gamma * t
                // Q1
                magma_zaxpy( drs.num_rows, -gamma[0], dtt.dval, inc, drs.dval, inc );
                printMatrix("RS", drs, queues[1]);

                // |rs|
                // Q1
                nrmr = magma_dznrm2( dofr, drs.dval, inc );           
// implicit sync Q1, Q2
                printD("norm(rs): k ...................%d, %lg\n", k, nrmr);


                if ( s == 1 || (k + 1) == s ) {
                    // x = x + beta * U(:,k)
                    // Q0
                    magmablasSetKernelStream( queues[0] );
                    magma_zaxpy( x->num_rows, beta[k], du.dval, 1, x->dval, 1 );
                    printMatrix("X", *x, queues[0]);
                    magmablasSetKernelStream( queues[1] );

                    // xs = xs - gamma * (xs - x) 
                    // Q0
                    magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma[0], x->dval, dxs.dval, queues[0] );
                    printMatrix("XS", dxs, queues[0]);
                }

                // update f
                if ( (k + 1) < s ) {
                    magmablasSetKernelStream( queues[0] );

                    // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                    // Q0
                    magma_zaxpy( sk - 1, -beta[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1 );
                    printMatrix("F", df, queues[0]);

                    // c(k:s) = f(k:s)
                    // Q0
                    magma_zcopy( sk, &df.dval[k], 1, &dc.dval[k], 1 );

                    // solve small system and make v orthogonal to P
                    // f(k:s) = M(k:s,k:s) c(k:s)
                    // c(k:s) = M1(k:s,k:s) c(k:s)
                    // Q0
                    magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk - 1, &dM1.dval[(k+1)*dM1.ld+(k+1)], dM1.ld, &dc.dval[k+1], inc );
                    printMatrix("C", dc, queues[0]);

                    magmablasSetKernelStream( queues[1] );
                }

/*
                // v = r
                // Q1
                magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
*/

              if ( (k + 1) == s ) {
                // v1 = r
                // Q0
                magma_zcopyvector_async( dofr, dr.dval, 1, dv1.dval, 1, queues[0] );
              }
              else {
                // v = r
                // Q1
                magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
              }
            }

            // store current timing and residual
            if ( solver_par->verbose > 0 ) {
                tempo2 = magma_sync_wtime( queue );
                if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                    solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) nrmr;
                    solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) tempo2-tempo1;
                }
            }

            // check convergence or iteration limit
            if ( nrmr <= solver_par->atol ||
                nrmr/nrmb <= solver_par->rtol || 
                solver_par->numiter >= solver_par->maxiter ) {
                s = k; // for the x-update outside the loop
                innerflag = 2;
                break;
            }

            if ( (k + 1) == s ) {
                // t = A v
                // t = A r
                // Q2
                CHECK( magma_z_spmv( c_one, A, dr, c_zero, dt, queues[2] ));
                printMatrix("T", dt, queues[2]);

                // t't
                // t'r
                // Q2
                CHECK( magma_zmdotc( doft, 2, dt.dval, dt.dval, d1, d2, dskp.dval, queues[2] ));
            }

if ( smoothing > 0 && (k + 1) < s ) {
// sync Q0
// c(k:s) = M1(k:s,k:s) c(k:s)
magma_queue_sync( queues[0] );
}

            // iter = iter + 1
            solver_par->numiter++;
        }

        // update solution approximation x
        if ( smoothing < 0 && innerflag != 1 ) {
            // dbeta = beta
            // Q1
            magma_zsetvector_async( s, beta, 1, dbeta.dval, 1, queues[1] );

            // x = x + beta * U(:,1:s)
            // Q1
            magmablas_zgemv( MagmaNoTrans, dU.num_rows, s, c_one, dU.dval, dU.ld, dbeta.dval, 1, c_one, x->dval, 1 );
            printMatrix("X", *x, queues[1]);
        }

        if ( smoothing >= 0 && innerflag == 2 ) { 
            // x = x + beta * U(:,k)
            // Q0
            magmablasSetKernelStream( queues[0] );
            magma_zaxpy( x->num_rows, beta[k], du.dval, 1, x->dval, 1 );
            printMatrix("X", *x, queues[0]);
            magmablasSetKernelStream( queues[1] );

            if ( smoothing > 0 ) {
                // xs = xs - gamma * (xs - x) 
                // Q0
                magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma[0], x->dval, dxs.dval, queues[0] );
                printMatrix("XS", dxs, queues[0]);
            }
        }

        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag > 0 ) {
            printD("IDR exited from inner loop.\n");
            break;
        }

        // computation of a new omega
//---------------------------------------
        // skp = dskp
        // Q
        magma_zgetvector( 2 , dskp.dval, 1, skp, 1 );
// implicit sync Q1, Q2

        // |t|
        nrmt = magma_dsqrt(MAGMA_Z_REAL(skp[0]));
        printD("tr, norm(t): k .................... %d, (%lg, %lg), %lg\n", k, MAGMA_Z_REAL(skp[1]), MAGMA_Z_IMAG(skp[1]), nrmt);
        
        // rho = abs(t'r / (|t| * |r|))
        rho = fabs( MAGMA_Z_REAL(skp[1]) / (nrmt * nrmr) );

        // om = t'r / (|t| * |t|)
        om[0] = skp[1] / MAGMA_Z_REAL(skp[0]); 
        if ( rho < angle )
            om[0] = om[0] * (angle / rho);
//---------------------------------------
        printD("omega: k .................... %d, (%lg, %lg)\n", k, MAGMA_Z_REAL(om[0]), MAGMA_Z_IMAG(om[0]));
        if ( MAGMA_Z_EQUAL(om[0], MAGMA_Z_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            printD("IDR exited from outer loop, divergence.\n");
            break;
        }

        // update residual vector
        // r = r - om * t
        // Q1
        magma_zaxpy(dr.num_rows, -om[0], dt.dval, 1, dr.dval, 1);
        printMatrix("R", dr, queues[1]);

        // update approximation vector
        // x = x + om * v
        // x = x + om * v1
        // Q0
        magmablasSetKernelStream( queues[0] );
        magma_zaxpy(x->num_rows, om[0], dv1.dval, 1, x->dval, 1);
        printMatrix("X", *x, queues[0]);
        magmablasSetKernelStream( queues[1] );

        if ( smoothing < 1 ) {
            // |r|
            // Q1
            nrmr = magma_dznrm2( dofr, dr.dval, inc );           
// implicit sync Q1, Q2
            printD("norm(r): k ...................%d, %lg\n", k, nrmr);

        // smoothing operation
        } else {
            // t = rs - r
            // Q1
            magma_zidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queues[1] );

            // t't
            // t'rs
            // Q1
            CHECK( magma_zmdotc( doft, 2, dtt.dval, dtt.dval, d1, d2, &dskp.dval[2], queues[1] ));

            // skp = dskp
            // Q
            magma_zgetvector( 2 , &dskp.dval[2], 1, &skp[2], 1 );
// implicit sync Q1, Q2

    magmablasSetKernelStream( queues[2] );

    // new RHS for small systems
    // f = P' r
    // Q2
    magmablas_zgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.ld, dr.dval, 1, c_zero, df.dval, 1 );
    printMatrix("F", df, queues[2]);
    // c = f
    // Q2
    magma_zcopy( s, df.dval, 1, dc.dval, 1 );
    // solve small system and make v orthogonal to P
    // f(k:s) = M(k:s,k:s) c(k:s)
    // c(k:s) = M1(k:s,k:s) c(k:s)
    // Q2
    magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, s, dM1.dval, dM1.ld, dc.dval, inc );
    printMatrix("C", dc, queues[2]);

    magmablasSetKernelStream( queues[1] );

            // gamma = t'rs / t't
            // comment: need to check this approach
            gamma[0] = skp[3] / skp[2];

            // rs = rs - gamma * t
            // Q1
            magma_zaxpy( drs.num_rows, -gamma[0], dtt.dval, inc, drs.dval, inc );
            printMatrix("RS", drs, queues[1]);

            // xs = xs - gamma * (xs - x) 
            // Q0
            magma_zidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma[0], x->dval, dxs.dval, queues[0] );
            printMatrix("XS", dxs, queues[0]);
        
            // |rs|
            // Q1
            nrmr = magma_dznrm2( dofr, drs.dval, inc );           
// implicit sync Q1, Q2
            printD("norm(rs): k ...................%d, %lg\n", k, nrmr);
        }

        // store current timing and residual
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) nrmr;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        // check convergence or iteration limit
        if ( nrmr <= solver_par->atol ||
            nrmr/nrmb <= solver_par->rtol || 
            solver_par->numiter >= solver_par->maxiter ) {
            printD("IDR exited from outer loop, convergence.\n");
            break;
        }

        if ( smoothing < 1 ) {
            // v = r
            // Q0
            magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[0] );

            // new RHS for small systems
            // f = P' r
            // Q1
            magmablas_zgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.ld, dr.dval, 1, c_zero, df.dval, 1 );
            printMatrix("F", df, queues[1]);

            // c = f
            // Q1
            magma_zcopy( s, df.dval, 1, dc.dval, 1 );

            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
            // c(k:s) = M1(k:s,k:s) c(k:s)
            // Q1
            magma_ztrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, s, dM1.dval, dM1.ld, dc.dval, inc );
            printMatrix("C", dc, queues[1]);

// sync Q0
// v = r
            magma_queue_sync( queues[0] );
        } else {
           // v = r
           // Q1
           magma_zcopyvector_async( dofr, dr.dval, 1, dv.dval, 1, queues[1] );
        }

#if MYDEBUG == 2
        // Note: exit loop after a few iterations
        if ( solver_par->numiter + 1 >= (2 * (s + 1)) ) {
            break;
        }
#endif
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );
            
    // sync all queues 
    for ( i = 0; i < nqueues; ++i ) {
        magma_queue_sync( queues[i] );
    }

    // set to Q
    magmablasSetKernelStream( queue );

    // comment: perform this copies concurrently 
    if ( smoothing > 0 ) {
        magma_zcopy( dofr, drs.dval, 1, dr.dval, 1 );
        magma_zcopy( dofx, dxs.dval, 1, x->dval, 1 );
    }

cudaProfilerStop();

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
//--------------STOP TIME----------------

    // last stats
    printMatrix("B_last", b, queue);
    printMatrix("X_last", *x, queue);
    printMatrix("R_last", dr, queue);
    printD("last norm(r): ................. %lg\n", nrmr);
    solver_par->iter_res = nrmr;
    CHECK( magma_zresidualvec( A, b, *x, &dr, &residual, queue ));
    solver_par->final_res = residual;
    printD("last residual: ................. %lg\n", residual);

    // set solver conclusion
    if ( info != MAGMA_SUCCESS ) {
        if ( solver_par->init_res > solver_par->final_res ) {
            info = MAGMA_SLOW_CONVERGENCE;
        }
        else {
            info = MAGMA_DIVERGENCE;
        }
    }
//---------------------------------------

cleanup:

    // sync all queues 
    for ( i = 0; i < nqueues; ++i ) {
        magma_queue_sync( queues[i] );
    }

    // set to Q
    magmablasSetKernelStream( queue );

    // free resources
    dr.dval = NULL;   // needed because its pointer is redirected to dt
    drs.dval = NULL;  // needed because its pointer is redirected to dtt
    magma_zmfree( &dP1, queue );
    magma_zmfree( &dP, queue );
    magma_zmfree( &dr, queue );
    magma_zmfree( &dG, queue );
    magma_zmfree( &dU, queue );
    magma_zmfree( &dM1, queue );
    magma_zmfree( &dM, queue );
    magma_zmfree( &df, queue );
    magma_zmfree( &dt, queue );
    magma_zmfree( &dc, queue );
    magma_zmfree( &dv, queue );
    magma_zmfree( &dv1, queue );
    if ( smoothing > 0 ) {
        magma_zmfree( &dxs, queue );
        magma_zmfree( &dtt, queue );
        magma_zmfree( &drs, queue ); 
    }
    magma_zmfree( &dskp, queue );
    magma_zmfree( &dalpha, queue );
    magma_zmfree( &dbeta, queue );
    magma_free_pinned( skp );
    magma_free_pinned( alpha );
    magma_free_pinned( beta );
    magma_free_pinned( mkk );
    magma_free_pinned( fk );
    magma_free_pinned( gamma );
    magma_free_pinned( om );
    magma_free( d1 );
    magma_free( d2 );

    // destroy asynchronous queues
    magma_queue_destroy( queues[0] );
    if ( q1flag == 1 ) {
        magma_queue_destroy( queues[1] );
    }
    magma_queue_destroy( queues[2] );

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
    /* magma_zidr_strms */
}
