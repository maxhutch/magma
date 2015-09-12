/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @author Hartwig Anzt
       @author Eduardo Ponce

       @generated from zpidr.cpp normal z -> c, Fri Sep 11 18:29:44 2015
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned Induced Dimension 
    Reduction method.


    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in]
    b           magma_c_matrix
                RHS b

    @param[in,out]
    x           magma_c_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_c_solver_par*
                solver parameters

    @param[in]
    precond_par magma_c_preconditioner*
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cposv
    ********************************************************************/

#define MYDEBUG 0
#define WRITEP 0

#if MYDEBUG == 1
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m)
#elif MYDEBUG == 2
#define printD(...) printf(__VA_ARGS__)
#define printMatrix(s,m) magma_cmatrixInfo2(s,m)
#else
#define printD(...)
#define printMatrix(s,m)
#endif


// Notes:
// 2. Overlap kernels using cuBLAS streams
//
// 3. Build dependency graph of IDR(s)-biortho
//
// 4. Apparently, some precision is lost using MAGMA when compared to MATLAB, probably is that matrices are not displayed with full precision on screen.
//
// 5. Optimize: merge kernels, reuse arrays, concurrent kernels


extern "C" void
magma_cmatrixInfo2(
    const char *s,
    magma_c_matrix A )
{
    printf(" %s dims = %d x %d\n", s, int(A.num_rows), int(A.num_cols));
    printf(" %s location = %d = %s\n", s, A.memory_location, (A.memory_location == Magma_CPU) ? "CPU" : "DEV");
    printf(" %s storage = %d = %s\n", s, A.storage_type, (A.storage_type == Magma_CSR) ? "CSR" : "DENSE");
    printf(" %s major = %d = %s\n", s, A.major, (A.major == MagmaRowMajor) ? "row" : "column");
    printf(" %s nnz = %d\n", s, int(A.nnz));
    if (A.memory_location == Magma_DEV)
        magma_cprint_gpu( A.num_rows, A.num_cols, A.dval, A.num_rows );
    else
        magma_cprint( A.num_rows, A.num_cols, A.val, A.num_rows );
}


extern "C" magma_int_t
magma_cpidr(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue = NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_IDR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // local constants
    const magmaFloatComplex c_zero = MAGMA_C_ZERO;
    const magmaFloatComplex c_one = MAGMA_C_ONE;
    const magmaFloatComplex c_n_one = MAGMA_C_NEG_ONE;

    // local variables
    magma_int_t info = 0;
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dofb = b.num_rows * b.num_cols;
    magma_int_t inc = 1;
    magma_int_t s;
    magma_int_t dof;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t *piv = NULL;
    magma_int_t innerflag;
    float residual;
    float nrm;
    float nrmb;
    float nrmr;
    float nrmt;
    float rho;
    float angle;
    magmaFloatComplex om;
    magmaFloatComplex tr;
    magmaFloatComplex alpha;
    magmaFloatComplex beta;
    magmaFloatComplex mkk;
    magmaFloatComplex fk;

    // local matrices and vectors
    magma_c_matrix P1 = {Magma_CSR}, dP1 = {Magma_CSR}, dP = {Magma_CSR};
    magma_c_matrix dr = {Magma_CSR};
    magma_c_matrix dG = {Magma_CSR};
    magma_c_matrix dU = {Magma_CSR};
    magma_c_matrix dM1 = {Magma_CSR}, dM = {Magma_CSR};
    magma_c_matrix df = {Magma_CSR};
    magma_c_matrix dt = {Magma_CSR};
    magma_c_matrix dc = {Magma_CSR};
    magma_c_matrix dv1 = {Magma_CSR}, dv = {Magma_CSR};
    magma_c_matrix dlu = {Magma_CSR};

    // local performance variables
    magma_int_t gpumem = 0;

    // chronometry
    real_Double_t tempo1, tempo2;

    gpumem += (A.nnz * sizeof(magmaFloatComplex)) + (A.nnz * sizeof(magma_index_t)) + ((A.num_rows + 1) * sizeof(magma_index_t));

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        printD("Error! matrix must be square.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }

    // initial s space
    // hack --> use "--restart" option as the shadow space number
    s = 1;
    if ( solver_par->restart != 30 ) {
        if ( solver_par->restart > A.num_cols )
            s = A.num_cols;
        else
            s = solver_par->restart;
    }
    solver_par->restart = s;

    // set max iterations
    solver_par->maxiter = min(2 * A.num_cols, solver_par->maxiter);

    // initial angle
    angle = 0.7;

    // initial solution vector
    // x = 0
    //magmablas_claset( MagmaFull, x->num_rows, x->num_cols, c_zero, c_zero, x->dval, x->num_rows );
    printMatrix("X", *x);
    gpumem += x->nnz * sizeof(magmaFloatComplex);

    // initial RHS
    // b = 1
    //magmablas_claset( MagmaFull, b.num_rows, b.num_cols, c_one, c_one, b.dval, b.num_rows );
    printMatrix("B", b);
    gpumem += b.nnz * sizeof(magmaFloatComplex);

    // |b|
    nrmb = magma_scnrm2( b.num_rows, b.dval, inc );

    // check for |b| == 0
    printD("init norm(b) ..........%f\n", nrmb);
    if ( nrmb == 0.0 ) {
        printD("RHS is zero, exiting...\n");
        magma_cscal( x->num_rows*x->num_cols, MAGMA_C_ZERO, x->dval, inc );
        solver_par->init_res = 0.0;
        solver_par->final_res = 0.0;
        solver_par->iter_res = 0.0;
        solver_par->runtime = 0.0;
        goto cleanup;
    }

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P1 = 0.0
    CHECK( magma_cvinit( &P1, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P1 = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dof = P1.num_rows * P1.num_cols;
    lapackf77_clarnv( &distr, iseed, &dof, P1.val );
    printMatrix("P1", P1);

    // transfer P1 to device
    CHECK( magma_cmtransfer( P1, &dP1, Magma_CPU, Magma_DEV, queue ));
    magma_cmfree( &P1, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_cqr(P1), QR factorization
        CHECK( magma_cqr( dP1.num_rows, dP1.num_cols, dP1, dP1.num_rows, &dP, NULL, queue ) );
    } else {
        // P = P1 / |P1|
        dof = dP1.num_rows * dP1.num_cols;        // can remove
        nrm = magma_scnrm2( dof, dP1.dval, inc );
        nrm = 1.0 / nrm;
        magma_csscal( dof, nrm, dP1.dval, inc );
        CHECK( magma_cmtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
    }
    magma_cmfree(&dP1, queue );
//---------------------------------------
    printMatrix("P", dP);
    gpumem += dP.nnz * sizeof(magmaFloatComplex);

#if WRITEP == 1
    // Note: write P matrix to file to use in MATLAB for validation
    magma_cprint_gpu( dP.num_rows, dP.num_cols, dP.dval, dP.num_rows );
#endif

    // initial residual
    // r = b - A x
    CHECK( magma_cvinit( &dr, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_cresidualvec( A, b, *x, &dr, &nrmr, queue));

    printMatrix("R" , dr);
    gpumem += dr.nnz * sizeof(magmaFloatComplex);

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

    // G(n,s) = 0
    CHECK( magma_cvinit( &dG, Magma_DEV, A.num_cols, s, c_zero, queue ));
    gpumem += dG.nnz * sizeof(magmaFloatComplex);

    // U(n,s) = 0
    CHECK( magma_cvinit( &dU, Magma_DEV, A.num_cols, s, c_zero, queue ));
    gpumem += dU.nnz * sizeof(magmaFloatComplex);

    // M1 = 0
    // M(s,s) = I
    CHECK( magma_cvinit( &dM1, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_cvinit( &dM, Magma_DEV, s, s, c_zero, queue ));
    magmablas_claset( MagmaFull, s, s, c_zero, c_one, dM.dval, s );
    gpumem += 2 * dM.nnz * sizeof(magmaFloatComplex);

    // f = 0
    CHECK( magma_cvinit( &df, Magma_DEV, dP.num_cols, dr.num_cols, c_zero, queue ));
    gpumem += df.nnz * sizeof(magmaFloatComplex);

    // t = 0
    CHECK( magma_cvinit( &dt, Magma_DEV, A.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += dt.nnz * sizeof(magmaFloatComplex);

    // c = 0
    CHECK( magma_cvinit( &dc, Magma_DEV, dM.num_cols, df.num_cols, c_zero, queue ));
    gpumem += dc.nnz * sizeof(magmaFloatComplex);

    // v1 = 0
    // v = 0
    CHECK( magma_cvinit( &dv1, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &dv, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += 2 * dv.nnz * sizeof(magmaFloatComplex);

    // lu = 0 
    CHECK( magma_cvinit( &dlu, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    gpumem += dlu.nnz * sizeof(magmaFloatComplex);

    // piv = 0
    CHECK( magma_imalloc_pinned(&piv, s));

    // om = 1
    om = MAGMA_C_ONE;

    
    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }
    
    innerflag = 0;
    solver_par->numiter = 0;

    // start iteration
    do
    {
        solver_par->numiter++;
    
        // new RHS for small systems
        // f = (r' P)' = P' r
        magmablas_cgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.num_rows, dr.dval, inc, c_zero, df.dval, inc );
        printMatrix("F", df);

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
    
            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
//---------------------------------------
            // c(k:s) = f(k:s)
            magma_ccopy( sk, &df.dval[k], inc, &dc.dval[k], inc );

            // M1 = M
            magma_ccopy( dM.num_rows * dM.num_cols, dM.dval, inc, dM1.dval, inc );

            // c(k:s) = M1(k:s,k:s) \ c(k:s)
            CHECK( magma_cgesv_gpu( sk, dc.num_cols, &dM1.dval[k*dM1.num_rows+k], dM1.num_rows, piv, &dc.dval[k], dc.num_rows, &info ) );
//---------------------------------------
            printMatrix("C", dc);

            // v1 = r - G(:,k:s) c(k:s)
//---------------------------------------
            // v1 = r
            magma_ccopy( dr.num_rows * dr.num_cols, dr.dval, inc, dv1.dval, inc );

            // v1 = v1 - G(:,k:s) c(k:s)
            magmablas_cgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, &dG.dval[k*dG.num_rows], dG.num_rows, &dc.dval[k], inc, c_one, dv1.dval, inc );
//---------------------------------------
            printMatrix("V1", dv1);

            // preconditioning operation 
            // v1 = L \ v1;
            // v1 = U \ v1;
//---------------------------------------
            CHECK( magma_c_applyprecond_left( A, dv1, &dlu, precond_par, queue )); 
            CHECK( magma_c_applyprecond_right( A, dlu, &dv1, precond_par, queue )); 
//---------------------------------------

            // compute new U
            // U(:,k) = om * v1 + U(:,k:s) c(k:s)
//---------------------------------------
            // v1 = om * v1 + U(:,k:s) c(k:s)
            magmablas_cgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.num_rows], dU.num_rows, &dc.dval[k], inc, om, dv1.dval, inc );

            // U(:,k) = v1
            magma_ccopy( dU.num_rows, dv1.dval, inc, &dU.dval[k*dU.num_rows], inc );
//---------------------------------------
            printMatrix("U", dU);

            // compute new U and G
            // G(:,k) = A U(:,k)
//---------------------------------------
            // v = A v1
            CHECK( magma_c_spmv( c_one, A, dv1, c_zero, dv, queue ));

            // G(:,k) = v
            magma_ccopy( dG.num_rows, dv.dval, inc, &dG.dval[k*dG.num_rows], inc );
//---------------------------------------
            printMatrix("G", dG);


            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {
                // alpha = P(:,i)' G(:,k) / M(i,i)
//---------------------------------------
                // alpha = P(:,i)' G(:,k)
                alpha = magma_cdotc( dP.num_rows, &dP.dval[i*dP.num_rows], inc, &dG.dval[k*dG.num_rows], inc );
                
                // alpha = alpha / M(i,i)
                magma_cgetvector( 1, &dM.dval[i*dM.num_rows+i], inc, &mkk, inc );
                alpha = alpha / mkk;
//---------------------------------------
                printD("bi-ortho: i, k, alpha ...................%d, %d, (%f, %f)\n", i, k, MAGMA_C_REAL(alpha), MAGMA_C_IMAG(alpha));

                // G(:,k) = G(:,k) - alpha * G(:,i)
                magma_caxpy( dG.num_rows, -alpha, &dG.dval[i*dG.num_rows], inc, &dG.dval[k*dG.num_rows], inc );
                printMatrix("G", dG);

                // U(:,k) = U(:,k) - alpha * U(:,i)
                magma_caxpy( dU.num_rows, -alpha, &dU.dval[i*dU.num_rows], inc, &dU.dval[k*dU.num_rows], inc );
                printMatrix("U", dU);
            }

            // new column of M = P'G, first k-1 entries are zero
            // Note: need to verify that first k-1 entries are zero

            // M(k:s,k) = (G(:,k)' P(:,k:s))' = P(:,k:s)' G(:,k)
            magmablas_cgemv( MagmaConjTrans, dP.num_rows, sk, c_one, &dP.dval[k*dP.num_rows], dP.num_rows, &dG.dval[k*dG.num_rows], inc, c_zero, &dM.dval[k*dM.num_rows+k], inc );
            printMatrix("M", dM);

            // check M(k,k) == 0
            magma_cgetvector( 1, &dM.dval[k*dM.num_rows+k], inc, &mkk, inc );
            if ( MAGMA_C_EQUAL(mkk, MAGMA_C_ZERO) ) {
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            // beta = f(k) / M(k,k)
            magma_cgetvector( 1, &df.dval[k], inc, &fk, inc );
            beta = fk / mkk;
            printD("beta: k ...................%d, (%f, %f)\n", k, MAGMA_C_REAL(beta), MAGMA_C_IMAG(beta));

            // x = x + beta * U(:,k)
            magma_caxpy( x->num_rows, beta, &dU.dval[k*dU.num_rows], inc, x->dval, inc );
            printMatrix("X", *x);

            // make r orthogonal to q_i, i = 1..k
            // r = r - beta * G(:,k)
            magma_caxpy( dr.num_rows, -beta, &dG.dval[k*dG.num_rows], inc, dr.dval, inc );
            printMatrix("R", dr);

            // |r|
            nrmr = magma_scnrm2( dofb, dr.dval, inc );
            printD("norm(r): k ...................%d, %f\n", k, nrmr);

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
                innerflag = 1;
                break;
            }

            // new f = P' r (first k components are zero)
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                magma_caxpy( sk - 1, -beta, &dM.dval[k*dM.num_rows+(k+1)], inc, &df.dval[k+1], inc );
                printMatrix("F", df);
            }

            // iter = iter + 1
            solver_par->numiter++;
        }

        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag == 1 ) {
            break;
        }

        // v = r
        magma_ccopy( dr.num_rows * dr.num_cols, dr.dval, inc, dv.dval, inc );
        printMatrix("V", dv);

        // preconditioning operation 
        // v = L \ v;
        // v = U \ v;
//---------------------------------------
        CHECK( magma_c_applyprecond_left( A, dv, &dlu, precond_par, queue )); 
        CHECK( magma_c_applyprecond_right( A, dlu, &dv, precond_par, queue )); 
//---------------------------------------

        // t = A v
        CHECK( magma_c_spmv( c_one, A, dv, c_zero, dt, queue ));
        printMatrix("T", dt);

        // computation of a new omega
        // om = omega(t, r, angle);
//---------------------------------------
        // |t|
        dof = dt.num_rows * dt.num_cols;
        nrmt = magma_scnrm2( dof, dt.dval, inc );

        // tr = t' r
        tr = magma_cdotc( dr.num_rows, dt.dval, inc, dr.dval, inc );

        // rho = abs(tr / (|t| * |r|))
        rho = fabs( MAGMA_C_REAL(tr) / (nrmt * nrmr) );

        // om = tr / (|t| * |t|)
        om = tr / (nrmt * nrmt);
        if ( rho < angle )
            om = om * angle / rho;
//---------------------------------------
        printD("omega: k .................... %d, (%f, %f)\n", k, MAGMA_C_REAL(om), MAGMA_C_IMAG(om));
        if ( MAGMA_C_EQUAL(om, MAGMA_C_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // update approximation vector
        // x = x + om * v
        magma_caxpy(x->num_rows, om, dv.dval, inc, x->dval, inc);
        printMatrix("X", *x);

        // update residual vector
        // r = r - om * t
        magma_caxpy(dr.num_rows, -om, dt.dval, inc, dr.dval, inc);
        printMatrix("R", dr);

        // residual norm
        nrmr = magma_scnrm2( dofb, dr.dval, inc );
        printD("norm(r): k ...................%d, %f\n", k, nrmr);

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
            break;
        }

#if MYDEBUG == 2
        // Note: exit loop after a few iterations
        if ( solver_par->numiter + 1 >= (2 * (s + 1)) ) {
            break;
        }
#endif
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
//--------------STOP TIME----------------

    // last stats
    solver_par->iter_res = nrmr;
    CHECK( magma_cresidualvec( A, b, *x, &dr, &residual, NULL ));
    solver_par->final_res = residual;

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

#if MYDEBUG > 0 || WRITEP == 1
    // print local stats
    printf("GPU memory = %f MB\n", (real_Double_t)gpumem / (1<<20));
#endif
    
    
cleanup:
    // free resources
    magma_cmfree(&P1, queue);
    magma_cmfree(&dP1, queue);
    magma_cmfree(&dP, queue);
    magma_cmfree(&dr, queue);
    magma_cmfree(&dG, queue);
    magma_cmfree(&dU, queue);
    magma_cmfree(&dM1, queue);
    magma_cmfree(&dM, queue);
    magma_cmfree(&df, queue);
    magma_cmfree(&dt, queue);
    magma_cmfree(&dc, queue);
    magma_cmfree(&dv1, queue);
    magma_cmfree(&dv, queue);
    magma_cmfree(&dlu, queue);
    magma_free_pinned(piv);

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
    /* magma_cpidr */
}
