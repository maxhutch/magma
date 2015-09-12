/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @author Hartwig Anzt
       @author Eduardo Ponce

       @generated from zidr.cpp normal z -> d, Fri Sep 11 18:29:44 2015
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Induced Dimension Reduction method.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in]
    b           magma_d_matrix
                RHS b

    @param[in,out]
    x           magma_d_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_d_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dposv
    ********************************************************************/


extern "C" magma_int_t
magma_didr(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    // set queue for old dense routines
    magma_queue_t orig_queue = NULL;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_IDR;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // constants
    const double c_zero = MAGMA_D_ZERO;
    const double c_one = MAGMA_D_ONE;
    const double c_n_one = MAGMA_D_NEG_ONE;

    // internal user parameters
    const magma_int_t smoothing = 0;   // 1 = enable, 0 = disable
    const double angle = 0.7;          // [0-1]

    // local variables
    magma_int_t iseed[4] = { 0, 0, 0, 1 };
    magma_int_t dof;
    magma_int_t s;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t *piv = NULL;
    magma_int_t innerflag;
    double residual;
    double nrm;
    double nrmb;
    double nrmr;
    double nrmt;
    double rho;
    double om;
    double tr;
    double gamma;
    double alpha;
    double mkk;
    double fk;

    // matrices and vectors
    magma_d_matrix P1 = {Magma_CSR}, dP1 = {Magma_CSR}, dP = {Magma_CSR};
    magma_d_matrix dr = {Magma_CSR};
    magma_d_matrix dG = {Magma_CSR};
    magma_d_matrix dU = {Magma_CSR};
    magma_d_matrix dM1 = {Magma_CSR}, dM = {Magma_CSR};
    magma_d_matrix df = {Magma_CSR};
    magma_d_matrix dt = {Magma_CSR};
    magma_d_matrix dc = {Magma_CSR};
    magma_d_matrix dv1 = {Magma_CSR}, dv = {Magma_CSR};
    magma_d_matrix dxs = {Magma_CSR};
    magma_d_matrix drs = {Magma_CSR};
    magma_d_matrix dbeta = {Magma_CSR}, beta = {Magma_CSR};

    // chronometry
    real_Double_t tempo1, tempo2;
   
    // initial s space
    // hack --> use "--restart" option as the shadow space number
    s = 1;
    if ( solver_par->restart != 30 ) {
        if ( solver_par->restart > A.num_cols ) {
            s = A.num_cols;
        }
        else {
            s = solver_par->restart;
        }
    }
    solver_par->restart = s;

    // set max iterations
    solver_par->maxiter = min( 2 * A.num_cols, solver_par->maxiter );

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        printf("Operator A is not square.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    // |b|
    nrmb = magma_dnrm2( b.num_rows, b.dval, 1 );

    // check for |b| == 0
    if ( nrmb == 0.0 ) {
        magma_dscal( x->num_rows * x->num_cols, MAGMA_D_ZERO, x->dval, 1 );
        solver_par->init_res = 0.0;
        solver_par->final_res = 0.0;
        solver_par->iter_res = 0.0;
        solver_par->runtime = 0.0;
        goto cleanup;
    }

    // r = b - A x
    CHECK( magma_dvinit( &dr, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dresidualvec( A, b, *x, &dr, &nrmr, queue ));
    
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
    // P1 = 0.0
    CHECK( magma_dvinit( &P1, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P1 = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dof = P1.num_rows * P1.num_cols;
    lapackf77_dlarnv( &distr, iseed, &dof, P1.val );

    // transfer P1 to device
    CHECK( magma_dmtransfer( P1, &dP1, Magma_CPU, Magma_DEV, queue ));
    magma_dmfree( &P1, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_dqr(P1), QR factorization
        CHECK( magma_dqr( dP1.num_rows, dP1.num_cols, dP1, dP1.ld, &dP, NULL, queue ));
    } else {
        // P = P1 / |P1|
        nrm = magma_dnrm2( dP1.num_rows * dP1.num_cols, dP1.dval, 1 );
        nrm = 1.0 / nrm;
        magma_dscal( dP1.num_rows * dP1.num_cols, nrm, dP1.dval, 1 );
        CHECK( magma_dmtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
    }
    magma_dmfree(&dP1, queue );
//---------------------------------------

    // allocate memory for the scalar products
    CHECK( magma_dvinit( &beta, Magma_CPU, s, 1, c_zero, queue ));
    CHECK( magma_dvinit( &dbeta, Magma_DEV, s, 1, c_zero, queue ));

    if ( smoothing == 1 ) {
        // set smoothing solution vector
        CHECK( magma_dmtransfer( *x, &dxs, Magma_DEV, Magma_DEV, queue ));

        // set smoothing residual vector
        CHECK( magma_dmtransfer( dr, &drs, Magma_DEV, Magma_DEV, queue ));
    }

    // G(n,s) = 0
    CHECK( magma_dvinit( &dG, Magma_DEV, A.num_cols, s, c_zero, queue ));

    // U(n,s) = 0
    CHECK( magma_dvinit( &dU, Magma_DEV, A.num_cols, s, c_zero, queue ));

    // M1 = 0
    // M(s,s) = I
    CHECK( magma_dvinit( &dM1, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_dvinit( &dM, Magma_DEV, s, s, c_zero, queue ));
    magmablas_dlaset( MagmaFull, s, s, c_zero, c_one, dM.dval, s );

    // f = 0
    CHECK( magma_dvinit( &df, Magma_DEV, dP.num_cols, dr.num_cols, c_zero, queue ));

    // t = 0
    CHECK( magma_dvinit( &dt, Magma_DEV, A.num_rows, dr.num_cols, c_zero, queue ));

    // c = 0
    CHECK( magma_dvinit( &dc, Magma_DEV, dM.num_cols, dr.num_cols, c_zero, queue ));

    // v1 = 0
    // v = 0
    CHECK( magma_dvinit( &dv1, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &dv, Magma_DEV, dr.num_rows, dr.num_cols, c_zero, queue ));

    // piv = 0
    CHECK( magma_imalloc_pinned( &piv, s ));

    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }
    
    om = MAGMA_D_ONE;
    innerflag = 0;
    solver_par->numiter = 0;

    // start iteration
    do
    {
        solver_par->numiter++;
    
        // new RHS for small systems
        // f = (r' P)' = P' r
        magmablas_dgemv( MagmaConjTrans, dP.num_rows, dP.num_cols, c_one, dP.dval, dP.ld, dr.dval, 1, c_zero, df.dval, 1 );

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
    
            // solve small system and make v orthogonal to P
            // f(k:s) = M(k:s,k:s) c(k:s)
            magma_dcopyvector( sk, &df.dval[k], 1, &dc.dval[k], 1 );
            magma_dcopyvector( dM.num_rows * dM.num_cols, dM.dval, 1, dM1.dval, 1 );
            magma_dtrsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk, &dM1.dval[k*dM1.ld+k], dM1.ld, &dc.dval[k], 1 );

            // v1 = r - G(:,k:s) c(k:s)
            magma_dcopyvector( dr.num_rows * dr.num_cols, dr.dval, 1, dv1.dval, 1 );
            magmablas_dgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, &dG.dval[k*dG.ld], dG.ld, &dc.dval[k], 1, c_one, dv1.dval, 1 );

            // compute new U
            // U(:,k) = om * v1 + U(:,k:s) c(k:s)
            magmablas_dgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.ld], dU.ld, &dc.dval[k], 1, om, dv1.dval, 1 );
            magma_dcopyvector( dU.num_rows, dv1.dval, 1, &dU.dval[k*dU.ld], 1 );

            // compute new G
            // G(:,k) = A U(:,k)
            CHECK( magma_d_spmv( c_one, A, dv1, c_zero, dv, queue ));
            magma_dcopyvector( dG.num_rows, dv.dval, 1, &dG.dval[k*dG.ld], 1 );

            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {
                // alpha = P(:,i)' G(:,k) / M(i,i)
                alpha = magma_ddot( dP.num_rows, &dP.dval[i*dP.ld], 1, &dG.dval[k*dG.ld], 1 );
                magma_dgetvector( 1, &dM.dval[i*dM.ld+i], 1, &mkk, 1 );
                alpha = alpha / mkk;

                // G(:,k) = G(:,k) - alpha * G(:,i)
                magma_daxpy( dG.num_rows, -alpha, &dG.dval[i*dG.ld], 1, &dG.dval[k*dG.ld], 1 );

                // U(:,k) = U(:,k) - alpha * U(:,i)
                magma_daxpy( dU.num_rows, -alpha, &dU.dval[i*dU.ld], 1, &dU.dval[k*dU.ld], 1 );
            }

            // new column of M = P'G, first k-1 entries are zero
            // M(k:s,k) = (G(:,k)' P(:,k:s))' = P(:,k:s)' G(:,k)
            magmablas_dgemv( MagmaConjTrans, dP.num_rows, sk, c_one, &dP.dval[k*dP.ld], dP.ld, &dG.dval[k*dG.ld], 1, c_zero, &dM.dval[k*dM.ld+k], 1 );

            // check M(k,k) == 0
            magma_dgetvector( 1, &dM.dval[k*dM.ld+k], 1, &mkk, 1 );
            if ( MAGMA_D_EQUAL(mkk, MAGMA_D_ZERO) ) {
                info = MAGMA_DIVERGENCE;
                innerflag = 1;
                break;
            }

            // beta = f(k) / M(k,k)
            magma_dgetvector( 1, &df.dval[k], 1, &fk, 1 );
            beta.val[k] = fk / mkk;

            // make r orthogonal to q_i, i = 1..k
            // r = r - beta * G(:,k)
            magma_daxpy( dr.num_rows, -beta.val[k], &dG.dval[k*dG.ld], 1, dr.dval, 1 );

            if ( smoothing == 0 ) {
                // |r|
                nrmr = magma_dnrm2( b.num_rows * b.num_cols, dr.dval, 1 );
            }
            else if ( smoothing == 1 ) {
                // x = x + beta * U(:,k)
                magma_daxpy( x->num_rows, beta.val[k], &dU.dval[k*dU.ld], 1, x->dval, 1 );

                // smoothing operation
//---------------------------------------
                // t = rs - r
                magma_dcopyvector( drs.num_rows * drs.num_cols, drs.dval, 1, dt.dval, 1 );
                magma_daxpy( dt.num_rows, c_n_one, dr.dval, 1, dt.dval, 1 );

                // gamma = (t' * rs) / (|t| * |t|)
                nrmt = magma_dnrm2( dt.num_rows * dt.num_cols, dt.dval, 1 );
                gamma = magma_ddot( dt.num_rows, dt.dval, 1, drs.dval, 1 );
                gamma = gamma / (nrmt * nrmt);

                // rs = rs - gamma * t
                magma_daxpy( drs.num_rows, -gamma, dt.dval, 1, drs.dval, 1 );

                // xs = xs - gamma * (xs - x) 
                magma_dcopyvector( dxs.num_rows * dxs.num_cols, dxs.dval, 1, dt.dval, 1 );
                magma_daxpy( dt.num_rows, c_n_one, x->dval, 1, dt.dval, 1 );
                magma_daxpy( dxs.num_rows, -gamma, dt.dval, 1, dxs.dval, 1 );

                // |rs|
                nrmr = magma_dnrm2( b.num_rows * b.num_cols, drs.dval, 1 );           
            }
//---------------------------------------

            // store current timing and residual
            if ( solver_par->verbose > 0 ) {
                tempo2 = magma_sync_wtime( queue );
                if ( (solver_par->numiter) % solver_par->verbose == 0 ) {
                    solver_par->res_vec[(solver_par->numiter) / solver_par->verbose]
                            = (real_Double_t)nrmr;
                    solver_par->timing[(solver_par->numiter) / solver_par->verbose]
                            = (real_Double_t)tempo2 - tempo1;
                }
            }

            // check convergence or iteration limit
            if ( nrmr <= solver_par->atol ||
                nrmr/nrmb <= solver_par->rtol || 
                solver_par->numiter >= solver_par->maxiter ) {
                s = k; // for the x-update outside the loop
                innerflag = 1;
                break;
            }

            // new f = P' r (first k components are zero)
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                magma_daxpy( sk - 1, -beta.val[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1 );
            }

            // iter = iter + 1
            solver_par->numiter++;
        }

        // update solution approximation x
        if ( smoothing == 0 ) {
            magma_dsetvector( s, beta.val, 1, dbeta.dval, 1 );
            magmablas_dgemv( MagmaNoTrans, dU.num_rows, s, c_one, &dU.dval[0], dU.ld, &dbeta.dval[0], 1, c_one, &x->dval[0], 1 );
        }

        // check convergence or iteration limit or invalid of inner loop
        if ( innerflag == 1 ) {
            break;
        }

        // v = r
        magma_dcopyvector( dr.num_rows * dr.num_cols, dr.dval, 1, dv.dval, 1 );

        // t = A v
        CHECK( magma_d_spmv( c_one, A, dv, c_zero, dt, queue ));

        // computation of a new omega
//---------------------------------------
        // |t|
        nrmt = magma_dnrm2( dt.num_rows * dt.num_cols, dt.dval, 1 );

        // tr = t' r
        tr = magma_ddot( dr.num_rows, dt.dval, 1, dr.dval, 1 );

        // rho = abs(tr / (|t| * |r|))
        rho = MAGMA_D_ABS( MAGMA_D_REAL(tr) / (nrmt * nrmr) );

        // om = tr / (|t| * |t|)
        om = tr / (nrmt * nrmt);
        if ( rho < angle ) {
            om = om * angle / rho;
        }
//---------------------------------------
        if ( MAGMA_D_EQUAL(om, MAGMA_D_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // update approximation vector
        // x = x + om * v
        magma_daxpy(x->num_rows, om, dv.dval, 1, x->dval, 1);

        // update residual vector
        // r = r - om * t
        magma_daxpy(dr.num_rows, -om, dt.dval, 1, dr.dval, 1);

        if ( smoothing == 0 ) {
            // residual norm
            nrmr = magma_dnrm2( b.num_rows * b.num_cols, dr.dval, 1 );
        }
        else if ( smoothing == 1 ) {
            // smoothing operation
//---------------------------------------
            // t = rs - r
            magma_dcopyvector( drs.num_rows * drs.num_cols, drs.dval, 1, dt.dval, 1 );
            magma_daxpy( dt.num_rows, c_n_one, dr.dval, 1, dt.dval, 1 );

            // |t|
            nrmt = magma_dnrm2( dt.num_rows * dt.num_cols, dt.dval, 1 );

            // gamma = (t' * rs) / (|t| * |t|)
            gamma = magma_ddot( dt.num_rows, dt.dval, 1, drs.dval, 1 );
            gamma = gamma / (nrmt * nrmt);

            // rs = rs - gamma * t
            magma_daxpy( drs.num_rows, -gamma, dt.dval, 1, drs.dval, 1 );

            // xs = xs - gamma * (xs - x) 
            magma_dcopyvector( dxs.num_rows * dxs.num_cols, dxs.dval, 1, dt.dval, 1 );
            magma_daxpy( dt.num_rows, c_n_one, x->dval, 1, dt.dval, 1 );
            magma_daxpy( dxs.num_rows, -gamma, dt.dval, 1, dxs.dval, 1 );

            // |rs|
            nrmr = magma_dnrm2( b.num_rows * b.num_cols, drs.dval, 1 );           
        }
//---------------------------------------

        // store current timing and residual
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter) % solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter) / solver_par->verbose]
                        = (real_Double_t)nrmr;
                solver_par->timing[(solver_par->numiter) / solver_par->verbose]
                        = (real_Double_t)tempo2 - tempo1;
            }
        }

        // check convergence or iteration limit
        if ( nrmr <= solver_par->atol ||
            nrmr/nrmb <= solver_par->rtol || 
            solver_par->numiter >= solver_par->maxiter ) {
            break;
        }
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );

    if ( smoothing == 1 ) {
        magma_dcopyvector( dr.num_rows * dr.num_cols, drs.dval, 1, dr.dval, 1 );
        magma_dcopyvector( x->num_rows * x->num_cols, dxs.dval, 1, x->dval, 1 );
    }

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t)tempo2 - tempo1;
//--------------STOP TIME----------------

    // get final stats
    solver_par->iter_res = nrmr;
    CHECK( magma_dresidualvec( A, b, *x, &dr, &residual, NULL ));
    solver_par->final_res = residual;

    // set solver conclusion
    if ( info != MAGMA_SUCCESS && info != MAGMA_DIVERGENCE ) {
        if ( solver_par->init_res > solver_par->final_res ) {
            info = MAGMA_SLOW_CONVERGENCE;
        }
    }

    
cleanup:
    // free resources
    magma_dmfree( &P1, queue );
    magma_dmfree( &dP1, queue );
    magma_dmfree( &dP, queue );
    magma_dmfree( &dr, queue );
    magma_dmfree( &dG, queue );
    magma_dmfree( &dU, queue );
    magma_dmfree( &dM1, queue );
    magma_dmfree( &dM, queue );
    magma_dmfree( &df, queue );
    magma_dmfree( &dt, queue );
    magma_dmfree( &dc, queue );
    magma_dmfree( &dv1, queue );
    magma_dmfree( &dv, queue );
    magma_dmfree( &dxs, queue );
    magma_dmfree( &drs, queue );
    magma_dmfree( &dbeta, queue );
    magma_dmfree( &beta, queue );
    magma_free_pinned( piv );

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
    /* magma_didr */
}
