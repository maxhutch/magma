/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt
       @author Eduardo Ponce
       @author Moritz Kreutzer

       @generated from sparse-iter/src/zpidr_merge.cpp normal z -> s, Mon May  2 23:31:00 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned Induced Dimension
    Reduction method applying kernel fusion.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in]
    b           magma_s_matrix
                RHS b

    @param[in,out]
    x           magma_s_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in]
    precond_par magma_s_preconditioner*
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sposv
    ********************************************************************/


extern "C" magma_int_t
magma_spidr_merge(
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;

    // prepare solver feedback
    solver_par->solver = Magma_PIDRMERGE;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    solver_par->init_res = 0.0;
    solver_par->final_res = 0.0;
    solver_par->iter_res = 0.0;
    solver_par->runtime = 0.0;

    // constants
    const float c_zero = MAGMA_S_ZERO;
    const float c_one = MAGMA_S_ONE;
    const float c_n_one = MAGMA_S_NEG_ONE;

    // internal user parameters
    const magma_int_t smoothing = 1;   // 0 = disable, 1 = enable
    const float angle = 0.7;          // [0-1]

    // local variables
    magma_int_t iseed[4] = {0, 0, 0, 1};
    magma_int_t dof;
    magma_int_t s;
    magma_int_t distr;
    magma_int_t k, i, sk;
    magma_int_t innerflag;
    magma_int_t ldd;
    float residual;
    float nrm;
    float nrmb;
    float nrmr;
    float nrmt;
    float rho;
    float om;
    float gamma;
    float fk;

    // matrices and vectors
    magma_s_matrix dxs = {Magma_CSR};
    magma_s_matrix dr = {Magma_CSR}, drs = {Magma_CSR};
    magma_s_matrix dP = {Magma_CSR}, dP1 = {Magma_CSR};
    magma_s_matrix dG = {Magma_CSR}, dGcol = {Magma_CSR};
    magma_s_matrix dU = {Magma_CSR};
    magma_s_matrix dM = {Magma_CSR}, hMdiag = {Magma_CSR};
    magma_s_matrix df = {Magma_CSR};
    magma_s_matrix dt = {Magma_CSR}, dtt = {Magma_CSR};
    magma_s_matrix dc = {Magma_CSR};
    magma_s_matrix dv = {Magma_CSR};
    magma_s_matrix dlu = {Magma_CSR};
    magma_s_matrix dskp = {Magma_CSR}, hskp = {Magma_CSR};
    magma_s_matrix dalpha = {Magma_CSR}, halpha = {Magma_CSR};
    magma_s_matrix dbeta = {Magma_CSR}, hbeta = {Magma_CSR};
    float *d1 = NULL, *d2 = NULL;

    // chronometry
    real_Double_t tempo1, tempo2;

    // initial s space
    // TODO: add option for 's' (shadow space number)
    // Hack: uses '--restart' option as the shadow space number.
    //       This is not a good idea because the default value of restart option is used to detect
    //       if the user provided a custom restart. This means that if the default restart value
    //       is changed then the code will think it was the user (unless the default value is
    //       also updated in the 'if' statement below.
    s = 1;
    if ( solver_par->restart != 50 ) {
        if ( solver_par->restart > A.num_cols ) {
            s = A.num_cols;
        } else {
            s = solver_par->restart;
        }
    }
    solver_par->restart = s;

    // set max iterations
    solver_par->maxiter = min( 2 * A.num_cols, solver_par->maxiter );

    // check if matrix A is square
    if ( A.num_rows != A.num_cols ) {
        //printf("Matrix A is not square.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }

    // |b|
    nrmb = magma_snrm2( b.num_rows, b.dval, 1, queue );
    if ( nrmb == 0.0 ) {
        magma_sscal( x->num_rows, MAGMA_S_ZERO, x->dval, 1, queue );
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    // t = 0
    // make t twice as large to contain both, dt and dr
    ldd = magma_roundup( b.num_rows, 32 );
    CHECK( magma_svinit( &dt, Magma_DEV, ldd, 2, c_zero, queue ));
    dt.num_rows = b.num_rows;
    dt.num_cols = 1;
    dt.nnz = dt.num_rows;

    // redirect the dr.dval to the second part of dt
    CHECK( magma_svinit( &dr, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    magma_free( dr.dval );
    dr.dval = dt.dval + ldd;

    // r = b - A x
    CHECK( magma_sresidualvec( A, b, *x, &dr, &nrmr, queue ));
    
    // |r|
    solver_par->init_res = nrmr;
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nrmr;
    }

    // check if initial is guess good enough
    if ( nrmr <= solver_par->atol ||
        nrmr/nrmb <= solver_par->rtol ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    // P = randn(n, s)
    // P = ortho(P)
//---------------------------------------
    // P = 0.0
    CHECK( magma_svinit( &dP, Magma_CPU, A.num_cols, s, c_zero, queue ));

    // P = randn(n, s)
    distr = 3;        // 1 = unif (0,1), 2 = unif (-1,1), 3 = normal (0,1) 
    dof = dP.num_rows * dP.num_cols;
    lapackf77_slarnv( &distr, iseed, &dof, dP.val );

    // transfer P to device
    CHECK( magma_smtransfer( dP, &dP1, Magma_CPU, Magma_DEV, queue ));
    magma_smfree( &dP, queue );

    // P = ortho(P1)
    if ( dP1.num_cols > 1 ) {
        // P = magma_sqr(P1), QR factorization
        CHECK( magma_sqr( dP1.num_rows, dP1.num_cols, dP1, dP1.ld, &dP, NULL, queue ));
    } else {
        // P = P1 / |P1|
        nrm = magma_snrm2( dof, dP1.dval, 1, queue );
        nrm = 1.0 / nrm;
        magma_sscal( dof, nrm, dP1.dval, 1, queue );
        CHECK( magma_smtransfer( dP1, &dP, Magma_DEV, Magma_DEV, queue ));
    }
    magma_smfree( &dP1, queue );
//---------------------------------------

    // allocate memory for the scalar products
    CHECK( magma_svinit( &hskp, Magma_CPU, 4, 1, c_zero, queue ));
    CHECK( magma_svinit( &dskp, Magma_DEV, 4, 1, c_zero, queue ));

    CHECK( magma_svinit( &halpha, Magma_CPU, s, 1, c_zero, queue ));
    CHECK( magma_svinit( &dalpha, Magma_DEV, s, 1, c_zero, queue ));

    CHECK( magma_svinit( &hbeta, Magma_CPU, s, 1, c_zero, queue ));
    CHECK( magma_svinit( &dbeta, Magma_DEV, s, 1, c_zero, queue ));

    // workspace for merged dot product
    CHECK( magma_smalloc( &d1, max(2, s) * b.num_rows ));
    CHECK( magma_smalloc( &d2, max(2, s) * b.num_rows ));

    // smoothing enabled
    if ( smoothing > 0 ) {
        // set smoothing solution vector
        CHECK( magma_smtransfer( *x, &dxs, Magma_DEV, Magma_DEV, queue ));

        // tt = 0
        // make tt twice as large to contain both, dtt and drs
        ldd = magma_roundup( b.num_rows, 32 );
        CHECK( magma_svinit( &dtt, Magma_DEV, ldd, 2, c_zero, queue ));
        dtt.num_rows = dr.num_rows;
        dtt.num_cols = 1;
        dtt.nnz = dtt.num_rows;

        // redirect the drs.dval to the second part of dtt
        CHECK( magma_svinit( &drs, Magma_DEV, dr.num_rows, 1, c_zero, queue ));
        magma_free( drs.dval );
        drs.dval = dtt.dval + ldd;

        // set smoothing residual vector
        magma_scopyvector( dr.num_rows, dr.dval, 1, drs.dval, 1, queue );
    }

    // G(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_rows, 32 );
        CHECK( magma_svinit( &dG, Magma_DEV, ldd, s, c_zero, queue ));
        dG.num_rows = A.num_rows;
    } else {
        CHECK( magma_svinit( &dG, Magma_DEV, A.num_rows, s, c_zero, queue ));
    }

    // dGcol represents a single column of dG, array pointer is set inside loop
    CHECK( magma_svinit( &dGcol, Magma_DEV, dG.num_rows, 1, c_zero, queue ));
    magma_free( dGcol.dval );

    // U(n,s) = 0
    if ( s > 1 ) {
        ldd = magma_roundup( A.num_cols, 32 );
        CHECK( magma_svinit( &dU, Magma_DEV, ldd, s, c_zero, queue ));
        dU.num_rows = A.num_cols;
    } else {
        CHECK( magma_svinit( &dU, Magma_DEV, A.num_cols, s, c_zero, queue ));
    }

    // M(s,s) = I
    CHECK( magma_svinit( &dM, Magma_DEV, s, s, c_zero, queue ));
    CHECK( magma_svinit( &hMdiag, Magma_CPU, s, 1, c_zero, queue ));
    magmablas_slaset( MagmaFull, dM.num_rows, dM.num_cols, c_zero, c_one, dM.dval, dM.ld, queue );

    // f = 0
    CHECK( magma_svinit( &df, Magma_DEV, dP.num_cols, 1, c_zero, queue ));

    // c = 0
    CHECK( magma_svinit( &dc, Magma_DEV, dM.num_cols, 1, c_zero, queue ));

    // v = 0
    CHECK( magma_svinit( &dv, Magma_DEV, dr.num_rows, 1, c_zero, queue ));

    // lu = 0
    CHECK( magma_svinit( &dlu, Magma_DEV, dr.num_rows, 1, c_zero, queue ));

    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }

    om = MAGMA_S_ONE;
    innerflag = 0;

    // start iteration
    do
    {
        solver_par->numiter++;
    
        // new RHS for small systems
        // f = P' r
        magma_sgemvmdot_shfl( dP.num_rows, dP.num_cols, dP.dval, dr.dval, d1, d2, df.dval, queue );

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
    
            // c(k:s) = M(k:s,k:s) \ f(k:s)
            magma_scopyvector( sk, &df.dval[k], 1, &dc.dval[k], 1, queue );
            magma_strsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk, &dM.dval[k*dM.ld+k], dM.ld, &dc.dval[k], 1, queue );

            // v = r - G(:,k:s) c(k:s)
            magma_scopyvector( dr.num_rows, dr.dval, 1, dv.dval, 1, queue );
            magmablas_sgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, &dG.dval[k*dG.ld], dG.ld, &dc.dval[k], 1, c_one, dv.dval, 1, queue );

            // preconditioning operation 
            // v = L \ v;
            // v = U \ v;
            CHECK( magma_s_applyprecond_left( MagmaNoTrans, A, dv, &dlu, precond_par, queue )); 
            CHECK( magma_s_applyprecond_right( MagmaNoTrans, A, dlu, &dv, precond_par, queue )); 
            
            // U(:,k) = om * v + U(:,k:s) c(k:s)
            magmablas_sgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.ld], dU.ld, &dc.dval[k], 1, om, dv.dval, 1, queue );
            magma_scopyvector( dU.num_rows, dv.dval, 1, &dU.dval[k*dU.ld], 1, queue );

            // G(:,k) = A U(:,k)
            dGcol.dval = dG.dval + k * dG.ld;
            CHECK( magma_s_spmv( c_one, A, dv, c_zero, dGcol, queue ));
            solver_par->spmv_count++;

            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {
                // alpha = P(:,i)' G(:,k)
                halpha.val[i] = magma_sdot( dP.num_rows, &dP.dval[i*dP.ld], 1, &dG.dval[k*dG.ld], 1, queue );

                // alpha = alpha / M(i,i)
                halpha.val[i] = halpha.val[i] / hMdiag.val[i];
                
                // G(:,k) = G(:,k) - alpha * G(:,i)
                magma_saxpy( dG.num_rows, -halpha.val[i], &dG.dval[i*dG.ld], 1, &dG.dval[k*dG.ld], 1, queue );
            }

            // non-first s iteration
            if ( k > 0 ) {
                // U update outside of loop using GEMV
                // U(:,k) = U(:,k) - U(:,1:k) * alpha(1:k)
                magma_ssetvector( k, halpha.val, 1, dalpha.dval, 1, queue );
                magmablas_sgemv( MagmaNoTrans, dU.num_rows, k, c_n_one, dU.dval, dU.ld, dalpha.dval, 1, c_one, &dU.dval[k*dU.ld], 1, queue );
            }

            // new column of M = P'G, first k-1 entries are zero
            // M(k:s,k) = P(:,k:s)' G(:,k)
            magma_sgemvmdot_shfl( dP.num_rows, sk, &dP.dval[k*dP.ld], &dG.dval[k*dG.ld], d1, d2, &dM.dval[k*dM.ld+k], queue );
            magma_sgetvector( 1, &dM.dval[k*dM.ld+k], 1, &hMdiag.val[k], 1, queue );

            // check M(k,k) == 0
            if ( MAGMA_S_EQUAL(hMdiag.val[k], MAGMA_S_ZERO) ) {
                innerflag = 1;
                info = MAGMA_DIVERGENCE;
                break;
            }

            // beta = f(k) / M(k,k)
            magma_sgetvector( 1, &df.dval[k], 1, &fk, 1, queue );
            hbeta.val[k] = fk / hMdiag.val[k]; 

            // check for nan
            if ( magma_s_isnan( hbeta.val[k] ) || magma_s_isinf( hbeta.val[k] )) {
                innerflag = 1;
                info = MAGMA_DIVERGENCE;
                break;
            }

            // r = r - beta * G(:,k)
            magma_saxpy( dr.num_rows, -hbeta.val[k], &dG.dval[k*dG.ld], 1, dr.dval, 1, queue );

            // smoothing disabled
            if ( smoothing <= 0 ) {
                // |r|
                nrmr = magma_snrm2( dr.num_rows, dr.dval, 1, queue );

            // smoothing enabled
            } else {
                // x = x + beta * U(:,k)
                magma_saxpy( x->num_rows, hbeta.val[k], &dU.dval[k*dU.ld], 1, x->dval, 1, queue );

                // smoothing operation
//---------------------------------------
                // t = rs - r
                magma_sidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queue );

                // t't
                // t'rs
                CHECK( magma_sgemvmdot_shfl( dt.ld, 2, dtt.dval, dtt.dval, d1, d2, &dskp.dval[2], queue ));
                magma_sgetvector( 2, &dskp.dval[2], 1, &hskp.val[2], 1, queue );

                // gamma = (t' * rs) / (t' * t)
                gamma = hskp.val[3] / hskp.val[2];
                
                // rs = rs - gamma * (rs - r) 
                magma_saxpy( drs.num_rows, -gamma, dtt.dval, 1, drs.dval, 1, queue );

                // xs = xs - gamma * (xs - x) 
                magma_sidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma, x->dval, dxs.dval, queue );

                // |rs|
                nrmr = magma_snrm2( drs.num_rows, drs.dval, 1, queue );       
//---------------------------------------
            }

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
                nrmr/nrmb <= solver_par->rtol ) {
                s = k + 1; // for the x-update outside the loop
                innerflag = 2;
                info = MAGMA_SUCCESS;
                break;
            }
            
            // non-last s iteration
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                magma_saxpy( sk-1, -hbeta.val[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1, queue );
            }

        }

        // smoothing disabled
        if ( smoothing <= 0 && innerflag != 1 ) {
            // update solution approximation x
            // x = x + U(:,1:s) * beta(1:s)
            magma_ssetvector( s, hbeta.val, 1, dbeta.dval, 1, queue );
            magmablas_sgemv( MagmaNoTrans, dU.num_rows, s, c_one, dU.dval, dU.ld, dbeta.dval, 1, c_one, x->dval, 1, queue );
        }

        // check convergence or iteration limit or invalid result of inner loop
        if ( innerflag > 0 ) {
            break;
        }

        // v = r
        magma_scopy( dr.num_rows, dr.dval, 1, dv.dval, 1, queue );

        // preconditioning operation 
        // v = L \ v;
        // v = U \ v;
        CHECK( magma_s_applyprecond_left( MagmaNoTrans, A, dv, &dlu, precond_par, queue )); 
        CHECK( magma_s_applyprecond_right( MagmaNoTrans, A, dlu, &dv, precond_par, queue )); 
            
        // t = A v
        CHECK( magma_s_spmv( c_one, A, dv, c_zero, dt, queue ));
        solver_par->spmv_count++;

        // computation of a new omega
//---------------------------------------
        // t't
        // t'r 
        CHECK( magma_sgemvmdot_shfl( dt.ld, 2, dt.dval, dt.dval, d1, d2, dskp.dval, queue ));
        magma_sgetvector( 2, dskp.dval, 1, hskp.val, 1, queue );

        // |t| 
        nrmt = magma_ssqrt( MAGMA_S_REAL(hskp.val[0]) );

        // rho = abs((t' * r) / (|t| * |r|))
        rho = MAGMA_D_ABS( MAGMA_S_REAL(hskp.val[1]) / (nrmt * nrmr) );

        // om = (t' * r) / (|t| * |t|)
        om = hskp.val[1] / hskp.val[0];
        if ( rho < angle ) {
            om = (om * angle) / rho;
        }
//---------------------------------------
        if ( MAGMA_S_EQUAL(om, MAGMA_S_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // update approximation vector
        // x = x + om * v
        magma_saxpy( x->num_rows, om, dv.dval, 1, x->dval, 1, queue );

        // update residual vector
        // r = r - om * t
        magma_saxpy( dr.num_rows, -om, dt.dval, 1, dr.dval, 1, queue );

        // smoothing disabled
        if ( smoothing <= 0 ) {
            // residual norm
            nrmr = magma_snrm2( dr.num_rows, dr.dval, 1, queue );

        // smoothing enabled
        } else {
            // smoothing operation
//---------------------------------------
            // t = rs - r
            magma_sidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queue );

            // t't
            // t'rs
            CHECK( magma_sgemvmdot_shfl( dt.ld, 2, dtt.dval, dtt.dval, d1, d2, &dskp.dval[2], queue ));
            magma_sgetvector( 2, &dskp.dval[2], 1, &hskp.val[2], 1, queue );

            // gamma = (t' * rs) / (t' * t)
            gamma = hskp.val[3] / hskp.val[2];

            // rs = rs - gamma * (rs - r) 
            magma_saxpy( drs.num_rows, -gamma, dtt.dval, 1, drs.dval, 1, queue );

            // xs = xs - gamma * (xs - x) 
            magma_sidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma, x->dval, dxs.dval, queue );

            // |rs|
            nrmr = magma_snrm2( drs.num_rows, drs.dval, 1, queue );           
//---------------------------------------
        }

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

        // check convergence
        if ( nrmr <= solver_par->atol ||
            nrmr/nrmb <= solver_par->rtol ) {
            info = MAGMA_SUCCESS;
            break;
        }
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );

    // smoothing enabled
    if ( smoothing > 0 ) {
        // x = xs
        magma_scopyvector( x->num_rows, dxs.dval, 1, x->dval, 1, queue );

        // r = rs
        magma_scopyvector( dr.num_rows, drs.dval, 1, dr.dval, 1, queue );
    }

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t)tempo2 - tempo1;
//--------------STOP TIME----------------

    // get final stats
    solver_par->iter_res = nrmr;
    CHECK( magma_sresidualvec( A, b, *x, &dr, &residual, queue ));
    solver_par->final_res = residual;

    // set solver conclusion
    if ( info != MAGMA_SUCCESS && info != MAGMA_DIVERGENCE ) {
        if ( solver_par->init_res > solver_par->final_res ) {
            info = MAGMA_SLOW_CONVERGENCE;
        }
    }


cleanup:
    // free resources
    // smoothing enabled
    if ( smoothing > 0 ) {
        drs.dval = NULL;  // needed because its pointer is redirected to dtt
        magma_smfree( &dxs, queue );
        magma_smfree( &drs, queue ); 
        magma_smfree( &dtt, queue );
    }
    dr.dval = NULL;       // needed because its pointer is redirected to dt
    dGcol.dval = NULL;    // needed because its pointer is redirected to dG
    magma_smfree( &dr, queue );
    magma_smfree( &dP, queue );
    magma_smfree( &dP1, queue );
    magma_smfree( &dG, queue );
    magma_smfree( &dGcol, queue );
    magma_smfree( &dU, queue );
    magma_smfree( &dM, queue );
    magma_smfree( &hMdiag, queue );
    magma_smfree( &df, queue );
    magma_smfree( &dt, queue );
    magma_smfree( &dc, queue );
    magma_smfree( &dv, queue );
    magma_smfree( &dlu, queue );
    magma_smfree( &dskp, queue );
    magma_smfree( &dalpha, queue );
    magma_smfree( &dbeta, queue );
    magma_smfree( &hskp, queue );
    magma_smfree( &halpha, queue );
    magma_smfree( &hbeta, queue );
    magma_free( d1 );
    magma_free( d2 );

    solver_par->info = info;
    return info;
    /* magma_spidr_merge */
}
