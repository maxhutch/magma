/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt
       @author Eduardo Ponce
       @author Moritz Kreutzer

       @generated from sparse-iter/src/zidr_strms.cpp normal z -> s, Mon May  2 23:30:58 2016
*/

#include "magmasparse_internal.h"
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Induced Dimension Reduction
    method applying kernel fusion and kernel overlap.

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
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sposv
    ********************************************************************/


extern "C" magma_int_t
magma_sidr_strms(
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;

    // prepare solver feedback
    solver_par->solver = Magma_IDRMERGE;
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

    // internal user options
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
    magma_int_t q;
    float residual;
    float nrm;
    float nrmb;
    float nrmr;
    float nrmt;
    float rho;
    float om;
    float gamma;

    // matrices and vectors
    magma_s_matrix dxs = {Magma_CSR};
    magma_s_matrix dr = {Magma_CSR}, drs = {Magma_CSR};
    magma_s_matrix dP = {Magma_CSR}, dP1 = {Magma_CSR};
    magma_s_matrix dG = {Magma_CSR}, dGcol = {Magma_CSR};
    magma_s_matrix dU = {Magma_CSR};
    magma_s_matrix dM = {Magma_CSR};
    magma_s_matrix df = {Magma_CSR};
    magma_s_matrix dt = {Magma_CSR}, dtt = {Magma_CSR};
    magma_s_matrix dc = {Magma_CSR};
    magma_s_matrix dv = {Magma_CSR};
    magma_s_matrix dskp = {Magma_CSR};
    magma_s_matrix dalpha = {Magma_CSR};
    magma_s_matrix dbeta = {Magma_CSR};
    float *hMdiag = NULL;
    float *hskp = NULL;
    float *halpha = NULL;
    float *hbeta = NULL;
    float *d1 = NULL, *d2 = NULL;
    
    // queue variables
    const magma_int_t nqueues = 3;     // number of queues
    magma_queue_t queues[nqueues];    

    // chronometry
    real_Double_t tempo1, tempo2;

    // create additional queues
    queues[0] = queue;
    for ( q = 1; q < nqueues; q++ ) {
        magma_queue_create( queue->device(), &(queues[q]) );
    }

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
    CHECK( magma_smalloc_pinned( &hskp, 5 ));
    CHECK( magma_svinit( &dskp, Magma_DEV, 4, 1, c_zero, queue ));

    CHECK( magma_smalloc_pinned( &halpha, s ));
    CHECK( magma_svinit( &dalpha, Magma_DEV, s, 1, c_zero, queue ));

    CHECK( magma_smalloc_pinned( &hbeta, s ));
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
    CHECK( magma_smalloc_pinned( &hMdiag, s ));
    magmablas_slaset( MagmaFull, dM.num_rows, dM.num_cols, c_zero, c_one, dM.dval, dM.ld, queue );

    // f = 0
    CHECK( magma_svinit( &df, Magma_DEV, dP.num_cols, 1, c_zero, queue ));

    // c = 0
    CHECK( magma_svinit( &dc, Magma_DEV, dM.num_cols, 1, c_zero, queue ));

    // v = r
    CHECK( magma_smtransfer( dr, &dv, Magma_DEV, Magma_DEV, queue ));

    //--------------START TIME---------------
    // chronometry
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->timing[0] = 0.0;
    }

cudaProfilerStart();

    om = MAGMA_S_ONE;
    gamma = MAGMA_S_ZERO;
    innerflag = 0;

    // new RHS for small systems
    // f = P' r
    // Q1
    magma_sgemvmdot_shfl( dP.num_rows, dP.num_cols, dP.dval, dr.dval, d1, d2, df.dval, queues[1] );

    // skp[4] = f(k)
    // Q1
    magma_sgetvector_async( 1, df.dval, 1, &hskp[4], 1, queues[1] );

    // c(k:s) = f(k:s)
    // Q1
    magma_scopyvector_async( s, df.dval, 1, dc.dval, 1, queues[1] );

    // c(k:s) = M(k:s,k:s) \ f(k:s)
    // Q1
    magma_strsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, s, dM.dval, dM.ld, dc.dval, 1, queues[1] );

    // start iteration
    do
    {
        solver_par->numiter++;

        // shadow space loop
        for ( k = 0; k < s; ++k ) {
            sk = s - k;
            dGcol.dval = dG.dval + k * dG.ld;

            // v = r - G(:,k:s) c(k:s)
            // Q1
            magmablas_sgemv( MagmaNoTrans, dG.num_rows, sk, c_n_one, dGcol.dval, dG.ld, &dc.dval[k], 1, c_one, dv.dval, 1, queues[1] );

            // U(:,k) = om * v + U(:,k:s) c(k:s)
            // Q1
            magmablas_sgemv( MagmaNoTrans, dU.num_rows, sk, c_one, &dU.dval[k*dU.ld], dU.ld, &dc.dval[k], 1, om, dv.dval, 1, queues[1] );

            // G(:,k) = A U(:,k)
            // Q1
            CHECK( magma_s_spmv( c_one, A, dv, c_zero, dGcol, queues[1] ));
            solver_par->spmv_count++;

            // bi-orthogonalize the new basis vectors
            for ( i = 0; i < k; ++i ) {
                // alpha = P(:,i)' G(:,k)
                // Q1
                halpha[i] = magma_sdot( dP.num_rows, &dP.dval[i*dP.ld], 1, dGcol.dval, 1, queues[1] );
                // implicit sync Q1 --> alpha = P(:,i)' G(:,k) 

                // alpha = alpha / M(i,i)
                halpha[i] = halpha[i] / hMdiag[i];
                    
                // G(:,k) = G(:,k) - alpha * G(:,i)
                // Q1
                magma_saxpy( dG.num_rows, -halpha[i], &dG.dval[i*dG.ld], 1, dGcol.dval, 1, queues[1] );
            }

            // sync Q1 --> G(:,k) = G(:,k) - alpha * G(:,i), skp[4] = f(k)
            magma_queue_sync( queues[1] );

            // new column of M = P'G, first k-1 entries are zero
            // M(k:s,k) = P(:,k:s)' G(:,k)
            // Q2
            magma_sgemvmdot_shfl( dP.num_rows, sk, &dP.dval[k*dP.ld], dGcol.dval, d1, d2, &dM.dval[k*dM.ld+k], queues[2] );

            // non-first s iteration
            if ( k > 0 ) {
                // alpha = dalpha
                // Q0
                magma_ssetvector_async( k, halpha, 1, dalpha.dval, 1, queues[0] );

                // U update outside of loop using GEMV
                // U(:,k) = U(:,k) - U(:,1:k) * alpha(1:k)
                // Q0
                magmablas_sgemv( MagmaNoTrans, dU.num_rows, k, c_n_one, dU.dval, dU.ld, dalpha.dval, 1, c_one, dv.dval, 1, queues[0] );
            }

            // Mdiag(k) = M(k,k)
            // Q2
            magma_sgetvector( 1, &dM.dval[k*dM.ld+k], 1, &hMdiag[k], 1, queues[2] );
            // implicit sync Q2 --> Mdiag(k) = M(k,k)

            // U(:,k) = v
            // Q0
            magma_scopyvector_async( dU.num_rows, dv.dval, 1, &dU.dval[k*dU.ld], 1, queues[0] );

            // check M(k,k) == 0
            if ( MAGMA_S_EQUAL(hMdiag[k], MAGMA_S_ZERO) ) {
                innerflag = 1;
                info = MAGMA_DIVERGENCE;
                break;
            }

            // beta = f(k) / M(k,k)
            hbeta[k] = hskp[4] / hMdiag[k];

            // check for nan
            if ( magma_s_isnan( hbeta[k] ) || magma_s_isinf( hbeta[k] )) {
                innerflag = 1;
                info = MAGMA_DIVERGENCE;
                break;
            }

            // r = r - beta * G(:,k)
            // Q2
            magma_saxpy( dr.num_rows, -hbeta[k], dGcol.dval, 1, dr.dval, 1, queues[2] );

            // non-last s iteration 
            if ( (k + 1) < s ) {
                // f(k+1:s) = f(k+1:s) - beta * M(k+1:s,k)
                // Q1
                magma_saxpy( sk-1, -hbeta[k], &dM.dval[k*dM.ld+(k+1)], 1, &df.dval[k+1], 1, queues[1] );

                // c(k+1:s) = f(k+1:s)
                // Q1
                magma_scopyvector_async( sk-1, &df.dval[k+1], 1, &dc.dval[k+1], 1, queues[1] );

                // c(k+1:s) = M(k+1:s,k+1:s) \ f(k+1:s)
                // Q1
                magma_strsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, sk-1, &dM.dval[(k+1)*dM.ld+(k+1)], dM.ld, &dc.dval[k+1], 1, queues[1] );

                // skp[4] = f(k+1)
                // Q1
                magma_sgetvector_async( 1, &df.dval[k+1], 1, &hskp[4], 1, queues[1] ); 
            }

            // smoothing disabled
            if ( smoothing <= 0 ) {
                // |r|
                // Q2
                nrmr = magma_snrm2( dr.num_rows, dr.dval, 1, queues[2] );           
                // implicit sync Q2 --> |r|

            // smoothing enabled
            } else {
                // smoothing operation
//---------------------------------------
                // t = rs - r
                // Q2
                magma_sidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queues[2] );

                // x = x + beta * U(:,k)
                // Q0
                magma_saxpy( x->num_rows, hbeta[k], &dU.dval[k*dU.ld], 1, x->dval, 1, queues[0] );

                // t't
                // t'rs
                // Q2
                CHECK( magma_sgemvmdot_shfl( dt.ld, 2, dtt.dval, dtt.dval, d1, d2, &dskp.dval[2], queues[2] ));

                // skp[2-3] = dskp[2-3]
                // Q2
                magma_sgetvector( 2, &dskp.dval[2], 1, &hskp[2], 1, queues[2] );
                // implicit sync Q2 --> skp = dskp

                // gamma = (t' * rs) / (t' * t)
                gamma = hskp[3] / hskp[2];
                
                // rs = rs - gamma * t 
                // Q1
                magma_saxpy( drs.num_rows, -gamma, dtt.dval, 1, drs.dval, 1, queues[1] );

                // xs = xs - gamma * (xs - x) 
                // Q0
                magma_sidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma, x->dval, dxs.dval, queues[0] );

                // |rs|
                // Q1
                nrmr = magma_snrm2( drs.num_rows, drs.dval, 1, queues[1] );       
                // implicit sync Q0 --> |r|
//---------------------------------------
            }

            // v = r
            // Q1
            magma_scopyvector_async( dr.num_rows, dr.dval, 1, dv.dval, 1, queues[1] );

            // last s iteration
            if ( (k + 1) == s ) {
               // t = A r
               // Q2
               CHECK( magma_s_spmv( c_one, A, dr, c_zero, dt, queues[2] ));
               solver_par->spmv_count++;

               // t't
               // t'r
               // Q2
               CHECK( magma_sgemvmdot_shfl( dt.ld, 2, dt.dval, dt.dval, d1, d2, dskp.dval, queues[2] ));
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
        }

        // smoothing disabled
        if ( smoothing <= 0 && innerflag != 1 ) {
            // dbeta(1:s) = beta(1:s)
            // Q0
            magma_ssetvector_async( s, hbeta, 1, dbeta.dval, 1, queues[0] );

            // x = x + U(:,1:s) * beta(1:s)
            // Q0
            magmablas_sgemv( MagmaNoTrans, dU.num_rows, s, c_one, dU.dval, dU.ld, dbeta.dval, 1, c_one, x->dval, 1, queues[0] );
        }

        // check convergence or iteration limit or invalid result of inner loop
        if ( innerflag > 0 ) {
            break;
        }

        // computation of a new omega
//---------------------------------------
        // skp[0-2] = dskp[0-2]
        // Q2
        magma_sgetvector( 2, dskp.dval, 1, hskp, 1, queues[2] );
        // implicit sync Q2 --> skp = dskp

        // |t|
        nrmt = magma_ssqrt( MAGMA_S_REAL(hskp[0]) );
        
        // rho = abs((t' * r) / (|t| * |r|))
        rho = MAGMA_D_ABS( MAGMA_S_REAL(hskp[1]) / (nrmt * nrmr) );

        // om = (t' * r) / (|t| * |t|)
        om = hskp[1] / hskp[0]; 
        if ( rho < angle ) {
            om = (om * angle) / rho;
        }
//---------------------------------------
        if ( MAGMA_S_EQUAL(om, MAGMA_S_ZERO) ) {
            info = MAGMA_DIVERGENCE;
            break;
        }

        // sync Q1 --> v = r
        magma_queue_sync( queues[1] );

        // r = r - om * t
        // Q2
        magma_saxpy( dr.num_rows, -om, dt.dval, 1, dr.dval, 1, queues[2] );

        // x = x + om * v
        // Q0
        magma_saxpy( x->num_rows, om, dv.dval, 1, x->dval, 1, queues[0] );

        // smoothing disabled
        if ( smoothing <= 0 ) {
            // |r|
            // Q2
            nrmr = magma_snrm2( dr.num_rows, dr.dval, 1, queues[2] );           
            // implicit sync Q2 --> |r|

            // v = r
            // Q0
            magma_scopyvector_async( dr.num_rows, dr.dval, 1, dv.dval, 1, queues[0] );

            // new RHS for small systems
            // f = P' r
            // Q1
            magma_sgemvmdot_shfl( dP.num_rows, dP.num_cols, dP.dval, dr.dval, d1, d2, df.dval, queues[1] );

            // skp[4] = f(k)
            // Q1
            magma_sgetvector_async( 1, df.dval, 1, &hskp[4], 1, queues[1] );

            // c(k:s) = f(k:s)
            // Q1
            magma_scopyvector_async( s, df.dval, 1, dc.dval, 1, queues[1] );

            // c(k:s) = M(k:s,k:s) \ f(k:s)
            // Q1
            magma_strsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, s, dM.dval, dM.ld, dc.dval, 1, queues[1] );

        // smoothing enabled
        } else {
            // smoothing operation
//---------------------------------------
            // t = rs - r
            // Q2
            magma_sidr_smoothing_1( drs.num_rows, drs.num_cols, drs.dval, dr.dval, dtt.dval, queues[2] );

            // t't
            // t'rs
            // Q2
            CHECK( magma_sgemvmdot_shfl( dt.ld, 2, dtt.dval, dtt.dval, d1, d2, &dskp.dval[2], queues[2] ));

            // skp[2-3] = dskp[2-3]
            // Q2
            magma_sgetvector( 2, &dskp.dval[2], 1, &hskp[2], 1, queues[2] );
            // implicit sync Q2 --> skp = dskp

            // gamma = (t' * rs) / (t' * t)
            gamma = hskp[3] / hskp[2];

            // rs = rs - gamma * (rs - r) 
            // Q2
            magma_saxpy( drs.num_rows, -gamma, dtt.dval, 1, drs.dval, 1, queues[2] );

            // xs = xs - gamma * (xs - x) 
            // Q0
            magma_sidr_smoothing_2( dxs.num_rows, dxs.num_cols, -gamma, x->dval, dxs.dval, queues[0] );

            // v = r
            // Q0
            magma_scopyvector_async( dr.num_rows, dr.dval, 1, dv.dval, 1, queues[0] );

            // new RHS for small systems
            // f = P' r
            // Q1
            magma_sgemvmdot_shfl( dP.num_rows, dP.num_cols, dP.dval, dr.dval, d1, d2, df.dval, queues[1] );

            // skp[4] = f(k)
            // Q1
            magma_sgetvector_async( 1, df.dval, 1, &hskp[4], 1, queues[1] );

            // c(k:s) = f(k:s)
            // Q1
            magma_scopyvector_async( s, df.dval, 1, dc.dval, 1, queues[1] );

            // |rs|
            // Q2
            nrmr = magma_snrm2( drs.num_rows, drs.dval, 1, queues[2] );           
            // implicit sync Q2 --> |r|

            // c(k:s) = M(k:s,k:s) \ f(k:s)
            // Q1
            magma_strsv( MagmaLower, MagmaNoTrans, MagmaNonUnit, s, dM.dval, dM.ld, dc.dval, 1, queues[1] );
//---------------------------------------
        }

        // store current timing and residual
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            magma_queue_sync( queue );
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
            info = MAGMA_SUCCESS;
            break;
        }

        // sync Q0 --> v = r
        magma_queue_sync( queues[0] );
    }
    while ( solver_par->numiter + 1 <= solver_par->maxiter );

    // sync all queues
    for ( q = 0; q < nqueues; q++ ) {
        magma_queue_sync( queues[q] );
    }

    // smoothing enabled
    if ( smoothing > 0 ) {
        // x = xs
        magma_scopyvector_async( x->num_rows, dxs.dval, 1, x->dval, 1, queue );

        // r = rs
        magma_scopyvector_async( dr.num_rows, drs.dval, 1, dr.dval, 1, queue );
    }

cudaProfilerStop();

    // get last iteration timing
    tempo2 = magma_sync_wtime( queue );
    magma_queue_sync( queue );
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
    // sync all queues, destory additional queues
    magma_queue_sync( queues[0] );
    for ( q = 1; q < nqueues; q++ ) {
        magma_queue_sync( queues[q] );
        magma_queue_destroy( queues[q] );
    }

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
    magma_smfree( &df, queue );
    magma_smfree( &dt, queue );
    magma_smfree( &dc, queue );
    magma_smfree( &dv, queue );
    magma_smfree( &dskp, queue );
    magma_smfree( &dalpha, queue );
    magma_smfree( &dbeta, queue );
    magma_free_pinned( hMdiag );
    magma_free_pinned( hskp );
    magma_free_pinned( halpha );
    magma_free_pinned( hbeta );
    magma_free( d1 );
    magma_free( d2 );

    solver_par->info = info;
    return info;
    /* magma_sidr_strms */
}
