/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zbombard.cpp normal z -> d, Mon May  2 23:31:01 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real general matrix A.
    This is a GPU implementation of the iterative bombardment suggested in
    Barrett et al.
    ''Algorithmic bombardment for the iterative solution of linear systems: 
      A poly-iterative approach''
    using BiCGSTAB, CGS and MQR. At this point, it only works for symmetric A.

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

    @ingroup magmasparse_dgesv
    ********************************************************************/

extern "C" magma_int_t
magma_dbombard(
    magma_d_matrix A, magma_d_matrix b, 
    magma_d_matrix *x, magma_d_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // 1=QMR, 2=CGS, 3+BiCGSTAB
    magma_int_t flag = 0;
    
    // prepare solver feedback
    solver_par->solver = Magma_BOMBARD;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // local variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE;
                        
    // solver variables
    double nom0, r0, res, Q_res, T_res, C_res, B_res, nomb;
    
    //QMR
    double Q_rho = c_one, Q_rho1 = c_one, Q_eta = -c_one , Q_pds = c_one, 
                        Q_thet = c_one, Q_thet1 = c_one, Q_epsilon = c_one, 
                        Q_beta = c_one, Q_delta = c_one, Q_pde = c_one, Q_rde = c_one,
                        Q_gamm = c_one, Q_gamm1 = c_one, Q_psi = c_one;
                        
    //TFQMR
    double T_rho = c_one, T_rho_l = c_one, T_eta = c_zero , T_c = c_zero , 
                        T_theta = c_zero , T_tau = c_zero, T_alpha = c_one, T_beta = c_zero,
                        T_sigma = c_zero;
                        
    //CGS
    double C_rho, C_rho_l = c_one, C_alpha, C_beta = c_zero;
    
    //BiCGSTAB
    double B_alpha, B_beta, B_omega, B_rho_old, B_rho_new;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // need to transpose the matrix
    
    // GPU workspace
    // QMR
    magma_d_matrix AT = {Magma_CSR}, Ah1 = {Magma_CSR}, Ah2 = {Magma_CSR},
                    Q_r={Magma_CSR}, r_tld={Magma_CSR}, Q_x={Magma_CSR},
                    Q_v={Magma_CSR}, Q_w={Magma_CSR}, Q_wt={Magma_CSR},
                    Q_d={Magma_CSR}, Q_s={Magma_CSR}, Q_z={Magma_CSR}, Q_q={Magma_CSR}, 
                    Q_p={Magma_CSR}, Q_pt={Magma_CSR}, Q_y={Magma_CSR}, d1={Magma_CSR}, d2={Magma_CSR};
    //TFQMR
    // GPU workspace
    magma_d_matrix  T_r={Magma_CSR}, T_pu_m={Magma_CSR}, T_x={Magma_CSR},
                    T_d={Magma_CSR}, T_w={Magma_CSR}, T_v={Magma_CSR},
                    T_u_mp1={Magma_CSR}, T_u_m={Magma_CSR}, T_Au={Magma_CSR}, 
                    T_Ad={Magma_CSR}, T_Au_new={Magma_CSR};
                    
    // CGS
    magma_d_matrix C_r={Magma_CSR}, C_rt={Magma_CSR}, C_x={Magma_CSR},
                    C_p={Magma_CSR}, C_q={Magma_CSR}, C_u={Magma_CSR}, C_v={Magma_CSR},  C_t={Magma_CSR},
                    C_p_hat={Magma_CSR}, C_q_hat={Magma_CSR}, C_u_hat={Magma_CSR}, C_v_hat={Magma_CSR};
    //BiCGSTAB
    magma_d_matrix B_r={Magma_CSR}, B_x={Magma_CSR}, B_p={Magma_CSR}, B_v={Magma_CSR}, 
                    B_s={Magma_CSR}, B_t={Magma_CSR};

                    
    CHECK( magma_dvinit( &r_tld, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d1, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d2, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // QMR
    CHECK( magma_dvinit( &Q_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_wt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Q_x, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // TFQMR
    CHECK( magma_dvinit( &T_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &T_u_mp1,Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &T_u_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &T_pu_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &T_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &T_d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &T_w, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &T_Ad, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &T_Au_new, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &T_Au, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &T_x, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // CGS
    CHECK( magma_dvinit( &C_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_rt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_x,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_p_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_q_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_u, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_u_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_v_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &C_t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // BiCGSTAB
    CHECK( magma_dvinit( &B_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &B_x,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &B_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &B_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &B_s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &B_t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver setup
    CHECK(  magma_dresidualvec( A, b, *x, &r_tld, &nom0, queue));
    solver_par->init_res = nom0;
    res = nom0;
    
    // QMR
    magma_dcopy( dofs, r_tld.dval, 1, Q_r.dval, 1, queue );   
    magma_dcopy( dofs, r_tld.dval, 1, Q_y.dval, 1, queue );   
    magma_dcopy( dofs, r_tld.dval, 1, Q_v.dval, 1, queue );  
    magma_dcopy( dofs, r_tld.dval, 1, Q_wt.dval, 1, queue );   
    magma_dcopy( dofs, r_tld.dval, 1, Q_z.dval, 1, queue ); 
    magma_dcopy( dofs, x->dval, 1, Q_x.dval, 1, queue ); 
    // transpose the matrix
    // transpose the matrix
    magma_dmtransfer( A, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_dmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransposeconjugate( Ah2, &Ah1, queue );
    magma_dmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_dmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransfer( Ah2, &AT, Magma_CPU, Magma_DEV, queue );
    magma_dmfree(&Ah2, queue );
    
    // TFQMR
    solver_par->init_res = nom0;
    magma_dcopy( dofs, r_tld.dval, 1, T_r.dval, 1, queue );   
    magma_dcopy( dofs, T_r.dval, 1, T_w.dval, 1, queue );   
    magma_dcopy( dofs, T_r.dval, 1, T_u_m.dval, 1, queue );  
    magma_dcopy( dofs, T_r.dval, 1, T_u_mp1.dval, 1, queue ); 
    magma_dcopy( dofs, T_u_m.dval, 1, T_pu_m.dval, 1, queue );  
    CHECK( magma_d_spmv( c_one, A, T_pu_m, c_zero, T_v, queue ));
    magma_dcopy( dofs, T_v.dval, 1, T_Au.dval, 1, queue );  
    
    // CGS
    magma_dcopy( dofs, r_tld.dval, 1, C_r.dval, 1, queue );   
    magma_dcopy( dofs, x->dval, 1, C_x.dval, 1, queue ); 
    
    // BiCGSTAB
    magma_dcopy( dofs, r_tld.dval, 1, B_r.dval, 1, queue );   
    magma_dcopy( dofs, x->dval, 1, B_x.dval, 1, queue ); 
    CHECK( magma_d_spmv( c_one, A, B_r, c_zero, B_v, queue ));     

    
    nomb = magma_dnrm2( dofs, b.dval, 1, queue );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }
    
    T_tau = magma_dsqrt( magma_ddot( dofs, T_r.dval, 1, r_tld.dval, 1, queue) );
    T_rho = magma_ddot( dofs, T_r.dval, 1, r_tld.dval, 1, queue );
    T_rho_l = T_rho;
    

    Q_psi = magma_dsqrt( magma_ddot( dofs, Q_z.dval, 1, Q_z.dval, 1, queue ));
    Q_rho = magma_dsqrt( magma_ddot( dofs, Q_y.dval, 1, Q_y.dval, 1, queue ));
    
    // BiCGSTAB
    B_rho_new = magma_ddot( dofs, B_r.dval, 1, B_r.dval, 1, queue );            
    B_rho_old = B_omega = B_alpha = MAGMA_D_MAKE( 1.0, 0. );
    
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
    magma_dqmr_1(  
    b.num_rows, 
    b.num_cols, 
    Q_rho,
    Q_psi,
    Q_y.dval, 
    Q_z.dval,
    Q_v.dval,
    Q_w.dval,
    queue );
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
            //QMR: delta = z' * y;
        Q_delta = magma_ddot( dofs, Q_z.dval, 1, Q_y.dval, 1, queue );
        
        // TFQMR
        T_alpha = T_rho / magma_ddot( dofs, T_v.dval, 1, r_tld.dval, 1, queue );
        T_sigma = T_theta * T_theta / T_alpha * T_eta; 
        
        
            //CGS: rho = r' * r_tld
        C_rho = magma_ddot( dofs, C_r.dval, 1, r_tld.dval, 1, queue );
        
            // BiCGSTAB
        B_rho_old = B_rho_new;    
        B_rho_new = magma_ddot( dofs, r_tld.dval, 1, B_r.dval, 1, queue );  // rho=<rr,r>
        B_beta = B_rho_new/B_rho_old * B_alpha/B_omega;   // beta=rho/rho_old *alpha/omega

        
        if( solver_par->numiter == 1 ){
                //QMR: p = y;
                //QMR: q = z;
            magma_dcopy( dofs, Q_y.dval, 1, Q_p.dval, 1, queue );
            magma_dcopy( dofs, Q_z.dval, 1, Q_q.dval, 1, queue );
            
                //QMR: u = r;
                //QMR: p = r;
            magma_dcgs_2(  
            b.num_rows, 
            b.num_cols, 
            C_r.dval,
            C_u.dval,
            C_p.dval,
            queue );
        }
        else{
            Q_pde = Q_psi * Q_delta / Q_epsilon;
            Q_rde = Q_rho * MAGMA_D_CONJ(Q_delta/Q_epsilon);
            
            C_beta = C_rho / C_rho_l;  
            
                //QMR p = y - pde * p
                //QMR q = z - rde * q
            magma_dqmr_2(  
            b.num_rows, 
            b.num_cols, 
            Q_pde,
            Q_rde,
            Q_y.dval,
            Q_z.dval,
            Q_p.dval, 
            Q_q.dval, 
            queue );
            
                  //CGS: u = r + beta*q;
                  //CGS: p = u + beta*( q + beta*p );
            magma_dcgs_1(  
            b.num_rows, 
            b.num_cols, 
            C_beta,
            C_r.dval,
            C_q.dval, 
            C_u.dval,
            C_p.dval,
            queue );
        }
        
        // TFQMR
        magma_dtfqmr_1(  
        b.num_rows, 
        b.num_cols, 
        T_alpha,
        T_sigma,
        T_v.dval, 
        T_Au.dval,
        T_u_m.dval,
        T_pu_m.dval,
        T_u_mp1.dval,
        T_w.dval, 
        T_d.dval,
        T_Ad.dval,
        queue );
        
        T_theta = magma_dsqrt( magma_ddot(dofs, T_w.dval, 1, T_w.dval, 1, queue) ) / T_tau;
        T_c = c_one / magma_dsqrt( c_one + T_theta*T_theta );
        T_tau = T_tau * T_theta *T_c;
        T_eta = T_c * T_c * T_alpha;
        T_sigma = T_theta * T_theta / T_alpha * T_eta;  
        
        magma_dtfqmr_2(  
        b.num_rows, 
        b.num_cols, 
        T_eta,
        T_d.dval,
        T_Ad.dval,
        T_x.dval, 
        T_r.dval, 
        queue );
        magma_dcopy( dofs, T_u_mp1.dval, 1, T_pu_m.dval, 1, queue );
        
            // BiCGSTAB: p = r + beta * ( p - omega * v )
        magma_dbicgstab_1(  
        b.num_rows, 
        b.num_cols, 
        B_beta,
        B_omega,
        B_r.dval, 
        B_v.dval,
        B_p.dval,
        queue );
        
        //QMR
        CHECK( magma_d_spmv( c_one, A, Q_p, c_zero, Q_pt, queue ));
        //TFQMR
        CHECK( magma_d_spmv( c_one, A, T_pu_m, c_zero, T_Au_new, queue ));
        //CGS
        CHECK( magma_d_spmv( c_one, A, C_p, c_zero, C_v_hat, queue ));
        // BiCGSTAB
        CHECK( magma_d_spmv( c_one, A, B_p, c_zero, B_v, queue ));      // v = Ap
        
        solver_par->spmv_count++;
        
            //QMR: epsilon = q' * pt;
        Q_epsilon = magma_ddot( dofs, Q_q.dval, 1, Q_pt.dval, 1, queue );
        Q_beta = Q_epsilon / Q_delta;
            //TFQMR
        magma_dcopy( dofs, T_Au_new.dval, 1, T_Au.dval, 1, queue );  
        magma_dcopy( dofs, T_u_mp1.dval, 1, T_u_m.dval, 1, queue ); 
            //CGS: alpha = r_tld' * v_hat
        C_alpha = C_rho / magma_ddot( dofs, r_tld.dval, 1, C_v_hat.dval, 1, queue );
            //BiCGSTAB
        B_alpha = B_rho_new / magma_ddot( dofs, r_tld.dval, 1, B_v.dval, 1, queue );

        
            //QMR: v = pt - beta * v
            //QMR: y = v
        magma_dqmr_3(  
        b.num_rows, 
        b.num_cols, 
        Q_beta,
        Q_pt.dval,
        Q_v.dval,
        Q_y.dval,
        queue );
        
        // TFQMR
        magma_dtfqmr_5(  
        b.num_rows, 
        b.num_cols, 
        T_alpha,
        T_sigma,
        T_v.dval, 
        T_Au.dval,
        T_pu_m.dval,
        T_w.dval, 
        T_d.dval,
        T_Ad.dval,
        queue ); 
        
                // TFQMR
        T_sigma = T_theta * T_theta / T_alpha * T_eta;  
        
        T_theta = magma_dsqrt( magma_ddot(dofs, T_w.dval, 1, T_w.dval, 1, queue) ) / T_tau;
        T_c = c_one / magma_dsqrt( c_one + T_theta*T_theta );
        T_tau = T_tau * T_theta *T_c;
        T_eta = T_c * T_c * T_alpha;
        
        // TFQMR
        magma_dtfqmr_2(  
        b.num_rows, 
        b.num_cols, 
        T_eta,
        T_d.dval,
        T_Ad.dval,
        T_x.dval, 
        T_r.dval, 
        queue );
        
        T_rho = magma_ddot( dofs, T_w.dval, 1, r_tld.dval, 1, queue );
        T_beta = T_rho / T_rho_l;
        T_rho_l = T_rho;
        
        magma_dtfqmr_3(  
        b.num_rows, 
        b.num_cols, 
        T_beta,
        T_w.dval,
        T_u_m.dval,
        T_u_mp1.dval, 
        queue );
        magma_dcopy( dofs, T_u_mp1.dval, 1, T_pu_m.dval, 1, queue );  
        
        
            //CGS: q = u - alpha v_hat
            //CGS: t = u + q
        magma_dcgs_3(  
        b.num_rows, 
        b.num_cols, 
        C_alpha,
        C_v_hat.dval,
        C_u.dval, 
        C_q.dval,
        C_t.dval, 
        queue );
        
            // BiCGSTAB: s = r - alpha v
        magma_dbicgstab_2(  
        b.num_rows, 
        b.num_cols, 
        B_alpha,
        B_r.dval,
        B_v.dval,
        B_s.dval, 
        queue );
            
        
        Q_rho1 = Q_rho;      
            //QMR rho = norm(y);
        Q_rho = magma_dsqrt( magma_ddot( dofs, Q_y.dval, 1, Q_y.dval, 1, queue ) );
        
            //QMR wt = A' * q - beta' * w;
        CHECK( magma_d_spmv( c_one, AT, Q_q, c_zero, Q_wt, queue ));
        //TFQMR
        CHECK( magma_d_spmv( c_one, A, T_pu_m, c_zero, T_Au_new, queue ));
            //CGS t = A u_hat
        CHECK( magma_d_spmv( c_one, A, C_t, c_zero, C_rt, queue )); 
            //BiCGSTAB
        CHECK( magma_d_spmv( c_one, A, B_s, c_zero, B_t, queue ));       // t=As
        
        solver_par->spmv_count++;
        
        //BiCGSTAB
        B_omega = magma_ddot( dofs, B_t.dval, 1, B_s.dval, 1, queue )   // omega = <s,t>/<t,t>
                   / magma_ddot( dofs, B_t.dval, 1, B_t.dval, 1, queue );

                   
       // QMR
        magma_daxpy( dofs, - MAGMA_D_CONJ( Q_beta ), Q_w.dval, 1, Q_wt.dval, 1, queue );  
                    // no precond: z = wt
        magma_dcopy( dofs, Q_wt.dval, 1, Q_z.dval, 1, queue );
        
        
        //TFQMR
        magma_dtfqmr_4(  
        b.num_rows, 
        b.num_cols, 
        T_beta,
        T_Au_new.dval,
        T_v.dval,
        T_Au.dval, 
        queue );
        
        magma_dcopy( dofs, T_u_mp1.dval, 1, T_u_m.dval, 1, queue ); 
        
            
        // QMR
        Q_thet1 = Q_thet;        
        Q_thet = Q_rho / (Q_gamm * MAGMA_D_MAKE( MAGMA_D_ABS(Q_beta), 0.0 ));
        Q_gamm1 = Q_gamm;        
        
        Q_gamm = c_one / magma_dsqrt(c_one + Q_thet*Q_thet);        
        Q_eta = - Q_eta * Q_rho1 * Q_gamm * Q_gamm / (Q_beta * Q_gamm1 * Q_gamm1);

        if( solver_par->numiter == 1 ){
            
                //QMR: d = eta * p + pds * d;
                //QMR: s = eta * pt + pds * d;
                //QMR: x = x + d;
                //QMR: r = r - s;
            magma_dqmr_4(  
            b.num_rows, 
            b.num_cols, 
            Q_eta,
            Q_p.dval,
            Q_pt.dval,
            Q_d.dval, 
            Q_s.dval, 
            Q_x.dval, 
            Q_r.dval, 
            queue );
        }
        else{

            Q_pds = (Q_thet1 * Q_gamm) * (Q_thet1 * Q_gamm);
            
                // d = eta * p + pds * d;
                // s = eta * pt + pds * d;
                // x = x + d;
                // r = r - s;
            magma_dqmr_5(  
            b.num_rows, 
            b.num_cols, 
            Q_eta,
            Q_pds,
            Q_p.dval,
            Q_pt.dval,
            Q_d.dval, 
            Q_s.dval, 
            Q_x.dval, 
            Q_r.dval, 
            queue );
        }
        
        
        // CGS: r = r -alpha*A u_hat
        // CGS: x = x + alpha u_hat
        magma_dcgs_4(  
        b.num_rows, 
        b.num_cols, 
        C_alpha,
        C_t.dval,
        C_rt.dval,
        C_x.dval, 
        C_r.dval,
        queue );
        C_rho_l = C_rho;  
        
            // BiCGSTAB: x = x + alpha * p + omega * s
            // BiCGSTAB: r = s - omega * t
        magma_dbicgstab_3(  
        b.num_rows, 
        b.num_cols, 
        B_alpha,
        B_omega,
        B_p.dval,
        B_s.dval,
        B_t.dval,
        B_x.dval,
        B_r.dval,
        queue );
        
            //QMR: psi = norm(z);
        Q_psi = magma_dsqrt( magma_ddot( dofs, Q_z.dval, 1, Q_z.dval, 1, queue ) );
        
            //QMR: v = y / rho
            //QMR: y = y / rho
            //QMR: w = wt / psi
            //QMR: z = z / psi
        magma_dqmr_1(  
        b.num_rows, 
        b.num_cols, 
        Q_rho,
        Q_psi,
        Q_y.dval, 
        Q_z.dval,
        Q_v.dval,
        Q_w.dval,
        queue );
        
        
        
        
        Q_res = magma_dnrm2( dofs, Q_r.dval, 1, queue );
        T_res = magma_dnrm2( dofs, T_r.dval, 1, queue );
        C_res = magma_dnrm2( dofs, C_r.dval, 1, queue );
        B_res = magma_dnrm2( dofs, B_r.dval, 1, queue );

        
            // printf(" %e   %e   %e\n", Q_res, C_res, B_res);
        if( Q_res < res ){
            res = Q_res;
            flag = 1;
        }
        if( T_res < res ){
            res = Q_res;
            flag = 2;
        }
        if( C_res < res ){
            res = C_res;
            flag = 3;
        }
        if( B_res < res ){
            res = B_res;
            flag = 4;
        }

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            info = MAGMA_SUCCESS;
            break;
        }
        if( magma_d_isnan_inf( Q_beta ) && magma_d_isnan_inf( C_beta ) && magma_d_isnan_inf( B_beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        } 
                

 
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
        
    // copy back the best solver
    switch ( flag ) {
        case 1:
            printf("%% QMR fastest solver.\n");
            magma_dcopy( dofs, Q_x.dval, 1, x->dval, 1, queue ); 
            break;
       case 2:
            printf("%% TFQMR fastest solver.\n");
            magma_dcopy( dofs, T_x.dval, 1, x->dval, 1, queue ); 
            break;
       case 3:
            printf("%% CGS fastest solver.\n");
            magma_dcopy( dofs, C_x.dval, 1, x->dval, 1, queue ); 
            break;
       case 4:
            printf("%% BiCGSTAB fastest solver.\n");
            magma_dcopy( dofs, B_x.dval, 1, x->dval, 1, queue ); 
            break;
    }


    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_dresidualvec( A, b, *x, &r_tld, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter  && info == MAGMA_SUCCESS ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->rtol*solver_par->init_res ||
            solver_par->iter_res < solver_par->atol ) {
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_dmfree(&r_tld, queue );
    magma_dmfree(&d1, queue );
    magma_dmfree(&d2, queue );
    magma_dmfree(&AT,  queue );
    
    // QMR
    magma_dmfree(&Q_r,  queue );
    magma_dmfree(&Q_v,  queue );
    magma_dmfree(&Q_w,  queue );
    magma_dmfree(&Q_wt, queue );
    magma_dmfree(&Q_d,  queue );
    magma_dmfree(&Q_s,  queue );
    magma_dmfree(&Q_z,  queue );
    magma_dmfree(&Q_q,  queue );
    magma_dmfree(&Q_p,  queue );
    magma_dmfree(&Q_pt, queue );
    magma_dmfree(&Q_y,  queue );
    magma_dmfree(&Q_x,  queue );
    magma_dmfree(&Ah1, queue );
    magma_dmfree(&Ah2, queue );
    // TFQMR
    magma_dmfree(&T_r, queue );
    magma_dmfree(&T_x,  queue );
    magma_dmfree(&T_d, queue );
    magma_dmfree(&T_w, queue );
    magma_dmfree(&T_v, queue );
    magma_dmfree(&T_u_m, queue );
    magma_dmfree(&T_u_mp1, queue );
    magma_dmfree(&T_pu_m, queue );
    magma_dmfree(&T_d, queue );
    magma_dmfree(&T_Au, queue );
    magma_dmfree(&T_Au_new, queue );
    magma_dmfree(&T_Ad, queue );
    // CGS
    magma_dmfree(&C_r,  queue );
    magma_dmfree(&C_rt, queue );
    magma_dmfree(&C_x,  queue );
    magma_dmfree(&C_p,  queue );
    magma_dmfree(&C_q,  queue );
    magma_dmfree(&C_u,  queue );
    magma_dmfree(&C_v,  queue );
    magma_dmfree(&C_t,  queue );
    magma_dmfree(&C_p_hat, queue );
    magma_dmfree(&C_q_hat, queue );
    magma_dmfree(&C_u_hat, queue );
    magma_dmfree(&C_v_hat, queue );
    // BiCGSTAB
    magma_dmfree(&B_r, queue );
    magma_dmfree(&B_x, queue );
    magma_dmfree(&B_p, queue );
    magma_dmfree(&B_v, queue );
    magma_dmfree(&B_s, queue );
    magma_dmfree(&B_t, queue );
    
    solver_par->info = info;
    return info;
}   /* magma_dbombard */
