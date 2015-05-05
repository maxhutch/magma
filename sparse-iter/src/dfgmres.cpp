/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @author Hartwig Anzt

       @generated from zfgmres.cpp normal z -> d, Sun May  3 11:22:59 2015
*/
#include "common_magmasparse.h"

#define PRECISION_d

/*
#define  q(i)     (q.dval + (i)*dofs)
#define  z(i)     (z.dval + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*(1+ldh)]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j) dH[(i)   + (j)*(1+ldh)]
*/

// simulate 2-D arrays at the cost of some arithmetic
#define V(i) (V.dval+(i)*dofs)
#define W(i) (W.dval+(i)*dofs)
//#define Vv(i) (&V.dval[(i)*n])
//#define Wv(i) (&W.dval[(i)*n])
#define H(i,j) (H[(j)*m1+(i)])
#define ABS(x)   ((x)<0 ? (-(x)) : (x))




#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


static void
GeneratePlaneRotation(double dx, double dy, double *cs, double *sn)
{
    
    if (dy == MAGMA_D_ZERO) {
        *cs = MAGMA_D_ONE;
        *sn = MAGMA_D_ZERO;
    } else if (MAGMA_D_ABS((dy)) > MAGMA_D_ABS((dx))) {
        double temp = dx / dy;
        *sn = MAGMA_D_ONE / magma_dsqrt( ( MAGMA_D_ONE + temp*temp)) ;
        *cs = temp * *sn;
    } else {
        double temp = dy / dx;
        *cs = MAGMA_D_ONE / magma_dsqrt( ( MAGMA_D_ONE + temp*temp )) ;
        *sn = temp * *cs;
    }

  //  real_Double_t rho = sqrt(MAGMA_D_REAL(MAGMA_D_CNJG(dx)*dx + MAGMA_D_CNJG(dy)*dy));
  //  *cs = dx / rho;
  //  *sn = dy / rho;
}

static void ApplyPlaneRotation(double *dx, double *dy, double cs, double sn)
{

    double temp = *dx;
    *dx =  cs * *dx + sn * *dy;
    *dy = -sn * temp + cs * *dy;

 //   double temp  =  MAGMA_D_CNJG(cs) * *dx +  MAGMA_D_CNJG(sn) * *dy;
 //   *dy = -(sn) * *dx + cs * *dy;
 //   *dx = temp;
}



/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real sparse matrix stored in the GPU memory.
    X and B are real vectors stored on the GPU memory.
    This is a GPU implementation of the right-preconditioned flexible GMRES.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                descriptor for matrix A

    @param[in]
    b           magma_d_matrix
                RHS b vector

    @param[in,out]
    x           magma_d_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_d_solver_par*
                solver parameters

    @param[in]
    precond_par magma_d_preconditioner*
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgesv
    ********************************************************************/

extern "C" magma_int_t
magma_dfgmres(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t dofs = A.num_rows;

    // prepare solver feedback
    solver_par->solver = Magma_PGMRES;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;
    
    //Chronometry
    real_Double_t tempo1, tempo2;

    magma_int_t dim = solver_par->restart;
    magma_int_t m1 = dim+1; // used inside H macro
    magma_int_t i, j, k;
    double beta;
    
    double rel_resid, resid0, r0=0.0, betanom = 0.0, nom;
    
    magma_d_matrix v_t={Magma_CSR}, w_t={Magma_CSR}, t={Magma_CSR}, t2={Magma_CSR}, V={Magma_CSR}, W={Magma_CSR};
    v_t.memory_location = Magma_DEV;
    v_t.num_rows = dofs;
    v_t.num_cols = 1;
    v_t.dval = NULL;
    v_t.storage_type = Magma_DENSE;

    w_t.memory_location = Magma_DEV;
    w_t.num_rows = dofs;
    w_t.num_cols = 1;
    w_t.dval = NULL;
    w_t.storage_type = Magma_DENSE;
    
    double temp;
    
    double *H={0}, *s={0}, *cs={0}, *sn={0};

    CHECK( magma_dvinit( &t, Magma_DEV, dofs, 1, MAGMA_D_ZERO, queue ));
    CHECK( magma_dvinit( &t2, Magma_DEV, dofs, 1, MAGMA_D_ZERO, queue ));
    
    CHECK( magma_dmalloc_pinned( &H, (dim+1)*dim ));
    CHECK( magma_dmalloc_pinned( &s,  dim+1 ));
    CHECK( magma_dmalloc_pinned( &cs, dim ));
    CHECK( magma_dmalloc_pinned( &sn, dim ));
    
    
    CHECK( magma_dvinit( &V, Magma_DEV, dofs*(dim+1), 1, MAGMA_D_ZERO, queue ));
    CHECK( magma_dvinit( &W, Magma_DEV, dofs*dim, 1, MAGMA_D_ZERO, queue ));
    
    CHECK(  magma_dresidual( A, b, *x, &nom, queue));

    solver_par->init_res = nom;
    
    if ( ( nom * solver_par->epsilon) < ATOLERANCE )
        r0 = ATOLERANCE;
    
    solver_par->numiter = 0;
    

    tempo1 = magma_sync_wtime( queue );
    do
    {
        // compute initial residual and its norm
        // A.mult(n, 1, x, n, V(0), n);                        // V(0) = A*x
        CHECK( magma_d_spmv( MAGMA_D_ONE, A, *x, MAGMA_D_ZERO, t, queue ));
        magma_dcopy( dofs, t.dval, 1, V(0), 1 );
        
        temp = MAGMA_D_MAKE(-1.0, 0.0);
        magma_daxpy(dofs,temp, b.dval, 1, V(0), 1);           // V(0) = V(0) - b
        beta = MAGMA_D_MAKE( magma_dnrm2( dofs, V(0), 1 ), 0.0); // beta = norm(V(0))
        if (solver_par->numiter == 0){
            solver_par->init_res = MAGMA_D_REAL( beta );
            resid0 = MAGMA_D_REAL( beta );
        
            if ( (r0 = resid0 * solver_par->epsilon) < ATOLERANCE )
                r0 = ATOLERANCE;
            if ( resid0 < r0 ) {
                solver_par->final_res = solver_par->init_res;
                solver_par->iter_res = solver_par->init_res;
                goto cleanup;
            }
        }
        if ( solver_par->verbose > 0 ) {
            solver_par->res_vec[0] = resid0;
            solver_par->timing[0] = 0.0;
        }
        temp = -1.0/beta;
        magma_dscal( dofs, temp, V(0), 1 );                 // V(0) = -V(0)/beta

        // save very first residual norm
        if (solver_par->numiter == 0)
            solver_par->init_res = MAGMA_D_REAL( beta );

        for (i = 1; i < dim+1; i++)
            s[i] = MAGMA_D_ZERO;
        s[0] = beta;

        i = -1;
        do
        {
            solver_par->numiter++;
            i++;
            
            // M.apply(n, 1, V(i), n, W(i), n);
            v_t.dval = V(i);
            CHECK( magma_d_applyprecond_left( A, v_t, &t, precond_par, queue ));
            CHECK( magma_d_applyprecond_right( A, t, &t2, precond_par, queue ));
            magma_dcopy( dofs, t2.dval, 1, W(i), 1 );

            // A.mult(n, 1, W(i), n, V(i+1), n);
            w_t.dval = W(i);
            CHECK( magma_d_spmv( MAGMA_D_ONE, A, w_t, MAGMA_D_ZERO, t, queue ));
            magma_dcopy( dofs, t.dval, 1, V(i+1), 1 );
            
            for (k = 0; k <= i; k++)
            {
                H(k, i) = magma_ddot(dofs, V(k), 1, V(i+1), 1);
                temp = -H(k,i);
                // V(i+1) -= H(k, i) * V(k);
                magma_daxpy(dofs,-H(k,i), V(k), 1, V(i+1), 1);
            }

            H(i+1, i) = MAGMA_D_MAKE( magma_dnrm2(dofs, V(i+1), 1), 0. ); // H(i+1,i) = ||r||
            temp = 1.0 / H(i+1, i);
            // V(i+1) = V(i+1) / H(i+1, i)
            magma_dscal(dofs, temp, V(i+1), 1);    //  (to be fused)
    
            for (k = 0; k < i; k++)
                ApplyPlaneRotation(&H(k,i), &H(k+1,i), cs[k], sn[k]);
          
            GeneratePlaneRotation(H(i,i), H(i+1,i), &cs[i], &sn[i]);
            ApplyPlaneRotation(&H(i,i), &H(i+1,i), cs[i], sn[i]);
            ApplyPlaneRotation(&s[i], &s[i+1], cs[i], sn[i]);
            
            betanom = MAGMA_D_ABS( s[i+1] );
            rel_resid = betanom / resid0;
            if ( solver_par->verbose > 0 ) {
                tempo2 = magma_sync_wtime( queue );
                if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                    solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) betanom;
                    solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) tempo2-tempo1;
                }
            }
            if (rel_resid <= solver_par->epsilon){
                break;
            }
        }
        while (i+1 < dim && solver_par->numiter+1 <= solver_par->maxiter);

        // solve upper triangular system in place
        for (j = i; j >= 0; j--)
        {
            s[j] /= H(j,j);
            for (k = j-1; k >= 0; k--)
                s[k] -= H(k,j) * s[j];
        }

        // update the solution
        for (j = 0; j <= i; j++)
        {
            // x = x + s[j] * W(j)
            magma_daxpy(dofs, s[j], W(j), 1, x->dval, 1);
        }
    }
    while (rel_resid > solver_par->epsilon
                && solver_par->numiter+1 <= solver_par->maxiter);

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK( magma_dresidual( A, b, *x, &residual, queue ));
    solver_par->iter_res = betanom;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        solver_par->info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->epsilon*solver_par->init_res ){
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    // free pinned memory
    magma_free_pinned(s);
    magma_free_pinned(cs);
    magma_free_pinned(sn);
    magma_free_pinned(H);

    //free DEV memory
    magma_dmfree( &V, queue);
    magma_dmfree( &W, queue);
    magma_dmfree( &t, queue);
    magma_dmfree( &t2, queue);

    solver_par->info = info;
    return info;

} /* magma_dfgmres */




