/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Stan Tomov
       @author Hartwig Anzt

       @generated from zpgmres.cpp normal z -> c, Fri Jul 18 17:34:29 2014
*/
#include <sys/time.h>
#include <time.h>

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <cblas.h>

#define PRECISION_c

#define  q(i)     (q.val + (i)*dofs)
#define  z(i)     (z.val + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*(1+ldh)]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j) dH[(i)   + (j)*(1+ldh)]


#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory. 
    This is a GPU implementation of the right-preconditioned GMRES method.

    Arguments
    ---------

    @param
    A           magma_c_sparse_matrix
                descriptor for matrix A

    @param
    b           magma_c_vector
                RHS b vector

    @param
    x           magma_c_vector*
                solution approximation

    @param
    solver_par  magma_c_solver_par*
                solver parameters

    @param
    precond_par magma_c_preconditioner*
                preconditioner

    @ingroup magmasparse_cgesv
    ********************************************************************/

magma_int_t
magma_cpgmres( magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector *x,  
               magma_c_solver_par *solver_par, 
               magma_c_preconditioner *precond_par ){

    // prepare solver feedback
    solver_par->solver = Magma_PGMRES;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // local variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO, c_one = MAGMA_C_ONE, 
                                                c_mone = MAGMA_C_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m = 0;
    magma_int_t restart = min( dofs-1, solver_par->restart );
    magma_int_t ldh = restart+1;
    float nom, rNorm, RNorm, nom0, betanom, r0 = 0.;

    // CPU workspace
    magma_setdevice(0);
    magmaFloatComplex *H, *HH, *y, *h1;
    magma_cmalloc_pinned( &H, (ldh+1)*ldh );
    magma_cmalloc_pinned( &y, ldh );
    magma_cmalloc_pinned( &HH, ldh*ldh );
    magma_cmalloc_pinned( &h1, ldh );

    // GPU workspace
    magma_c_vector r, q, q_t, z, z_t, t;
    magma_c_vinit( &t, Magma_DEV, dofs, c_zero );
    magma_c_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_c_vinit( &q, Magma_DEV, dofs*(ldh+1), c_zero );
    magma_c_vinit( &z, Magma_DEV, dofs*(ldh+1), c_zero );
    magma_c_vinit( &z_t, Magma_DEV, dofs, c_zero );
    q_t.memory_location = Magma_DEV; 
    q_t.val = NULL; 
    q_t.num_rows = q_t.nnz = dofs;

    magmaFloatComplex *dy, *dH = NULL;
    if (MAGMA_SUCCESS != magma_cmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    if (MAGMA_SUCCESS != magma_cmalloc( &dH, (ldh+1)*ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;

    // GPU stream
    magma_queue_t stream[2];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );
    magmablasSetKernelStream(stream[0]);

    magma_cscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_ccopy( dofs, b.val, 1, r.val, 1 );             //  r = b
    nom0 = betanom = magma_scnrm2( dofs, r.val, 1 );     //  nom0= || r||
    nom = nom0  * nom0;
    solver_par->init_res = nom0;
    H(1,0) = MAGMA_C_MAKE( nom0, 0. ); 
    magma_csetvector(1, &H(1,0), 1, &dH(1,0), 1);
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;

    //Chronometry
    real_Double_t tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    if( solver_par->verbose > 0 ){
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){
        magma_ccopy(dofs, r.val, 1, q(0), 1);       //  q[0] = 1.0/H(1,0) r
        magma_cscal(dofs, 1./H(1,0), q(0), 1);      //  (to be fused)

        for(k=1; k<=restart; k++) {
            q_t.val = q(k-1);
            magmablasSetKernelStream(stream[0]);
            // preconditioner
            //  z[k] = M^(-1) q(k)
            magma_c_applyprecond_left( A, q_t, &t, precond_par );      
            magma_c_applyprecond_right( A, t, &z_t, precond_par );     
  
            magma_ccopy(dofs, z_t.val, 1, z(k-1), 1);                  

            // r = A q[k] 
            magma_c_spmv( c_one, A, z_t, c_zero, r );


            if (solver_par->ortho == Magma_MGS ) {
                // modified Gram-Schmidt
                magmablasSetKernelStream(stream[0]);
                for (i=1; i<=k; i++) {
                    H(i,k) =magma_cdotc(dofs, q(i-1), 1, r.val, 1);            
                        //  H(i,k) = q[i] . r
                    magma_caxpy(dofs,-H(i,k), q(i-1), 1, r.val, 1);            
                       //  r = r - H(i,k) q[i]
                }
                H(k+1,k) = MAGMA_C_MAKE( magma_scnrm2(dofs, r.val, 1), 0. );
                      //  H(k+1,k) = sqrt(r . r) 
                if (k < restart) {
                        magma_ccopy(dofs, r.val, 1, q(k), 1);                  
                      //  q[k] = 1.0/H[k][k-1] r
                        magma_cscal(dofs, 1./H(k+1,k), q(k), 1);               
                      //  (to be fused)   
                 }
            } else if (solver_par->ortho == Magma_FUSED_CGS ) {
                // fusing cgemv with scnrm2 in classical Gram-Schmidt
                magmablasSetKernelStream(stream[0]);
                magma_ccopy(dofs, r.val, 1, q(k), 1);  
                    // dH(1:k+1,k) = q[0:k] . r
                magmablas_cgemv(MagmaTrans, dofs, k+1, c_one, q(0), 
                                dofs, r.val, 1, c_zero, &dH(1,k), 1);
                    // r = r - q[0:k-1] dH(1:k,k)
                magmablas_cgemv(MagmaNoTrans, dofs, k, c_mone, q(0), 
                                dofs, &dH(1,k), 1, c_one, r.val, 1);
                   // 1) dH(k+1,k) = sqrt( dH(k+1,k) - dH(1:k,k) )
                magma_ccopyscale(  dofs, k, r.val, q(k), &dH(1,k) );  
                   // 2) q[k] = q[k] / dH(k+1,k) 

                magma_event_record( event[0], stream[0] );
                magma_queue_wait_event( stream[1], event[0] );
                magma_cgetvector_async(k+1, &dH(1,k), 1, &H(1,k), 1, stream[1]); 
                    // asynch copy dH(1:(k+1),k) to H(1:(k+1),k)
            } else {
                // classical Gram-Schmidt (default)
                // > explicitly calling magmabls
                magmablasSetKernelStream(stream[0]);                                                  
                magmablas_cgemv(MagmaTrans, dofs, k, c_one, q(0), 
                                dofs, r.val, 1, c_zero, &dH(1,k), 1); 
                                // dH(1:k,k) = q[0:k-1] . r
                #ifndef SCNRM2SCALE 
                // start copying dH(1:k,k) to H(1:k,k)
                magma_event_record( event[0], stream[0] );
                magma_queue_wait_event( stream[1], event[0] );
                magma_cgetvector_async(k, &dH(1,k), 1, &H(1,k), 
                                                    1, stream[1]);
                #endif
                                  // r = r - q[0:k-1] dH(1:k,k)
                magmablas_cgemv(MagmaNoTrans, dofs, k, c_mone, q(0), 
                                    dofs, &dH(1,k), 1, c_one, r.val, 1);
                #ifdef SCNRM2SCALE
                magma_ccopy(dofs, r.val, 1, q(k), 1);                 
                    //  q[k] = r / H(k,k-1) 
                magma_scnrm2scale(dofs, q(k), dofs, &dH(k+1,k) );     
                    //  dH(k+1,k) = sqrt(r . r) and r = r / dH(k+1,k)

                magma_event_record( event[0], stream[0] );            
                            // start sending dH(1:k,k) to H(1:k,k)
                magma_queue_wait_event( stream[1], event[0] );        
                            // can we keep H(k+1,k) on GPU and combine?
                magma_cgetvector_async(k+1, &dH(1,k), 1, &H(1,k), 1, stream[1]);
                #else
                H(k+1,k) = MAGMA_C_MAKE( magma_scnrm2(dofs, r.val, 1), 0. );   
                            //  H(k+1,k) = sqrt(r . r) 
                if( k<solver_par->restart ){
                        magmablasSetKernelStream(stream[0]);
                        magma_ccopy(dofs, r.val, 1, q(k), 1);                  
                            //  q[k]    = 1.0/H[k][k-1] r
                        magma_cscal(dofs, 1./H(k+1,k), q(k), 1);              
                            //  (to be fused)   
                 }
                #endif
            }
        }
        magma_queue_sync( stream[1] );
        for( k=1; k<=restart; k++ ){
            /*     Minimization of  || b-Ax ||  in H_k       */ 
            for (i=1; i<=k; i++) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                cblas_cdotc_sub( i+1, &H(1,k), 1, &H(1,i), 1, &HH(k,i) );
                #else
                HH(k,i) = cblas_cdotc(i+1, &H(1,k), 1, &H(1,i), 1);
                #endif
            }
            h1[k] = H(1,k)*H(1,0); 
            if (k != 1)
                for (i=1; i<k; i++) {
                    for (m=i+1; m<k; m++){
                        HH(k,m) -= HH(k,i) * HH(m,i);
                    }
                    HH(k,k) -= HH(k,i) * HH(k,i) / HH(i,i);
                    HH(k,i) = HH(k,i)/HH(i,i);
                    h1[k] -= h1[i] * HH(k,i);   
                }    
            y[k] = h1[k]/HH(k,k); 
            if (k != 1)  
                for (i=k-1; i>=1; i--) {
                    y[i] = h1[i]/HH(i,i);
                    for (j=i+1; j<=k; j++)
                        y[i] -= y[j] * HH(j,i);
                }                    
            m = k;
            rNorm = fabs(MAGMA_C_REAL(H(k+1,k)));
        }

        magma_csetmatrix_async(m, 1, y+1, m, dy, m, stream[0]);
        magmablasSetKernelStream(stream[0]);
        magma_cgemv(MagmaNoTrans, dofs, m, c_one, z(0), dofs, dy, 1, 
                                                    c_one, x->val, 1); 
        magma_c_spmv( c_mone, A, *x, c_zero, r );      //  r = - A * x
        magma_caxpy(dofs, c_one, b.val, 1, r.val, 1);  //  r = r + b
        H(1,0) = MAGMA_C_MAKE( magma_scnrm2(dofs, r.val, 1), 0. ); 
                                            //  RNorm = H[1][0] = || r ||
        RNorm = MAGMA_C_REAL( H(1,0) );
        betanom = fabs(RNorm);  

        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if (  betanom  < r0 ) {
            break;
        } 
    }

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    float residual;
    magma_cresidual( A, b, *x, &residual );
    solver_par->iter_res = betanom;
    solver_par->final_res = residual;

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res > solver_par->final_res ){
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -2;
    }
    else{
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -1;
    }
    // free pinned memory
    magma_free_pinned( H );
    magma_free_pinned( y );
    magma_free_pinned( HH );
    magma_free_pinned( h1 );
    // free GPU memory
    magma_free(dy); 
    if (dH != NULL ) magma_free(dH); 
    magma_c_vfree(&t);
    magma_c_vfree(&r);
    magma_c_vfree(&q);
    magma_c_vfree(&z);
    magma_c_vfree(&z_t);

    // free GPU streams and events
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_event_destroy( event[0] );
    magmablasSetKernelStream(NULL);

    return MAGMA_SUCCESS;
}   /* magma_cgmres */

