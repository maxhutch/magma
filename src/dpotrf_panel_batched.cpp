/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotrf_panel_batched.cpp normal z -> d, Mon May  2 23:30:27 2016
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
////////////////////////////////////////////////////////////////////////////////////////
/**
    \n
    This is an internal routine.
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ, 
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    //===============================================
    //  panel factorization
    //===============================================
    if (n < nb) {
        printf("magma_dpotrf_panel error n < nb %d < %d \n",(int) n, (int) nb);
        return -101;
    }

#if 0
    arginfo = magma_dpotf2_batched(
                       uplo, n, nb,
                       dA_array, ldda,
                       dW1_displ, dW2_displ,
                       dW3_displ, dW4_displ,
                       info_array, gbstep,
                       batchCount, queue);
#else
    arginfo = magma_dpotf2_batched(
                       uplo, nb, nb,
                       dA_array, ldda,
                       dW1_displ, dW2_displ,
                       dW3_displ, dW4_displ,
                       info_array, gbstep,
                       batchCount, queue);

    if ((n-nb) > 0) {
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, nb, 0, batchCount, queue);
        magmablas_dtrsm_work_batched( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                              1, n-nb, nb, 
                              MAGMA_D_ONE,
                              dA_array,    ldda, 
                              dW0_displ,   ldda, 
                              dX_array,    n-nb, 
                              dinvA_array, dinvA_length, 
                              dW1_displ,   dW2_displ, 
                              dW3_displ,   dW4_displ,
                              0, batchCount, queue );
    }
#endif
    return arginfo;
}


////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////
/**
    \n
    This is an internal routine.
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ,  
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }
    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
        arginfo = -1;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }
    if (m < n) {
        printf("error m < n %d < %d \n", (int) m, (int) n);
        arginfo = -101;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    double **dA_displ  = NULL;
    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));

    double alpha = MAGMA_D_NEG_ONE;
    double beta  = MAGMA_D_ONE;
    magma_int_t panel_nb = n;
    if (panel_nb <= min_recpnb) {
        //printf("calling bottom panel recursive with m=%d nb=%d\n",m,n);
        //  panel factorization
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, 0, 0, batchCount, queue);
        //magma_dpotrf_rectile_batched(uplo, m, panel_nb, 16,
        arginfo = magma_dpotrf_panel_batched(uplo, m, panel_nb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ,
                           info_array, gbstep,
                           batchCount, queue);
    }
    else {
        // split A over two [A A2]
        // panel on A1, update on A2 then panel on A1    
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;
        magma_int_t m1 = m;
        magma_int_t m2 = m-n1;
        magma_int_t p1 = 0;
        magma_int_t p2 = n1;
        // panel on A1
        //printf("calling recursive panel on A1 with m=%d nb=%d min_recpnb %d\n",m1,n1,min_recpnb);
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, p1, p1, batchCount, queue);        
        arginfo = magma_dpotrf_recpanel_batched(
                           uplo, m1, n1, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ, 
                           info_array, gbstep,
                           batchCount, queue);
        if (arginfo != 0) {
            magma_free(dA_displ);
            return arginfo;
        }

        // update A2
        //printf("calling update A2 with             m=%d n=%d k=%d\n",m2,n2,n1);
        magma_ddisplace_pointers(dA_displ,  dA_array, ldda, p1+n1, p1, batchCount, queue);        
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, p1+n1, p2, batchCount, queue);        
        magma_dgemm_batched( MagmaNoTrans, MagmaConjTrans, m2, n2, n1,
                             alpha, dA_displ, ldda, 
                             dA_displ, ldda, 
                             beta,  dW0_displ, ldda, 
                             batchCount, queue );
        // panel on A2
        //printf("calling recursive panel on A2 with m=%d nb=%d min_recpnb %d\n",m2,n2,min_recpnb);
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, p2, p2, batchCount, queue);        
        arginfo = magma_dpotrf_recpanel_batched(
                                      uplo, m2, n2, min_recpnb,
                                      dA_displ, ldda,
                                      dX_array, dX_length,
                                      dinvA_array, dinvA_length,
                                      dW0_displ, dW1_displ, dW2_displ,
                                      dW3_displ, dW4_displ,
                                      info_array, gbstep,
                                      batchCount, queue);
    }

    magma_free(dA_displ);
    return arginfo;
}


////////////////////////////////////////////////////////////////////////////////////////
/**
    \n
    This is an internal routine.
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    double** dA_array,    magma_int_t ldda,
    double** dX_array,    magma_int_t dX_length,
    double** dinvA_array, magma_int_t dinvA_length,
    double** dW0_displ, double** dW1_displ,  
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    //magma_int_t DEBUG=0;

    // Quick return if possible
    if (m == 0 || n == 0) {
        return 1;
    }
    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
        return -100;
    }
    if (m < n) {
        printf("error m < n %d < %d \n", (int) m, (int) n);
        return -101;
    }

    double **dA_displ  = NULL;
    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));

    double alpha = MAGMA_D_NEG_ONE;
    double beta  = MAGMA_D_ONE;
    magma_int_t panel_nb = n;
    if (panel_nb <= min_recpnb) {
        // if (DEBUG == 1) printf("calling bottom panel recursive with n=%d\n",(int) panel_nb);
        //  panel factorization
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, 0, 0, batchCount, queue);
        magma_dpotrf_panel_batched(
                           uplo, m, panel_nb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ,
                           info_array, gbstep,
                           batchCount, queue);
    }
    else {
        // split A over two [A11 A12;  A21 A22; A31 A32]
        // panel on tile A11, 
        // trsm on A21, using A11
        // update on A22 then panel on A22.  
        // finally a trsm on [A31 A32] using the whole [A11 A12; A21 A22]     
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;
        magma_int_t p1 = 0;
        magma_int_t p2 = n1;

        // panel on A11
        //if (DEBUG == 1) printf("calling recursive panel on A11=A(%d,%d) with n=%d min_recpnb %d\n",(int) p1, (int) p1, (int) n1, (int) min_recpnb);
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, p1, p1, batchCount, queue);        
        magma_dpotrf_rectile_batched(
                           uplo, n1, n1, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ, 
                           info_array, gbstep,
                           batchCount, queue);

        // TRSM on A21
        //if (DEBUG == 1) printf("calling trsm on A21=A(%d,%d) using A11 == A(%d,%d) with m=%d k=%d \n",p2,p1,p1,p1,n2,n1);
        magma_ddisplace_pointers(dA_displ,  dA_array, ldda, p1, p1, batchCount, queue);        
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, p2, p1, batchCount, queue);
        magmablas_dtrsm_work_batched( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                              1, n2, n1, 
                              MAGMA_D_ONE,
                              dA_displ,    ldda, 
                              dW0_displ,   ldda, 
                              dX_array,    n2, 
                              dinvA_array, dinvA_length, 
                              dW1_displ,   dW2_displ, 
                              dW3_displ,   dW4_displ,
                              0, batchCount, queue );
        // update A22
        //if (DEBUG == 1) printf("calling update A22=A(%d,%d) using A21 == A(%d,%d) with m=%d n=%d k=%d\n",p2,p2,p2,p1,n2,n2,n1);
        magma_ddisplace_pointers(dA_displ,  dA_array, ldda, p2, p1, batchCount, queue);        
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, p2, p2, batchCount, queue);        // NEED TO BE REPLACED BY HERK
        magma_dgemm_batched( MagmaNoTrans, MagmaConjTrans, n2, n2, n1,
                             alpha, dA_displ, ldda, 
                             dA_displ, ldda, 
                             beta,  dW0_displ, ldda, 
                             batchCount, queue );

        // panel on A22
        //if (DEBUG == 1) printf("calling recursive panel on A22=A(%d,%d) with n=%d min_recpnb %d\n",p2,p2,n2,min_recpnb);
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, p2, p2, batchCount, queue);        
        magma_dpotrf_rectile_batched(
                           uplo, n2, n2, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ, 
                           info_array, gbstep,
                           batchCount, queue);
    }

    if (m > n) {
        // TRSM on A3:
        //if (DEBUG == 1) printf("calling trsm AT THE END on A3=A(%d,%d): using A1222 == A(%d,%d) with m=%d k=%d \n",n,0,0,0,m-n,n);
        magma_ddisplace_pointers(dA_displ,  dA_array, ldda, 0, 0, batchCount, queue);        
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, n, 0, batchCount, queue);
        magmablas_dtrsm_work_batched( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                              1, m-n, n, 
                              MAGMA_D_ONE,
                              dA_displ,    ldda, 
                              dW0_displ,   ldda, 
                              dX_array,    m-n, 
                              dinvA_array, dinvA_length, 
                              dW1_displ,   dW2_displ, 
                              dW3_displ,   dW4_displ,
                              0, batchCount, queue );
    }

    magma_free(dA_displ);
    return 0;
}
