/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2013
       
       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/
#include "common_magma.h"
#include "batched_kernel_param.h"
////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ, 
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{
    //===============================================
    //  panel factorization
    //===============================================
    if(n<nb){
        printf("magma_zpotrf_panel error n < nb %d < %d \n",(int) n, (int) nb);
        return -101;
    }

#if 0
    magma_zpotf2_batched(
                       uplo, n, nb,
                       dA_array, ldda,
                       dW1_displ, dW2_displ,
                       dW3_displ, dW4_displ,
                       info_array, gbstep,
                       batchCount, myhandle);
#else
    magma_zpotf2_batched(
                       uplo, nb, nb,
                       dA_array, ldda,
                       dW1_displ, dW2_displ,
                       dW3_displ, dW4_displ,
                       info_array, gbstep,
                       batchCount, myhandle, queue);

    if((n-nb) > 0){
        magma_zdisplace_pointers(dW0_displ, dA_array, ldda, nb, 0, batchCount, queue);
        magmablas_ztrsm_work_batched(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                              1, n-nb, nb, 
                              MAGMA_Z_ONE,
                              dA_array,    ldda, 
                              dW0_displ,   ldda, 
                              dX_array,    n-nb, 
                              dinvA_array, dinvA_length, 
                              dW1_displ,   dW2_displ, 
                              dW3_displ,   dW4_displ,
                              0, batchCount, queue);
    }
#endif
    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ,  
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{

    // Quick return if possible
    if (m ==0 || n == 0) {
        return 1;
    }
    if (uplo == MagmaUpper) {
       printf("Upper side is unavailable \n");
       return -100;
    }
    if(m<n){
        printf("error m < n %d < %d \n", (int) m, (int) n);
        return -101;
    }

    magmaDoubleComplex **dA_displ  = NULL;
    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));

    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;
    magma_int_t panel_nb = n;
    if(panel_nb <= min_recpnb){
        //printf("calling bottom panel recursive with m=%d nb=%d\n",m,n);
        //  panel factorization
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, 0, 0, batchCount, queue);
        //magma_zpotrf_rectile_batched(uplo, m, panel_nb, 16,
        magma_zpotrf_panel_batched( uplo, m, panel_nb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ,
                           info_array, gbstep,
                           batchCount, myhandle, queue);


    }
    else{
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
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, p1, p1, batchCount, queue);        
        magma_zpotrf_recpanel_batched(
                           uplo, m1, n1, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ, 
                           info_array, gbstep,
                           batchCount, myhandle, queue);

        // update A2
        //printf("calling update A2 with             m=%d n=%d k=%d\n",m2,n2,n1);
        magma_zdisplace_pointers(dA_displ,  dA_array, ldda, p1+n1, p1, batchCount, queue);        
        magma_zdisplace_pointers(dW0_displ, dA_array, ldda, p1+n1, p2, batchCount, queue);        
        magmablas_zgemm_batched(MagmaNoTrans, MagmaConjTrans, m2, n2, n1,
                              alpha, dA_displ, ldda, 
                              dA_displ, ldda, 
                              beta,  dW0_displ, ldda, 
                              batchCount, queue);
        // panel on A2
        //printf("calling recursive panel on A2 with m=%d nb=%d min_recpnb %d\n",m2,n2,min_recpnb);
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, p2, p2, batchCount, queue);        
        magma_zpotrf_recpanel_batched(
                           uplo, m2, n2, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ,
                           info_array, gbstep,
                           batchCount, myhandle, queue);
    }

    magma_free(dA_displ);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zpotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ,  
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue)
{
    //magma_int_t DEBUG=0;

    // Quick return if possible
    if (m ==0 || n == 0) {
        return 1;
    }
    if (uplo == MagmaUpper) {
       printf("Upper side is unavailable \n");
       return -100;
    }
    if(m<n){
        printf("error m < n %d < %d \n", (int) m, (int) n);
        return -101;
    }

    magmaDoubleComplex **dA_displ  = NULL;
    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));

    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;
    magma_int_t panel_nb = n;
    if(panel_nb <= min_recpnb){
        // if(DEBUG==1) printf("calling bottom panel recursive with n=%d\n",(int) panel_nb);
        //  panel factorization
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, 0, 0, batchCount, queue);
        magma_zpotrf_panel_batched(
                           uplo, m, panel_nb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ,
                           info_array, gbstep,
                           batchCount, myhandle, queue);
    }
    else{
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
        //if(DEBUG==1) printf("calling recursive panel on A11=A(%d,%d) with n=%d min_recpnb %d\n",(int) p1, (int) p1, (int) n1, (int) min_recpnb);
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, p1, p1, batchCount, queue);        
        magma_zpotrf_rectile_batched(
                           uplo, n1, n1, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ, 
                           info_array, gbstep,
                           batchCount, myhandle, queue);

        // TRSM on A21
        //if(DEBUG==1) printf("calling trsm on A21=A(%d,%d) using A11==A(%d,%d) with m=%d k=%d \n",p2,p1,p1,p1,n2,n1);
        magma_zdisplace_pointers(dA_displ,  dA_array, ldda, p1, p1, batchCount, queue);        
        magma_zdisplace_pointers(dW0_displ, dA_array, ldda, p2, p1, batchCount, queue);
        magmablas_ztrsm_work_batched(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                              1, n2, n1, 
                              MAGMA_Z_ONE,
                              dA_displ,    ldda, 
                              dW0_displ,   ldda, 
                              dX_array,    n2, 
                              dinvA_array, dinvA_length, 
                              dW1_displ,   dW2_displ, 
                              dW3_displ,   dW4_displ,
                              0, batchCount, queue);
        // update A22
        //if(DEBUG==1) printf("calling update A22=A(%d,%d) using A21==A(%d,%d) with m=%d n=%d k=%d\n",p2,p2,p2,p1,n2,n2,n1);
        magma_zdisplace_pointers(dA_displ,  dA_array, ldda, p2, p1, batchCount, queue);        
        magma_zdisplace_pointers(dW0_displ, dA_array, ldda, p2, p2, batchCount, queue);        
        magmablas_zgemm_batched(MagmaNoTrans, MagmaConjTrans, n2, n2, n1,
                              alpha, dA_displ, ldda, 
                              dA_displ, ldda, 
                              beta,  dW0_displ, ldda, 
                              batchCount, queue);

        // panel on A22
        //if(DEBUG==1) printf("calling recursive panel on A22=A(%d,%d) with n=%d min_recpnb %d\n",p2,p2,n2,min_recpnb);
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, p2, p2, batchCount, queue);        
        magma_zpotrf_rectile_batched(
                           uplo, n2, n2, min_recpnb,
                           dA_displ, ldda,
                           dX_array, dX_length,
                           dinvA_array, dinvA_length,
                           dW0_displ, dW1_displ, dW2_displ,
                           dW3_displ, dW4_displ, 
                           info_array, gbstep,
                           batchCount, myhandle, queue);
    }

    if(m>n){
        // TRSM on A3:
        //if(DEBUG==1) printf("calling trsm AT THE END on A3=A(%d,%d): using A1222==A(%d,%d) with m=%d k=%d \n",n,0,0,0,m-n,n);
        magma_zdisplace_pointers(dA_displ,  dA_array, ldda, 0, 0, batchCount, queue);        
        magma_zdisplace_pointers(dW0_displ, dA_array, ldda, n, 0, batchCount, queue);
        magmablas_ztrsm_work_batched(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                              1, m-n, n, 
                              MAGMA_Z_ONE,
                              dA_displ,    ldda, 
                              dW0_displ,   ldda, 
                              dX_array,    m-n, 
                              dinvA_array, dinvA_length, 
                              dW1_displ,   dW2_displ, 
                              dW3_displ,   dW4_displ,
                              0, batchCount, queue);
    }

    magma_free(dA_displ);
    return 0;
}




