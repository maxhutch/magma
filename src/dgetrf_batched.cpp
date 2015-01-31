/*
   -- MAGMA (version 1.6.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date January 2015

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from zgetrf_batched.cpp normal z -> d, Fri Jan 30 19:00:19 2015
*/
#include "common_magma.h"
#include "batched_kernel_param.h"
///////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    DGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    If the current stream is NULL, this version replaces it with a new
    stream to overlap computation with communication.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE_PRECISION array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_dgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgetrf_batched(
        magma_int_t m, magma_int_t n,
        double **dA_array, 
        magma_int_t ldda,
        magma_int_t **ipiv_array, 
        magma_int_t *info_array, 
        magma_int_t batchCount, magma_queue_t queue)
{
#define A(i_, j_)  (A + (i_) + (j_)*ldda)   

    magma_int_t min_mn = min(m, n);
    cudaMemset(info_array, 0, batchCount*sizeof(magma_int_t));

    /* Check arguments */
    magma_int_t arginfo = 0;
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        if(min_mn == 0 ) return arginfo;

    if( m >  2048 || n > 2048 ){
        printf("=========================================================================================\n");
        printf("   WARNING batched routines are designed for small sizes it might be better to use the\n   Native/Hybrid classical routines if you want performance\n");
        printf("=========================================================================================\n");
    }


//#define ENABLE_TIMER3

#if defined(ENABLE_TIMER3)
    real_Double_t   tall=0.0, tloop=0., talloc=0., tdalloc=0.;
    tall   = magma_sync_wtime(0);
    talloc = magma_sync_wtime(0);
#endif

    double neg_one = MAGMA_D_NEG_ONE;
    double one  = MAGMA_D_ONE;
    magma_int_t ib, i, k, pm;
    magma_int_t nb = BATRF_NB;
    magma_int_t gemm_crossover = nb > 32 ? 127 : 160;
    // magma_int_t gemm_crossover = n;// use only stream gemm

#if defined(USE_CUOPT)    
    cublasHandle_t myhandle;
    cublasCreate_v2(&myhandle);
#else
    cublasHandle_t myhandle=NULL;
#endif

    magma_int_t     **dipiv_displ   = NULL;
    double **dA_displ   = NULL;
    double **dW0_displ  = NULL;
    double **dW1_displ  = NULL;
    double **dW2_displ  = NULL;
    double **dW3_displ  = NULL;
    double **dW4_displ  = NULL;
    double **dinvA_array = NULL;
    double **dwork_array = NULL;


    magma_malloc((void**)&dipiv_displ,   batchCount * sizeof(*dipiv_displ));
    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dW0_displ,  batchCount * sizeof(*dW0_displ));
    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dW4_displ,  batchCount * sizeof(*dW4_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dwork_array, batchCount * sizeof(*dwork_array));


    magma_int_t invA_msize = ((n+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB;
    magma_int_t dwork_msize = n*nb;
    magma_int_t **pivinfo_array    = NULL;
    magma_int_t *pivinfo           = NULL; 
    double* dinvA      = NULL;
    double* dwork      = NULL;// dinvA and dwork are workspace in dtrsm
    double **cpuAarray = NULL;
    magma_dmalloc( &dinvA, invA_msize * batchCount);
    magma_dmalloc( &dwork, dwork_msize * batchCount );
    magma_malloc((void**)&pivinfo_array, batchCount * sizeof(*pivinfo_array));
    magma_malloc((void**)&pivinfo, batchCount * m * sizeof(magma_int_t));
    magma_malloc_cpu((void**) &cpuAarray, batchCount*sizeof(double*));

   /* check allocation */
    if ( dA_displ  == NULL || dW0_displ == NULL || dW1_displ   == NULL || dW2_displ   == NULL || 
         dW3_displ == NULL || dW4_displ == NULL || dinvA_array == NULL || dwork_array == NULL || 
         dinvA     == NULL || dwork     == NULL || cpuAarray   == NULL || 
         dipiv_displ == NULL || pivinfo_array == NULL || pivinfo == NULL) {
        magma_free(dA_displ);
        magma_free(dW0_displ);
        magma_free(dW1_displ);
        magma_free(dW2_displ);
        magma_free(dW3_displ);
        magma_free(dW4_displ);
        magma_free(dinvA_array);
        magma_free(dwork_array);
        magma_free( dinvA );
        magma_free( dwork );
        free(cpuAarray);
        magma_free(dipiv_displ);
        magma_free(pivinfo_array);
        magma_free(pivinfo);
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }


    magmablas_dlaset_q(MagmaFull, invA_msize, batchCount, MAGMA_D_ZERO, MAGMA_D_ZERO, dinvA, invA_msize, queue);
    magmablas_dlaset_q(MagmaFull, dwork_msize, batchCount, MAGMA_D_ZERO, MAGMA_D_ZERO, dwork, dwork_msize, queue);
    dset_pointer(dwork_array, dwork, 1, 0, 0, dwork_msize, batchCount, queue);
    dset_pointer(dinvA_array, dinvA, TRI_NB, 0, 0, invA_msize, batchCount, queue);
    set_ipointer(pivinfo_array, pivinfo, 1, 0, 0, m, batchCount, queue);


    // printf(" I am in dgetrfbatched\n");
    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);
    magma_int_t streamid;
    const magma_int_t nbstreams=32;
    magma_queue_t stream[nbstreams];
    for(i=0; i<nbstreams; i++){
        magma_queue_create( &stream[i] );
    }
    magma_getvector( batchCount, sizeof(double*), dA_array, 1, cpuAarray, 1);



#if defined(ENABLE_TIMER3)
    printf(" I am after malloc\n");
    talloc = magma_sync_wtime(0) - talloc;
    tloop  = magma_sync_wtime(0);
#endif


    for(i = 0; i < min_mn; i+=nb) 
    {
        magmablasSetKernelStream(NULL);

        ib = min(nb, min_mn-i);
        pm = m-i;
        magma_idisplace_pointers(dipiv_displ, ipiv_array, ldda, i, 0, batchCount, queue);
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, i, i, batchCount, queue);
        //===============================================
        //  panel factorization
        //===============================================
#if 0
        arginfo = magma_dgetf2_batched(
                pm, ib,
                dA_displ, ldda,
                dW1_displ, dW2_displ, dW3_displ,
                dipiv_displ, 
                info_array, i, batchCount, myhandle);   
#else
        arginfo = magma_dgetrf_recpanel_batched(
                pm, ib, 16,
                dA_displ, ldda,
                dipiv_displ, pivinfo_array,
                dwork_array, nb, 
                dinvA_array, invA_msize, 
                dW0_displ, dW1_displ, dW2_displ, 
                dW3_displ, dW4_displ,
                info_array, i, 
                batchCount, myhandle, queue);   
#endif
        if(arginfo != 0 ) goto fin;
        //===============================================
        // end of panel
        //===============================================

#define RUN_ALL
#ifdef RUN_ALL
        // setup pivinfo before adjusting ipiv
        setup_pivinfo_batched(pivinfo_array, dipiv_displ, pm, ib, batchCount, queue);
        adjust_ipiv_batched(dipiv_displ, ib, i, batchCount, queue);

        // stepinit_ipiv(pivinfo_array, pm, batchCount);// for debug and check swap, it create an ipiv


#if 0
        dlaswp_batched( i, dA_displ, ldda,
                i, i+ib,
                dipiv_displ, pivinfo_array, batchCount);
#else
        magma_ddisplace_pointers(dA_displ, dA_array, ldda, i, 0, batchCount, queue);
        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, i, 0, batchCount, queue);
        magma_dlaswp_rowparallel_batched( i, dA_displ, ldda,
                dW0_displ, ldda,
                i, i+ib,
                pivinfo_array, batchCount, queue);

#endif

        if( (i + ib) < n)
        {
            // swap right side and trsm     
            magma_ddisplace_pointers(dA_displ, dA_array, ldda, i, i+ib, batchCount, queue);
            dset_pointer(dwork_array, dwork, nb, 0, 0, dwork_msize, batchCount, queue); // I don't think it is needed Azzam
            magma_dlaswp_rowparallel_batched( n-(i+ib), dA_displ, ldda,
                    dwork_array, nb,
                    i, i+ib,
                    pivinfo_array, batchCount, queue);


            magma_ddisplace_pointers(dA_displ, dA_array, ldda, i, i, batchCount, queue);
            magma_ddisplace_pointers(dW0_displ, dA_array, ldda, i, i+ib, batchCount, queue);
            magmablas_dtrsm_outofplace_batched(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 1,
                    ib, n-i-ib,
                    MAGMA_D_ONE,
                    dA_displ,    ldda, // dA
                    dwork_array,  nb, // dB
                    dW0_displ,   ldda, // dX
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    0, batchCount, queue);


            if( (i + ib) < m)
            {    
                // if gemm size is >160 use a streamed classical cublas gemm since it is faster
                // the batched is faster only when M=N<=160 for K40c
                //-------------------------------------------
                //          USE STREAM  GEMM
                //-------------------------------------------
                if( (m-i-ib) > gemm_crossover  && (n-i-ib) > gemm_crossover)   
                { 
                    //printf("caling streamed dgemm %d %d %d \n", m-i-ib, n-i-ib, ib);

                    // since it use different stream I need to wait the TRSM and swap.
                    // But since the code use the NULL stream everywhere, 
                    // so I don't need it, because the NULL stream do the sync by itself
                    //magma_queue_sync(NULL); 
                    //
                    for(k=0; k<batchCount; k++)
                    {
                        streamid = k%nbstreams;                                       
                        magmablasSetKernelStream(stream[streamid]);
                        magma_dgemm(MagmaNoTrans, MagmaNoTrans, 
                                m-i-ib, n-i-ib, ib,
                                neg_one, cpuAarray[k] + (i+ib)+i*ldda, ldda, 
                                         cpuAarray[k] + i+(i+ib)*ldda, ldda,
                                one,     cpuAarray[k] + (i+ib)+(i+ib)*ldda, ldda);
                    }
                    // need to synchronise to be sure that dgetf2 do not start before
                    // finishing the update at least of the next panel
                    // BUT no need for it as soon as the other portion of the code 
                    // use the NULL stream which do the sync by itself 
                    //magma_device_sync(); 
                }
                //-------------------------------------------
                //          USE BATCHED GEMM
                //-------------------------------------------
                else
                {
                    magma_ddisplace_pointers(dA_displ, dA_array,  ldda, i+ib,    i, batchCount, queue);
                    magma_ddisplace_pointers(dW1_displ, dA_array, ldda,    i, i+ib, batchCount, queue);
                    magma_ddisplace_pointers(dW2_displ, dA_array, ldda, i+ib, i+ib, batchCount, queue);
                    //printf("caling batched dgemm %d %d %d \n", m-i-ib, n-i-ib, ib);
                    magmablas_dgemm_batched( MagmaNoTrans, MagmaNoTrans, m-i-ib, n-i-ib, ib, 
                            neg_one, dA_displ, ldda, 
                            dW1_displ, ldda, 
                            one,  dW2_displ, ldda, 
                            batchCount, queue);
                } // end of batched/stream gemm
            } // end of  if( (i + ib) < m) 
        } // end of if( (i + ib) < n)
#endif
    }// end of for

fin:
    magma_queue_sync(NULL);

#if defined(ENABLE_TIMER3)
    tloop   = magma_sync_wtime(0) - tloop;
    tdalloc = magma_sync_wtime(0);

#endif

    for(i=0; i<nbstreams; i++){
        magma_queue_destroy( stream[i] );
    }
    magmablasSetKernelStream(cstream);


#if defined(USE_CUOPT)    
    cublasDestroy_v2(myhandle);
#endif

    magma_free(dA_displ);
    magma_free(dW0_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dW4_displ);
    magma_free(dinvA_array);
    magma_free(dwork_array);
    magma_free( dinvA );
    magma_free( dwork );
    free(cpuAarray);
    magma_free(dipiv_displ);
    magma_free(pivinfo_array);
    magma_free(pivinfo);

#if defined(ENABLE_TIMER3)
    tdalloc = magma_sync_wtime(0) - tdalloc;
    tall = magma_sync_wtime(0) - tall;
    printf("here is the timing from inside dgetrf_batched talloc: %10.5f  tloop: %10.5f tdalloc: %10.5f tall: %10.5f sum: %10.5f\n", talloc, tloop, tdalloc, tall, talloc+tloop+tdalloc );
#endif
    
    return arginfo;

}





