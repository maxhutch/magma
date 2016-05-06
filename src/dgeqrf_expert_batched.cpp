/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar
       @author Tingxing Dong

       @generated from src/zgeqrf_expert_batched.cpp normal z -> d, Mon May  2 23:30:29 2016
*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "batched_kernel_param.h"

/**
    Purpose
    -------
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit, the elements on and above the diagonal of the array
             contain the min(M,N)-by-N upper trapezoidal matrix R (R is
             upper triangular if m >= n); the elements below the diagonal,
             with the array TAU, represent the orthogonal matrix Q as a
             product of min(m,n) elementary reflectors (see Further
             Details).

    @param[in]
    ldda     INTEGER
             The leading dimension of the array dA.  LDDA >= max(1,M).
             To benefit from coalescent memory accesses LDDA must be
             divisible by 16.

    @param[in,out]
    dR_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array on the GPU, dimension (LDDR, N/NB)
             dR should be of size (LDDR, N) when provide_RT > 0 and 
             of size (LDDT, NB) otherwise. NB is the local blocking size.
             On exit, the elements of R are stored in dR only when provide_RT > 0.

    @param[in]
    lddr     INTEGER
             The leading dimension of the array dR.  
             LDDR >= min(M,N) when provide_RT == 1
             otherwise LDDR >= min(NB, min(M,N)). 
             NB is the local blocking size.
             To benefit from coalescent memory accesses LDDR must be
             divisible by 16.

    @param[in,out]
    dT_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array on the GPU, dimension (LDDT, N/NB)
             dT should be of size (LDDT, N) when provide_RT > 0 and 
             of size (LDDT, NB) otherwise. NB is the local blocking size.
             On exit, the elements of T are stored in dT only when provide_RT > 0.

    @param[in]
    lddt     INTEGER
             The leading dimension of the array dT.  
             LDDT >= min(NB,min(M,N)). NB is the local blocking size.
             To benefit from coalescent memory accesses LDDR must be
             divisible by 16.

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

    @param[in]
    provide_RT INTEGER
               provide_RT = 0 no R and no T in output. 
               dR and dT are used as local workspace to store the R and T of each step.
               provide_RT = 1 the whole R of size (min(M,N), N) and the nbxnb  block of T are provided in output. 
               provide_RT = 2 the nbxnb diag block of R and of T are provided in output. 

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_dgeqrf_comp
    ********************************************************************/


extern "C" magma_int_t
magma_dgeqrf_expert_batched(
    magma_int_t m, magma_int_t n, 
    double **dA_array, magma_int_t ldda, 
    double **dR_array, magma_int_t lddr,
    double **dT_array, magma_int_t lddt,
    double **dtau_array, magma_int_t provide_RT,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
#define dA(i, j)  (dA + (i) + (j)*ldda)   // A(i, j) means at i row, j column
    
    /* Local Parameter */
    magma_int_t nb = magma_get_dgeqrf_batched_nb(m);
    
    magma_int_t nnb = 8;
    magma_int_t min_mn = min(m, n);

    /* Check arguments */
    cudaMemset(info_array, 0, batchCount*sizeof(magma_int_t));
    magma_int_t arginfo = 0;
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;
    else if (lddr < min_mn && provide_RT == 1)
        arginfo = -6;
    else if (lddr < min(min_mn, nb))
        arginfo = -6;
    else if (lddt < min(min_mn, nb))
        arginfo = -8;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        if (min_mn == 0 ) return arginfo;

    if ( m >  2048 || n > 2048 ) {
        printf("=========================================================================================\n");
        printf("   WARNING batched routines are designed for small sizes it might be better to use the\n   Native/Hybrid classical routines if you want performance\n");
        printf("=========================================================================================\n");
    }


    magma_int_t i, k, ib=nb, jb=nnb, offset_RT=0, use_stream;
    magma_int_t ldw, offset; 

    double **dW0_displ = NULL;
    double **dW1_displ = NULL;
    double **dW2_displ = NULL;
    double **dW3_displ = NULL;
    double **dW4_displ = NULL;
    double **dW5_displ = NULL;
    double **dR_displ  = NULL;
    double **dT_displ  = NULL;

    double *dwork = NULL;
    double **cpuAarray = NULL;
    double **cpuTarray = NULL;

    magma_malloc((void**)&dW0_displ, batchCount * sizeof(*dW0_displ));
    magma_malloc((void**)&dW1_displ, batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ, batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ, batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dW4_displ, batchCount * sizeof(*dW4_displ));
    magma_malloc((void**)&dW5_displ, batchCount * sizeof(*dW5_displ));
    magma_malloc((void**)&dR_displ,  batchCount * sizeof(*dR_displ));
    magma_malloc((void**)&dT_displ,  batchCount * sizeof(*dT_displ));

    magma_dmalloc(&dwork,  (2 * nb * n) * batchCount);
    magma_malloc_cpu((void**) &cpuAarray, batchCount*sizeof(double*));
    magma_malloc_cpu((void**) &cpuTarray, batchCount*sizeof(double*));

    /* check allocation */
    if ( dW0_displ == NULL || dW1_displ == NULL || dW2_displ == NULL || 
         dW3_displ == NULL || dW4_displ == NULL || dW5_displ == NULL || 
         dR_displ  == NULL || dT_displ  == NULL || dwork     == NULL ||
         cpuAarray == NULL || cpuTarray == NULL ) {
        magma_free(dW0_displ);
        magma_free(dW1_displ);
        magma_free(dW2_displ);
        magma_free(dW3_displ);
        magma_free(dW4_displ);
        magma_free(dW5_displ);
        magma_free(dR_displ);
        magma_free(dT_displ);
        magma_free(dwork);
        magma_free_cpu(cpuAarray);
        magma_free_cpu(cpuTarray);
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magma_ddisplace_pointers(dR_displ, dR_array, lddr, 0, 0, batchCount, queue); 
    magma_ddisplace_pointers(dT_displ, dT_array, lddt, 0, 0, batchCount, queue); 
    // set dR and dT to zero. if provide_RT == 0 only a tile of size nbxnb is used and overwritten at each step
    magmablas_dlaset_batched( MagmaFull, lddr, (provide_RT > 0 ? n:min(min_mn,nb)), MAGMA_D_ZERO, MAGMA_D_ZERO, dR_displ, lddr, batchCount, queue ); 
    magmablas_dlaset_batched( MagmaFull, lddt, (provide_RT > 0 ? n:min(min_mn,nb)), MAGMA_D_ZERO, MAGMA_D_ZERO, dT_displ, lddt, batchCount, queue );
    /*
    if ( provide_RT > 0 )
    {
        magmablas_dlaset_q( MagmaFull, lddr, n*batchCount, MAGMA_D_ZERO, MAGMA_D_ZERO, dR, lddr, queue );
        magmablas_dlaset_q( MagmaFull, lddt, n*batchCount, MAGMA_D_ZERO, MAGMA_D_ZERO, dT, lddt, queue );
    }
    else
    {
        magmablas_dlaset_q( MagmaFull, lddr, nb*batchCount, MAGMA_D_ZERO, MAGMA_D_ZERO, dR, lddr, queue );
        magmablas_dlaset_q( MagmaFull, lddt, nb*batchCount, MAGMA_D_ZERO, MAGMA_D_ZERO, dT, lddt, queue );
    }
    */
    magma_int_t streamid;
    const magma_int_t nbstreams=10;
    magma_queue_t queues[nbstreams];
    for (i=0; i < nbstreams; i++) {
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[i] );
    }
    magma_getvector( batchCount, sizeof(double*), dA_array, 1, cpuAarray, 1, queue);
    magma_getvector( batchCount, sizeof(double*), dT_array, 1, cpuTarray, 1, queue);


    for (i=0; i < min_mn; i += nb)
    {
        ib = min(nb, min_mn-i);  
        //===============================================
        // panel factorization
        //===============================================

        magma_ddisplace_pointers(dW0_displ, dA_array, ldda, i, i, batchCount, queue); 
        magma_ddisplace_pointers(dW2_displ, dtau_array, 1, i, 0, batchCount, queue);
        if ( provide_RT > 0 )
        {
            offset_RT = i;
            magma_ddisplace_pointers(dR_displ, dR_array, lddr, (provide_RT == 1 ? offset_RT:0), offset_RT, batchCount, queue); 
            magma_ddisplace_pointers(dT_displ, dT_array, lddt, 0, offset_RT, batchCount, queue); 
        }

        //dwork is used in panel factorization and trailing matrix update
        //dW4_displ, dW5_displ are used as workspace and configured inside
        magma_dgeqrf_panel_batched(m-i, ib, jb, 
                                   dW0_displ, ldda, 
                                   dW2_displ, 
                                   dT_displ, lddt, 
                                   dR_displ, lddr,
                                   dW1_displ,
                                   dW3_displ,
                                   dwork, 
                                   dW4_displ, dW5_displ,
                                   info_array,
                                   batchCount, queue);
           
        //===============================================
        // end of panel
        //===============================================

        //===============================================
        // update trailing matrix
        //===============================================
        if ( (n-ib-i) > 0)
        {
            //dwork is used in panel factorization and trailing matrix update
            //reset dW4_displ
            ldw = nb;
            magma_dset_pointer( dW4_displ, dwork, 1, 0, 0,  ldw*n, batchCount, queue );
            offset = ldw*n*batchCount;
            magma_dset_pointer( dW5_displ, dwork + offset, 1, 0, 0,  ldw*n, batchCount, queue );    

            // set the diagonal of v as one and the upper triangular part as zero already set inside geqrf_panel
            //magmablas_dlaset_batched( MagmaUpper, ib, ib, MAGMA_D_ZERO, MAGMA_D_ONE, dW0_displ, ldda, batchCount, queue ); 
            //magma_ddisplace_pointers(dW2_displ, dtau_array, 1, i, 0, batchCount, queue); 

            // it is faster since it is using BLAS-3 GEMM routines, different from lapack implementation 
            magma_dlarft_batched(m-i, ib, 0,
                             dW0_displ, ldda,
                             dW2_displ,
                             dT_displ, lddt, 
                             dW4_displ, nb*lddt,
                             batchCount, queue);

            
            // perform C = (I-V T^H V^H) * C, C is the trailing matrix
            //-------------------------------------------
            //          USE STREAM  GEMM
            //-------------------------------------------
            use_stream = magma_drecommend_cublas_gemm_stream(MagmaNoTrans, MagmaNoTrans, m-i-ib, n-i-ib, ib);
            if ( use_stream )   
            { 
                magma_queue_sync(queue); 
                for (k=0; k < batchCount; k++)
                {
                    streamid = k%nbstreams;                                       
                    // the queue gemm must take cpu pointer 
                    magma_dlarfb_gpu_gemm( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                m-i, n-i-ib, ib,
                                cpuAarray[k] + i + i * ldda, ldda, 
                                cpuTarray[k] + offset_RT*lddt, lddt,
                                cpuAarray[k] + i + (i+ib) * ldda, ldda,
                                dwork + nb * n * k, -1,
                                dwork + nb * n * batchCount + nb * n * k, -1, queues[streamid] );
                }

                // need to synchronise to be sure that panel does not start before
                // finishing the update at least of the next panel
                // if queue is NULL, no need to sync
                if ( queue != NULL ) {
                    for (magma_int_t s=0; s < nbstreams; s++)
                        magma_queue_sync(queues[s]);
                }
            }
            //-------------------------------------------
            //          USE BATCHED GEMM
            //-------------------------------------------
            else
            {
                //direct trailing matrix in dW1_displ
                magma_ddisplace_pointers(dW1_displ, dA_array, ldda, i, i+ib, batchCount, queue); 

                magma_dlarfb_gemm_batched( 
                            MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise, 
                            m-i, n-i-ib, ib,
                            (const double**)dW0_displ, ldda,
                            (const double**)dT_displ, lddt,
                            dW1_displ,  ldda,
                            dW4_displ,  ldw,
                            dW5_displ, ldw,
                            batchCount, queue );
            }
        }// update the trailing matrix 
        //===============================================

        // copy dR back to V after the trailing matrix update, 
        // only when provide_RT=0 otherwise the nbxnb block of V is set to diag=1/0
        // The upper portion of V could be set totaly to 0 here
        if ( provide_RT == 0 )
        {
            magmablas_dlacpy_batched( MagmaUpper, ib, ib, dR_displ, lddr, dW0_displ, ldda, batchCount, queue );
        }
    }

    magma_queue_sync(queue);
    for (k=0; k < nbstreams; k++) {
        magma_queue_destroy( queues[k] );
    }
    
    magma_free(dW0_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dW4_displ);
    magma_free(dW5_displ);
    magma_free(dR_displ);
    magma_free(dT_displ);
    magma_free(dwork);
    magma_free_cpu(cpuAarray);
    magma_free_cpu(cpuTarray);

    return arginfo;
}
