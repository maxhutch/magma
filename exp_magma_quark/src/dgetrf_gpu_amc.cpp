/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
 
*/ 

#include <math.h>
#include "common_magma.h"

#include "magma_amc.h"

#include "schedule.h" 

#include "core_d.h"

 
#define dA(i,j) (dA + (j)*nb*dA_LD + (i)*nb) 
#define dAT(i,j) (dAT + (i)*nb*dAT_LD + (j)*nb) 
 

 
#define A(I,J) (A + ((J)%A_NMAX)*nb*A_LD + (I)*nb)
#define colptr(J) A(0,(J))

#ifdef PARALLEL_AWORK 
/*Parallel allocation of the workspace is to avoid pinned memory overhead: Now obsolete*/
#undef A
#undef colptr
#define A(I,J) &(A[((J)%A_SIZE)][(I)*nb])
//#define colptr(J) A[((J)%A_SIZE)] 
#define colptr(J) (void *)(intptr_t)(((J)%A_SIZE)+1) //A[((J)%A_SIZE)]
#endif


#define ipiv(I) (ipiv+(I)*nb) 
 



/*
*
*  magma_dgetrf_gpu_amc:
*
*  This function allocate the workspace and call magma_dgetrf_gpu_work_amc
*  REC: recursif Algorithm
*  ASYNC: asynchronous algorithm
*/

extern "C" magma_int_t magma_dgetrf_gpu_amc(magma_int_t m, magma_int_t n,  
                 double *dA, magma_int_t dA_LD, 
                 magma_int_t *ipiv, magma_int_t *info) 
{ 
/*  -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       November 2011 
 
    Purpose 
    ======= 
 
    DGETRF_GPU_AMC computes an LU factorization of a general M-by-N matrix A 
    using partial pivoting with row interchanges. The technique used for the panel factorization
    is the parallel recursif LU (see lawn 259).
 
    The factorization has the form 
       A = P * L * U 
    where P is a permutation matrix, L is lower triangular with unit 
    diagonal elements (lower trapezoidal if m > n), and U is upper 
    triangular (upper trapezoidal if m < n). 
 
    This is the right-looking Level 3 BLAS version of the algorithm. 
 
    Arguments 
    ========= 
 
    M       (input) INTEGER 
            The number of rows of the matrix A.  M >= 0. 
 
    N       (input) INTEGER 
            The number of columns of the matrix A.  N >= 0. 
 
    A       (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N). 
            On entry, the M-by-N matrix to be factored. 
            On exit, the factors L and U from the factorization 
            A = P*L*U; the unit diagonal elements of L are not stored. 
 
    LDDA     (input) INTEGER 
            The leading dimension of the array A.  LDDA >= max(1,M). 
 
    IPIV    (output) INTEGER array, dimension (min(M,N)) 
            The pivot indices; for 1 <= i <= min(M,N), row i of the 
            matrix was interchanged with row IPIV(i). 
 
    INFO    (output) INTEGER 
            = 0:  successful exit 
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  or another error occured, such as memory allocation failed. 
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization 
                  has been completed, but the factor U is exactly 
                  singular, and division by zero will occur if it is used 
                  to solve a system of equations. 
    =====================================================================    */ 
 
     
     /*Workspace*/
     double *AWORK;
     magma_int_t AWORK_LD, AWORK_n;
    
      
     int nbcores; /*Number of cores available for the whole factorization*/ 
    // int panel_num_threads; /*Number of threads for the panel*/ 
     double dcpu; /*percentage of the matrix to allocate on the CPUs*/ 
     int nb;
     amc_args_t *args;

#if (dbglevel >=1) 
    double t1;
#endif



    /* Check arguments */ 
    *info = 0; 
    if (m < 0) 
        *info = -1; 
    else if (n < 0) 
        *info = -2; 
    else if (dA_LD < max(1,m)) 
        *info = -4; 
 
    if (*info != 0) { 
        magma_xerbla( __func__, -(*info) ); 
        return *info; 
    } 
 
    /* Quick return if possible */ 
    if (m == 0 || n == 0) 
        return *info; 
      
    /* Get parameters */  
    
    
    
    args = magma_amc_args_get_default();

    if(args->nb==0)
     nb    = magma_get_dgetrf_nb(m) ;//magma dgetrf block size
    else
     nb = args->nb;

     nbcores = args->P;  
    
     dcpu = args->dcpu;
 
     /*check and fix parameters */
     if(dcpu>1.0) dcpu = 1.0;

     /*Compute the dimension of the workspace matrix for the cpu*/
     AWORK_LD = m;
     AWORK_n = (int) ceil(n*dcpu);
     //AWORK_n += 1*nb; /* +1 avoid current panel to be overwritten*/

     /*Make LD and n multiple of 32*/
     if(AWORK_LD%32!=0) AWORK_LD = ((AWORK_LD + 31)/32)*32;
     if(AWORK_n%32!=0) AWORK_n = ((AWORK_n + 31)/32)*32; 
     

     /*Allocate the CPU part of the matrix to factorize*/
#if (dbglevel >=1)
    t1 = magma_wtime();
#endif
    
    if (MAGMA_SUCCESS != magma_dmalloc_pinned(&AWORK, AWORK_LD*AWORK_n)) { 
    //if (MAGMA_SUCCESS != magma_dmalloc_cpu(&AWORK, AWORK_LD*AWORK_n)) {
            *info = MAGMA_ERR_HOST_ALLOC; 
            return *info; 
    } 
    
#if (dbglevel >=1)
    printf("[DBG] Time memory malloc (pinned):%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

    /*First touch the workspace by each thread*/
    //magma_amc_dmemset(AWORK, 0.0, AWORK_LD*AWORK_n, nb, nbcores);

    /* Call the workspace interface */
    magma_dgetrf_gpu_work_amc(m, n, dA, dA_LD, ipiv, info, AWORK, AWORK_LD, AWORK_n);

#if (dbglevel >=1)
    printf("[DBG] Time Factorization:%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

    magma_free_pinned(AWORK);

#if (dbglevel >=1)
   printf("[DBG] Time memory free memory:%f\n",magma_wtime()-t1); 
   t1 = magma_wtime();
#endif

#if (dbglevel==10)     
    ca_dbg_printMat_transpose_gpu(m, n, dA, dA_LD,"dA = LU"); 
#endif 

    return *info; 
}   /* End of MAGMA_DGETRF_REC_ASYNC_GPU */





/*
*
*  DGETRF_REC_ASYNC_WORK_GPU
*
*  This function performs the factorization when the workspace is already allocated.
*/


extern "C" magma_int_t 
magma_dgetrf_gpu_work_amc(
magma_int_t m, magma_int_t n,  
double *dA, magma_int_t dA_LD, 
magma_int_t *ipiv, magma_int_t *info,
/*workspace on the cpu side*/
double *AWORK, magma_int_t AWORK_LD, magma_int_t AWORK_n
) 
{ 
/*  -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       November 2011 
 
    Purpose 
    ======= 
 
    DGETRF_GPU_WORK_AMC computes an LU factorization of a general M-by-N matrix A 
    using partial pivoting with row interchanges. The technique used for the panel factorization
    is the parallel recursif LU (see lawn 259).
 
    The factorization has the form 
       A = P * L * U 
    where P is a permutation matrix, L is lower triangular with unit 
    diagonal elements (lower trapezoidal if m > n), and U is upper 
    triangular (upper trapezoidal if m < n). 
 
    This is the right-looking Level 3 BLAS version of the algorithm. 
 
    Arguments 
    ========= 
 
    M       (input) INTEGER 
            The number of rows of the matrix A.  M >= 0. 
 
    N       (input) INTEGER 
            The number of columns of the matrix A.  N >= 0. 
 
    A       (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N). 
            On entry, the M-by-N matrix to be factored. 
            On exit, the factors L and U from the factorization 
            A = P*L*U; the unit diagonal elements of L are not stored. 
 
    LDDA     (input) INTEGER 
            The leading dimension of the array A.  LDDA >= max(1,M). 
 
    IPIV    (output) INTEGER array, dimension (min(M,N)) 
            The pivot indices; for 1 <= i <= min(M,N), row i of the 
            matrix was interchanged with row IPIV(i). 
 
    INFO    (output) INTEGER 
            = 0:  successful exit 
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  or another error occured, such as memory allocation failed. 
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization 
                  has been completed, but the factor U is exactly 
                  singular, and division by zero will occur if it is used 
                  to solve a system of equations.

    =====================================================================    */ 
 
 
 
    double c_one     = MAGMA_D_ONE; 
    double c_neg_one = MAGMA_D_NEG_ONE; 
 
    int ONE = 1; 
 
    magma_int_t iinfo, nb; 
    magma_int_t mindim; 
    magma_int_t nrows, ncols; 
    //double *work; 
 
 
     magma_int_t dm_max, dn_max; 
     magma_int_t I, J, K, M, N, U_K; 
  
     //magma_int_t A_K; 
     double *dAT; 
     magma_int_t dAT_LD; 
      
      
     double *dAP_set,*dAP_get; 
     magma_int_t dAP_LD; 
      
     

     //magma_int_t nrows, ncols; 
     magma_int_t gpu_nrows, gpu_ncols; 
  
     int nbcores; /*Number of cores available for the whole factorization*/ 
     int panel_num_threads; /*Number of threads for the panel*/ 
     double dcpu; /*percentage of the matrix to allocate on the CPUs*/ 
  
    int B_rows;

    double t1;
    

     /* Recommanded dimension in the workspace*/ 
     int A_m, A_n, A_N, A_NMAX, A_LD;
     double *A;
#ifdef USE_CALU     
     int i_nrows;
#endif

     amc_args_t *args;
    /*magma_event_t *A_event;*/ /*Control bucket*/



    /* Check arguments */ 
    *info = 0; 
    if (m < 0) 
        *info = -1; 
    else if (n < 0) 
        *info = -2; 
    else if (dA_LD < max(1,m)) 
        *info = -4; 
    else if (AWORK_LD < max(1,m)) 
        *info = -5;

    if (*info != 0) { 
        magma_xerbla( __func__, -(*info) ); 
        return *info; 
    } 
 
    /* Quick return if possible */ 
    if (m == 0 || n == 0) 
        return *info; 

      
     /*Get parameters*/ 
    args = magma_amc_args_get_default();
     nb= args->nb;

     nbcores = args->P;  
     panel_num_threads = args->Pr; 
     dcpu = args->dcpu;

     /* Check and fix parameters */
     if(nb==0)
        nb     = magma_get_dgetrf_nb(m) ;/*magma dgetrf block size*/ 
    else
        nb = args->nb;

     if(nb>n) nb = n; 
     if(panel_num_threads>nbcores) panel_num_threads = nbcores;

     /* Compute the maximum number of panels we can store in the workspace*/
     A_NMAX = (int) (AWORK_n/ nb);

     /* Compute the recommanded number of columns for the cpu part*/
     A_n = (int) ceil(n*dcpu);

     /*Make sure we work with multiple of 32*/
     /*
     if(A_n%32!=0) {
         A_n = ((A_n + 31)/32)*32;
     }
     */

     /* Compute the recommanded number of panels for the cpu part*/
     A_N = (int) (A_n/ nb);
     
     /* Check if there are enough workspace. In case the user gave a workspace lower than the optimal*/
     /* NOTE: using small workspace may reduce performance*/
     if(A_N>A_NMAX){    
#if (dbglevel >=1)
        printf("[DBG_WARNING] Resizing buffer to feet user preferences. Recommanded:%d, Max given:%d\n",A_N, A_NMAX); 
#endif
        A_N = A_NMAX;
    }
      
     

     A = AWORK;
     A_m = m;
     A_LD = AWORK_LD;


#if (dbglevel >=1)
    /* Initialize the tracing*/
    ca_dbg_trace_init(nbcores,1); //nbcores + 1 GPU
#endif

#if (dbglevel >=1)
    t1 = magma_wtime();
#endif

     /*Transfer the first column block of the matrix from the GPU to the CPUs*/ 
    
    magma_dgetmatrix(A_m, A_n, dA, dA_LD, A, A_LD); 

#if (dbglevel >=1)
    printf("[DBG] Time First getmatrix: %f\n",magma_wtime()-t1);
    t1 = magma_wtime();
#endif
#if (dbglevel==10)  
    ca_dbg_printMat(m, A_n, A, A_LD,"A after first getMatrix"); 
#endif


     /*Allocate a workspace for the panels transposition*/ 
     dAP_LD = m; 
     if(dAP_LD%32!=0) dAP_LD = ((dAP_LD + 31)/32)*32;/*Make dAP_LD multiple of 32*/

     if (MAGMA_SUCCESS != magma_dmalloc(&dAP_set, dAP_LD*nb)) { 
            *info = MAGMA_ERR_DEVICE_ALLOC; 
            return *info; 
    } 

     if (MAGMA_SUCCESS != magma_dmalloc(&dAP_get, dAP_LD*nb)) { 
            magma_free(dAP_set);
            *info = MAGMA_ERR_DEVICE_ALLOC; 
            return *info; 
    }


#if (dbglevel >=1)
    printf("[DBG] Time workspace memory alloc (dAP): %f\n",magma_wtime()-t1);
    t1 = magma_wtime();
#endif

    /*Transpose the gpu part of the matrix in/out of place*/

    if ((m == n) ){  //&& (m % 32 == 0) && (dA_LD%32 == 0)
         dAT = dA;
         dAT_LD= dA_LD;
       magmablas_dtranspose_inplace(m, dAT, dAT_LD); 
    } 
    else { 
        
      
     dm_max = m;
     dn_max = n;

    /*Make sure m and n are multiple of 32*/
     
     if(dm_max%32!=0) dm_max = ((dm_max + 31)/32)*32;
     if(dn_max%32!=0) dn_max = ((dn_max + 31)/32)*32;
     
     if (MAGMA_SUCCESS != magma_dmalloc(&dAT, dm_max*dn_max )) { 
        magma_free(dAP_set); 
        magma_free(dAP_get);
        *info = MAGMA_ERR_DEVICE_ALLOC; 
        return *info; 
     }

     dAT_LD = dn_max; 
     magmablas_dtranspose2( dAT, dAT_LD, dA, dA_LD, m, n );  
   }

#if (dbglevel >=1)
    printf("[DBG] Time First transposition: %f\n",magma_wtime()-t1);
    t1 = magma_wtime();
#endif

#if (dbglevel==10) 
    ca_dbg_printMat_transpose_gpu(m, n, dAT, dAT_LD,"matrix dAT to factorize"); 
#endif



     /* Compute the maximun number of steps*/
     mindim = min(m, n); 
     M      = (int) ceil( (double) m / nb); 
     N      = (int) ceil( (double) mindim / nb); /*N = n/nb*/


     /*Let the asynchronous algorithm begin*/ 
     
#if (dbglevel >=1)
     printf("Starting recursif code ... m:%d, n:%d, nb:%d, nbcores:%d, N:%d, A_N:%d\n", m, n, nb, nbcores, N, A_N); //Summary
#endif



     /*Initialize the scheduler*/ 
     magma_schedule_init(nbcores, 1); 


     K = 0; 
#ifdef USE_CALU
     /*initialize calu environment*/
     core_dtslu_alloc(panel_num_threads, A_m, nb);
     core_dtslu_init(panel_num_threads);

     /*Initialize rows indice: required*/
     for(I=0;I<A_m;I++) ipiv[I]=I;
#else
     /*initialize parallel recursif panel environment*/
     CORE_zgetrf_reclap_init();
#endif


     magma_schedule_set_task_priority(INT_MAX-1);

     /*Schedule the first panel factorization*/ 
#ifdef USE_CALU
     magma_insert_core_dtslu(A_m, nb, A(0,K), A_LD, ipiv(0), &iinfo, panel_num_threads, colptr(K));

     B_rows = (int) ceil((double) (M-K-1)/panel_num_threads);
     B_rows = max(B_rows,4); /*maximun of 4*/ 
     //B_rows = max(B_rows,1);

     for(I=K+1; I<=M-1; I+=B_rows){ 
     
        i_nrows = min(B_rows*nb, m-I*nb);
        magma_insert_core_dtrsm_gatherv('R', 'U', 'N', 'N', i_nrows, nb, c_one, A(0,K), A_LD, A(I,K), A_LD, colptr(K));
     }
#else
     magma_insert_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, ipiv(0), &iinfo, panel_num_threads, colptr(K));  
#endif
 
     /*Transfer the factorized panel to the GPU (transposition included)*/ 
     magma_insert_dsetmatrix_transpose(A_m, nb, A(0,K), A_LD, dAT(0,K), dAT_LD, dAP_set, dAP_LD, colptr(K), dAT(K,K)); 
 
 
#if (dbglevel==10) 
    magma_schedule_barrier(); 
    ca_dbg_printMat(m, nb, A(0,0), A_LD,"A(0,0)"); 
    ca_dbg_printMat_transpose_gpu(m, n, dAT, dAT_LD,"dA"); 
#endif 
 
     for(K=0;K<=N-1;K++){ 
     
          /*insert the coarse update of the trailing submatrix corresponding to panel K to the GPU, that is submatrix A[K+1:M, K+1+d-1:N]*/ 

          gpu_nrows = m - (K+1)*nb; 
          gpu_ncols = n - (K+1+A_N-1)*nb; 
 
          if(gpu_ncols >0) 
          { 
 
              /*NOTE: Here we work on the matrix transpose*/

              /*Set the priority max for the GPU computations*/
              magma_schedule_set_task_priority(INT_MAX);
             //// magma_schedule_set_task_priority(INT_MAX - N*K);

              /*schedule a swap of the trailing submatrix in the gpu using ipiv[K]*/ 
              /*dependency dAT((K+1)-1, (K+A_N)-1) = dAT(K, K+A_N-1) with previous dgemm*/
              magma_insert_dlaswp(gpu_ncols, dAT(K, K+A_N), dAT_LD, ONE, nb, ipiv(K), ONE, dAT(K, K+A_N-1)); /*non blocking*/                  
              //printf("debug barrier\n");
              //magma_schedule_barrier();
              magma_insert_dtrsm(MagmaRight,  MagmaUpper, MagmaNoTrans, MagmaUnit, gpu_ncols, nb, c_one, dAT(K,K), dAT_LD, dAT(K,K+A_N), dAT_LD);/*non blocking*/ 
 
              /* aij^T = aij^T - (lik.ukj)^T = aij^T - ukj^T.lik^T*/ 
              magma_insert_dgemm(MagmaNoTrans,MagmaNoTrans, gpu_ncols, gpu_nrows, nb, c_neg_one, dAT(K,K+A_N), dAT_LD, dAT(K+1,K), dAT_LD, c_one, dAT(K+1,K+A_N), dAT_LD);/*non blocking*/    
       
          } 
          
          /*iterate over the rest of the columns to update the trailing submatrix on the cpu*/ 
          for(J=K+1;J<=min(K+A_N-1, N-1);J++){ 
 
               ncols = min(nb, n - J*nb); 
 
               /*Set the priority max for column having the next panel (look ahead of deep 1),
               and process the rest of the update in a right looking way*/
               if(J==K+1)
                   magma_schedule_set_task_priority(INT_MAX -2 );
                  //// magma_schedule_set_task_priority(INT_MAX - N*K -1);
               else
                   magma_schedule_set_task_priority(INT_MAX -3 - J );//- N*K


               /*dependency colptr(J): make sure column J is sent from GPU, and all previous update was done*/
               magma_insert_core_dlaswp(ncols, A(K,J), A_LD, ONE, nb, ipiv(K), ONE, colptr(J)); 
 
               magma_insert_core_dtrsm('L', 'L', 'N', 'U', nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD, colptr(J)); 
 
             /*Compute the number of blocs rows to group together before the update. To avoid scheduling overhead.*/
              B_rows = (int) ceil((double) (M-K-1)/panel_num_threads);
              //B_rows = max(B_rows,4); /*maximun of 4*/ 
              //B_rows = max(B_rows,1);

               for(I=K+1; I<=M-1; I+=B_rows){ 
     
                    nrows = min(B_rows*nb, m-I*nb); 
                    
                    /*dep colptr(K):make sure the panel is not overwritten or swapped since dgemm use A[I,K]*/
                    /*dep colptr(J): Gather all dgemm on one column and create dependencies with previous dgemm and the next panel*/
                    magma_insert_core_dgemm('N','N', nrows, ncols, nb, c_neg_one, A(I,K), A_LD, A(K,J), A_LD, c_one, A(I,J), A_LD, colptr(K), colptr(J)); 
               } 

               if(J==K+1) 
               { 
                    /*Look ahead and insert the next panel*/ 
                    nrows = m - (K+1)*nb; 
                    ncols = min(nb, n - (K+1)*nb); 
 
                    /*Schedule the next panel factorization with maximum priority*/ 
                    magma_schedule_set_task_priority(INT_MAX -1);
#ifdef USE_CALU
                    magma_insert_core_dtslu(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, panel_num_threads, colptr(K+1));

                    B_rows = (int) ceil((double) (M-(K+1)-1)/panel_num_threads);
                    B_rows = max(B_rows,4); /*maximun of 4*/ 
                     //B_rows = max(B_rows,1);

                     for(I=K+2; I<=M-1; I+=B_rows){ 
     
                        i_nrows = min(B_rows*nb, m-I*nb);
                        magma_insert_core_dtrsm_gatherv('R', 'U', 'N', 'N', i_nrows, ncols, c_one, A(K+1,K+1), A_LD, A(I,K+1), A_LD, colptr(K+1));
                        //dtrsm("R", "U", "N", "N", &nrowPblock, &panel_NB, &dONE, &(A[M*pos+pos]), &LDA, &(A[lpos]), &LDA); //
                     }

#else
                   magma_insert_core_dgetrf_rec(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, panel_num_threads, colptr(K+1)); 
#endif 
 
                    /*Determine the upper part of the matrix done by the CPU on that column and send it to the GPU with the panel*/ 
                    U_K = max(0, K+1 - A_N +1); 
                    nrows = m - U_K*nb; 
 
                    /*Transfer the upper part of the matrix for that column and the factorized panel to the GPU*/ 
                    magma_insert_dsetmatrix_transpose(nrows, ncols, A(U_K, K+1), A_LD, dAT(U_K, K+1), dAT_LD, dAP_set, dAP_LD, colptr(K+1), dAT(K+1,K+1));
 
               } 
 
          } 
 
           /*Transfer asynchronously one column (column K+A_N) from the GPU to the CPU to balance work*/  
            /*Make sure this is inserted after all dgemm before it schedules to replace a current panel in case A_N< N*/
           if(K+A_N<N) { 
              ncols = min(nb, gpu_ncols); 
 
              magma_schedule_set_task_priority(INT_MAX);

              magma_insert_dgetmatrix_transpose(gpu_nrows, ncols, dAT(K+1,K+A_N), dAT_LD, A(K+1,K+A_N), A_LD, dAP_get, dAP_LD, colptr(K+A_N)); //blocking
           

          /*if A_N==1 there is no look-ahead, so insert the panel here*/
           if(A_N==1){
              /*Look ahead and insert the next panel*/ 
              nrows = m - (K+1)*nb; 
              ncols = min(nb, n - (K+1)*nb); 
              /*Schedule the next panel factorization with maximum priority*/ 
              magma_schedule_set_task_priority(INT_MAX -1);

#ifdef USE_CALU
                magma_insert_core_dtslu(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, panel_num_threads, colptr(K+1)); 

                B_rows = (int) ceil((double) (M-(K+1)-1)/panel_num_threads);
                B_rows = max(B_rows,4); /*maximun of 4*/ 
                //B_rows = max(B_rows,1);

                for(I=K+2; I<=M-1; I+=B_rows){ 
     
                    i_nrows = min(B_rows*nb, m-I*nb);
                    magma_insert_core_dtrsm_gatherv('R', 'U', 'N', 'N', i_nrows, ncols, c_one, A(K+1,K+1), A_LD, A(I,K+1), A_LD, colptr(K+1));
                    //dtrsm("R", "U", "N", "N", &nrowPblock, &panel_NB, &dONE, &(A[M*pos+pos]), &LDA, &(A[lpos]), &LDA); //
                }

#else
                 magma_insert_core_dgetrf_rec(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, panel_num_threads, colptr(K+1)); 
                 //magma_insert_core_dgetrf(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, colptr(K+1));
#endif

               
 
                /*Determine the upper part of the matrix done by the CPU on that column and send it to the GPU with the panel*/ 
                U_K = max(0, K+1 - A_N +1); 
                nrows = m - U_K*nb; 
 
                    ///magma_schedule_set_task_priority(INT_MAX);
                    /*Transfer the upper part of the matrix for that column and the factorized panel to the GPU*/ 
                magma_insert_dsetmatrix_transpose(nrows, ncols, A(U_K, K+1), A_LD, dAT(U_K, K+1), dAT_LD, dAP_set, dAP_LD, colptr(K+1), dAT(K+1,K+1));
           }
         }
#if (dbglevel==10)     
    magma_schedule_barrier(); 
    ca_dbg_printMat(m, A_n, A, A_LD,"A"); 
    ca_dbg_printMat_transpose_gpu(m, n, dAT, dAT_LD,"dA"); 
#endif 
         
     } //Step K done

 /*Wait for all thread termination*/
 magma_schedule_barrier(); 

     /*TODO: don't need quark here*/
     /*Perform a sequence of left swap on the matrix corresponding to the different panel*/ 
     for(K=1;K<=N-1;K++){ 
 
#if (dbglevel >=1)
    ca_trace_start();
#endif
        nrows = min(nb,m - K*nb); 
 
        ncols = min(K*nb,n); 

        /*dep dAT(K-1): Make sure the last swap is completed, and also the dgemm using the panel*/

       // magma_insert_dlaswp(ncols, dAT(K, 0), dAT_LD, ONE, nrows, ipiv(K), ONE, dAT(K-1,0)); 
        magmablas_dlaswp(ncols, dAT(K, 0), dAT_LD, ONE, nrows, ipiv(K), ONE);

#if (dbglevel >=1)
ca_trace_end_1gpu('W');
#endif
     } 
 
     
        
/*Shutdown the scheduler*/
     magma_schedule_delete();

/*update permutation vector indexes*/ 
     for(K=1;K<=N-1;K++){ 
 
        nrows = min(nb, n-K*nb); 
        for(J=0;J<=nrows-1;J++){ 
            ipiv[K*nb+J] += K*nb; 
        } 
     } 

#if dbglevel>=1
    printf("[DBG] Time Factorization:%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif


 /*No need for synchro, since dtranspose is blocking*/
   if (m == n) {
      magmablas_dtranspose_inplace(m, dAT, dAT_LD); //( m, dAT, dAT_LD ); 
      dA = dAT; 
   } 
   else { 
      magmablas_dtranspose2( dA, dA_LD, dAT, dAT_LD, n, m ); 
      magma_free( dAT ); 
   } 

#if dbglevel>=1
    printf("[DBG] Time Final in/out of place transpose:%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif


#ifdef USE_CALU
    core_dtslu_free();
#endif 

   magma_free( dAP_set ); 
   magma_free( dAP_get );

#if dbglevel>=1
    printf("[DBG] Time memory free (dAP):%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

#if (dbglevel==10)     
    ca_dbg_printMat_transpose_gpu(m, n, dA, dA_LD,"dA = LU"); 
#endif 


#if dbglevel>=1
    /*Finalize the tracing*/
    ca_dbg_trace_finalize();
    printf("[DBG] Time llog:%f\n",magma_wtime()-t1); 
#endif

    return *info; 
}   /* End of MAGMA_DGETRF_REC_ASYNC_WORK_GPU */


