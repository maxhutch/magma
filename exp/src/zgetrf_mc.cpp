/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define  A(m,n) (a+(n)*(*lda)+(m))

/* Task execution code */
static void SCHED_zgemm(Quark* quark)
{
  int M;
  int N;
  int K;
  cuDoubleComplex *A1;
  int LDA;
  cuDoubleComplex *A2;
  cuDoubleComplex *A3;

  cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
  cuDoubleComplex one = MAGMA_Z_ONE;
    
  quark_unpack_args_7(quark, M, N, K, A1, LDA, A2, A3);

  blasf77_zgemm("n", "n", 
    &M, &N, &K, &mone, A1, &LDA, A2, &LDA, &one, A3, &LDA);

}

/* Task execution code */
static void SCHED_panel_update(Quark* quark)
{
  int N;
  cuDoubleComplex *A1;
  int LDA;
  int K2;
  int *IPIV;
  cuDoubleComplex *A2;
  int M;
  int K;
  cuDoubleComplex *A3;
  cuDoubleComplex *A4;

  int ione=1;
  cuDoubleComplex mone = MAGMA_Z_NEG_ONE;
  cuDoubleComplex one = MAGMA_Z_ONE;
    
  quark_unpack_args_10(quark, N, A1, LDA, K2, IPIV, A2, M, K, A3, A4);

  lapackf77_zlaswp(&N, A1, &LDA, &ione, &K2, IPIV, &ione); 

  blasf77_ztrsm("l", "l", "n", "u",
    &K2, &N, &one, A2, &LDA, A1, &LDA);

  if (M > 0) {

  blasf77_zgemm("n","n", 
    &M, &N, &K, &mone, A3, &LDA, A1, &LDA, &one, A4, &LDA);

  }

}

/* Task execution code */
void SCHED_zgetrf(Quark* quark)
{
  int M;
  int N;
  cuDoubleComplex *A;
  int LDA;
  int *IPIV;

  int *iinfo;

  int info;

  quark_unpack_args_5(quark, M, N, A, LDA, IPIV);

  lapackf77_zgetrf(&M, &N, A, &LDA, IPIV, &info); 

  if (info > 0) {
    iinfo[1] = iinfo[0] + info;
  }

}

/* Task execution code */
void SCHED_zlaswp(Quark* quark)
{
  int N;
  cuDoubleComplex *A;
  int LDA;
  int K2;
  int *IPIV;

  int ione=1;

  quark_unpack_args_5(quark, N, A, LDA, K2, IPIV);

  lapackf77_zlaswp(&N, A, &LDA, &ione, &K2, IPIV, &ione); 

}

extern "C" magma_int_t 
magma_zgetrf_mc(magma_context *cntxt,
        int *m, int *n,
        cuDoubleComplex *a, int *lda,
        int *ipiv, int *info)
{
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

    Purpose   
    =======   
    ZGETRF computes an LU factorization of a general COMPLEX_16 
    M-by-N matrix A using partial pivoting with row interchanges.   

    The factorization has the form   
       A = P * L * U   
    where P is a permutation matrix, L is lower triangular with unit   
    diagonal elements (lower trapezoidal if m > n), and U is upper   
    triangular (upper trapezoidal if m < n).   

    This is the right-looking Level 3 BLAS version of the algorithm.   

    Arguments   
    =========   
    CNTXT   (input) MAGMA_CONTEXT
            CNTXT specifies the MAGMA hardware context for this routine.   

    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the M-by-N matrix to be factored.   
            On exit, the factors L and U from the factorization   
            A = P*L*U; the unit diagonal elements of L are not stored.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    IPIV    (output) INTEGER array, dimension (min(M,N))   
            The pivot indices; for 1 <= i <= min(M,N), row i of the   
            matrix was interchanged with row IPIV(i).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization   
                  has been completed, but the factor U is exactly   
                  singular, and division by zero will occur if it is used   
                  to solve a system of equations.   
    =====================================================================    */

    if (cntxt->num_cores == 1 && cntxt->num_gpus == 1)
      {
    //int result = magma_zgetrf(*m, *n, a, *lda, ipiv, info);
    //return result;
      }
    
    int EN_BEE   = cntxt->nb;
    Quark* quark = cntxt->quark;

    int i,j,l;
    int ii,jj,ll;

    void *fakedep;

    int ione=1;

    cuDoubleComplex fone = MAGMA_Z_ONE;
    cuDoubleComplex mone = MAGMA_Z_NEG_ONE;

    int M,N,MM,NN,MMM,K;

    int priority=0;

    *info = 0;
    
    int nb = (EN_BEE==-1)? magma_get_zpotrf_nb(*n): EN_BEE;

    /* Check arguments */
    if (*m < 0) {
      *info = -1;
    } else if (*n < 0) {
      *info = -2;
    } else if (*lda < max(1,*m)) {
      *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    
    int k = min(*m,*n);

    int iinfo[2];
    iinfo[1] = 0;

    char label[10000];
    
    ii = -1;
    
    /* Loop across diagonal blocks */
    for (i = 0; i < k; i += nb) 
      {
    ii++;

    jj = -1;

    priority = 10000 - ii;

    /* Update panels in left looking fashion */
    for (j = 0; j < i; j += nb) 
      { 
        jj++;

        NN=min(nb,(*n)-i);
        MM=min(nb,(*m)-j);

        l = j + nb;

        MMM = min(nb,(*m)-l);

        sprintf(label, "UPDATE %d %d", ii, jj);
        
        QUARK_Insert_Task(quark, SCHED_panel_update, 0,
                  sizeof(int),             &NN,      VALUE,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INOUT,
                  sizeof(int),             lda,        VALUE,
                  sizeof(int),             &MM,      VALUE,
                  sizeof(cuDoubleComplex)*nb,        &ipiv[j], INPUT,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(j,j),   INPUT,
                  sizeof(int),             &MMM,     VALUE,
                  sizeof(int),             &nb,      VALUE,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
                  sizeof(int),             &priority,VALUE | TASK_PRIORITY,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT,
                  strlen(label)+1,         label,    VALUE | TASKLABEL,
                  5,                       "cyan",   VALUE | TASKCOLOR,
                  0);
        
        ll = jj + 1;
        
        /* Split gemm into tiles */
        for (l = j + (2*nb); l < (*m); l += nb) 
          {
        ll++;
        
        MMM = min(nb,(*m)-l);
        
        fakedep = (void *)(intptr_t)(j+1);
        
        sprintf(label, "GEMM %d %d %d", ii, jj, ll);
        
        QUARK_Insert_Task(quark, SCHED_zgemm, 0,
                  sizeof(int),             &MMM,     VALUE,
                  sizeof(int),             &NN,      VALUE,
                  sizeof(int),             &nb,      VALUE,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
                  sizeof(int),             lda,        VALUE,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INPUT,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
                  sizeof(int),             &priority,VALUE | TASK_PRIORITY,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT | GATHERV,
                  sizeof(void*),           fakedep,  OUTPUT | GATHERV,
                  strlen(label)+1,         label,    VALUE | TASKLABEL,
                  5,                       "blue",   VALUE | TASKCOLOR,
                  0);
        
          }  
        
      }
    
    M=(*m)-i;
    N=min(nb,(*n)-i);
    
    iinfo[0] = i;
    
    sprintf(label, "GETRF %d", ii);
    
    QUARK_Insert_Task(quark, SCHED_zgetrf, 0,
              sizeof(int),             &M,       VALUE,
              sizeof(int),             &N,       VALUE,
              sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   INOUT,
              sizeof(int),             lda,        VALUE,
              sizeof(cuDoubleComplex)*nb,        &ipiv[i], OUTPUT,
              sizeof(int),             iinfo,    OUTPUT,
              sizeof(int),             &priority,VALUE | TASK_PRIORITY,
              strlen(label)+1,         label,    VALUE | TASKLABEL,
              6,                       "green",  VALUE | TASKCOLOR,
              0);
    
      }
    
    K = (*m)/nb;
    
    if ((K*nb)==(*m)) {
      ii = K - 1;
      K = *m;
    } else {
      ii = k;
      K = (K+1)*nb;
    }
    
    priority = 0;
    
    /* If n > m */
    for (i = K; i < (*n); i += nb) 
      {
    ii++;
    
    jj = -1;
    
    /* Update remaining panels in left looking fashion */
    for (j = 0; j < (*m); j += nb) 
      { 
        jj++;

        NN=min(nb,(*n)-i);
        MM=min(nb,(*m)-j);
        
        l = j + nb;
        
        MMM = min(nb,(*m)-l);
        
        sprintf(label, "UPDATE %d %d", ii, jj);
        
        QUARK_Insert_Task(quark, SCHED_panel_update, 0,
                  sizeof(int),             &NN,      VALUE,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INOUT,
                  sizeof(int),             lda,        VALUE,
                  sizeof(int),             &MM,      VALUE,
                  sizeof(cuDoubleComplex)*nb,        &ipiv[j], INPUT,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(j,j),   INPUT,
                  sizeof(int),             &MMM,     VALUE,
                  sizeof(int),             &nb,      VALUE,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
                  sizeof(int),             &priority,VALUE | TASK_PRIORITY,
                  sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT,
                  strlen(label)+1,         label,    VALUE | TASKLABEL,
                  5,                       "cyan",   VALUE | TASKCOLOR,
                  0);
        
        ll = jj + 1;
        
        /* Split gemm into tiles */
        for (l = j + (2*nb); l < (*m); l += nb) {
          
          ll++;
          
          MMM = min(nb,(*m)-l);
          
          fakedep = (void *)(intptr_t)(j+1);
          
          sprintf(label, "GEMM %d %d %d", ii, jj, ll);
          
          QUARK_Insert_Task(quark, SCHED_zgemm, 0,
                sizeof(int),             &MMM,     VALUE,
                sizeof(int),             &NN,      VALUE,
                sizeof(int),             &nb,      VALUE,
                sizeof(cuDoubleComplex)*(*m)*(*n), A(l,j),   INPUT,
                sizeof(int),             lda,        VALUE,
                sizeof(cuDoubleComplex)*(*m)*(*n), A(j,i),   INPUT,
                sizeof(cuDoubleComplex)*(*m)*(*n), A(l,i),   INOUT,
                sizeof(int),             &priority,VALUE | TASK_PRIORITY,
                sizeof(cuDoubleComplex)*(*m)*(*n), A(i,i),   OUTPUT | GATHERV,
                sizeof(void*),           fakedep,  OUTPUT | GATHERV,
                strlen(label)+1,         label,    VALUE | TASKLABEL,
                5,                       "blue",   VALUE | TASKCOLOR,
                0);
          
        }
        
      }
    
      }
    
    ii = -1;
    
    /* Swap behinds */
    for (i = 0; i < k; i += nb) {
      
      ii++;
      
      jj = -1;
      
      MM = min(nb,(*m)-i);
      MM = min(MM,(*n)-i);
      
      for (j = 0; j < i; j += nb) {
    
    jj++;
    
    fakedep = (void *)(intptr_t)(j+1);
    
    sprintf(label, "LASWPF %d %d", ii, jj);
    
    QUARK_Insert_Task(quark, SCHED_zlaswp, 0,
              sizeof(int),             &nb,       VALUE,
              sizeof(cuDoubleComplex)*(*m)*(*n), A(i,j),    INOUT,
              sizeof(int),             lda,         VALUE,
              sizeof(int),             &MM,       VALUE,
              sizeof(cuDoubleComplex)*nb,        &ipiv[i],  INPUT,
              sizeof(int),             &priority, VALUE | TASK_PRIORITY,
              sizeof(void*),           fakedep,   INPUT,
              sizeof(cuDoubleComplex)*(*m)*(*n), A(i+nb,j), OUTPUT,
              strlen(label)+1,         label,     VALUE | TASKLABEL,
              7,                       "purple",  VALUE | TASKCOLOR,
              0);
    
      }
      
    }
    
    /* Synchronization point */
    QUARK_Barrier(quark);
    
    /* Fix pivot */
    ii = -1;

    for (i = 0; i < k; i +=nb) {
      ii++;
      for (j = 0; j < min(nb,(k-i)); j++) {
    ipiv[ii*nb+j] += ii*nb;
      } 
    } 
    
    QUARK_Barrier(quark);
    
}

#undef A

