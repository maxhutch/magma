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
#define  T(m) (work+(m)*(nb))
#define  W(k,n) &(local_work[(mt)*(n-1)+(k)])

void getro (char *trans, const magma_int_t m, const magma_int_t n, 
        const cuDoubleComplex *A, const magma_int_t LDA, 
        cuDoubleComplex *B, const magma_int_t LDB) 
{
  const cuDoubleComplex *Atmp;

  magma_int_t i, j;

  for (i=0; i<m; i++) {
    Atmp = A + i;
    for (j=0; j<n; j++) {

      if (trans[0] == 'C') {

        double *ap,*tp;
        cuDoubleComplex tmp;
        ap = (double *)&(*Atmp);
        tp = (double *)&tmp;
        tp[0] = ap[0];
        tp[1] = -ap[1];

        *B = tmp;

      } else {

        *B = *Atmp;

      }

      B += 1;
      Atmp += LDA;
    }
    B += (LDB - n);
  }
}

// task execution code
void SCHED_zlarfb(Quark* quark)
{
  magma_int_t M;
  magma_int_t N;
  magma_int_t MM;
  magma_int_t NN;
  magma_int_t IB;
  magma_int_t LDV;
  magma_int_t LDC;
  magma_int_t LDT;
  magma_int_t LDW;

  cuDoubleComplex *V;
  cuDoubleComplex *C;
  cuDoubleComplex *T;
  cuDoubleComplex **W;

  quark_unpack_args_13(quark, M, N, MM, NN, IB, V, LDV, C, LDC, T, LDT, W, LDW);

  if (M < 0) {
    printf("SCHED_zlarfb:  illegal value of M\n");
  }
  if (N < 0) {
    printf("SCHED_zlarfb:  illegal value of N\n");
  }
  if (IB < 0) {
    printf("SCHED_zlarfb:  illegal value of IB\n");
  }

  *W = (cuDoubleComplex*) malloc(LDW*MM*sizeof(cuDoubleComplex));

  getro(MagmaConjTransStr, MM, NN, C, LDC, *W, LDW);

  cuDoubleComplex c_one = MAGMA_Z_ONE;

  blasf77_ztrmm("r","l","n","u",
    &NN, &MM, &c_one, V, &LDV, *W, &LDW);

  magma_int_t K=M-MM;

  blasf77_zgemm(MagmaConjTransStr, "n", &NN, &MM, &K,
    &c_one, &C[MM], &LDC, &V[MM], &LDV, &c_one, *W, &LDW);

  blasf77_ztrmm("r", "u", "n", "n", 
    &NN, &MM, &c_one, T, &LDT, *W, &LDW);

}

// task execution code
void SCHED_zgeqrt(Quark* quark)
{
  magma_int_t M;
  magma_int_t N;
  magma_int_t IB;
  cuDoubleComplex *A;
  magma_int_t LDA;
  cuDoubleComplex *T;
  magma_int_t LDT;
  cuDoubleComplex *TAU;
  cuDoubleComplex *WORK;

  magma_int_t iinfo;
  magma_int_t lwork=-1;

  quark_unpack_args_9(quark, M, N, IB, A, LDA, T, LDT, TAU, WORK);

  if (M < 0) { 
    printf("SCHED_zgeqrt: illegal value of M\n");
  }

  if (N < 0) { 
    printf("SCHED_zgeqrt: illegal value of N\n");
  }

  if ((IB < 0) || ( (IB == 0) && ((M > 0) && (N > 0)) )) {
    printf("SCHED_zgeqrt: illegal value of IB\n");
  }

  if ((LDA < max(1,M)) && (M > 0)) {
    printf("SCHED_zgeqrt: illegal value of LDA\n");
  }

  if ((LDT < max(1,IB)) && (IB > 0)) {
    printf("SCHED_zgeqrt: illegal value of LDT\n");
  }

  lapackf77_zgeqrf(&M, &N, A, &LDA, TAU, WORK, &lwork, &iinfo);
  lwork=(magma_int_t)MAGMA_Z_REAL(WORK[0]);
  lapackf77_zgeqrf(&M, &N, A, &LDA, TAU, WORK, &lwork, &iinfo);

  lapackf77_zlarft("F", "C", &M, &N, A, &LDA, TAU, T, &LDT);

}

// task execution code
void SCHED_ztrmm(Quark *quark)
{
  magma_int_t m;
  magma_int_t n;
  cuDoubleComplex alpha;
  cuDoubleComplex *a;
  magma_int_t lda;
  cuDoubleComplex **b;
  magma_int_t ldb;
  cuDoubleComplex beta;
  cuDoubleComplex *c;
  magma_int_t ldc;
  cuDoubleComplex *work;

  magma_int_t j;

  magma_int_t one = 1;

  quark_unpack_args_11(quark, m, n, alpha, a, lda, b, ldb, beta, c, ldc, work);

  if (m < 0) {
    printf("SCHED_ztrmm:  illegal value of m\n");
  }

  if (n < 0) {
    printf("SCHED_ztrmm:  illegal value of n\n");
  }

  getro(MagmaConjTransStr, n, m, *b, ldb, work, m);

  blasf77_ztrmm("l", "l", "n", "u", 
    &m, &n, &alpha, a, &lda, work, &m);

  for (j = 0; j < n; j++)
  {
    blasf77_zaxpy(&m, &beta, &(work[j*m]), &one, &(c[j*ldc]), &one);
  }

}

// task execution code
static void SCHED_zgemm(Quark *quark)
{
  magma_int_t m;
  magma_int_t n;
  magma_int_t k;
  cuDoubleComplex alpha;
  cuDoubleComplex *a;
  magma_int_t lda;
  cuDoubleComplex **b;
  magma_int_t ldb;
  cuDoubleComplex beta;
  cuDoubleComplex *c;
  magma_int_t ldc;
  
  cuDoubleComplex *fake;

  magma_int_t dkdk;

  quark_unpack_args_13(quark, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, fake, dkdk);
      
  blasf77_zgemm("n", MagmaConjTransStr, 
    &m, &n, &k, &alpha, a, &lda, *b, &ldb, &beta, c, &ldc);

}

// wrapper around QUARK_Insert_Task .. makes code more readable at task insertion time
void QUARK_Insert_Task_zgemm(Quark *quark, Quark_Task_Flags *task_flags, 
  magma_int_t m, 
  magma_int_t n, 
  magma_int_t k,
  cuDoubleComplex alpha,
  cuDoubleComplex *a,
  magma_int_t lda,
  cuDoubleComplex **b,
  magma_int_t ldb,
  cuDoubleComplex beta,
  cuDoubleComplex *c,
  magma_int_t ldc,
  cuDoubleComplex *fake,
  char *dag_label,
  magma_int_t priority, 
  magma_int_t dkdk)
{

  QUARK_Insert_Task(quark, SCHED_zgemm, task_flags,
    sizeof(magma_int_t),           &m,     VALUE,
    sizeof(magma_int_t),           &n,     VALUE,
    sizeof(magma_int_t),           &k,     VALUE,
    sizeof(cuDoubleComplex),         &alpha, VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, a,      INPUT,
    sizeof(magma_int_t),           &lda,   VALUE,
    sizeof(cuDoubleComplex*),        b,      INPUT,
    sizeof(magma_int_t),           &ldb,   VALUE,
    sizeof(cuDoubleComplex),         &beta,  VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, c,      INOUT | LOCALITY,
    sizeof(magma_int_t),           &ldc,   VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, fake,   OUTPUT | GATHERV,
    sizeof(magma_int_t),           &priority, VALUE | TASK_PRIORITY,
    sizeof(magma_int_t),&dkdk,VALUE,
    strlen(dag_label)+1,   dag_label, VALUE | TASKLABEL,
6,                     "purple",   VALUE | TASKCOLOR,
    0);
}

// wrapper around QUARK_Insert_Task .. makes code more readable at task insertion time
void QUARK_Insert_Task_ztrmm(Quark *quark, Quark_Task_Flags *task_flags,
  magma_int_t m,
  magma_int_t n,
  cuDoubleComplex alpha, 
  cuDoubleComplex *a,
  magma_int_t lda,
  cuDoubleComplex **b,
  magma_int_t ldb,
  cuDoubleComplex beta,
  cuDoubleComplex *c,
  magma_int_t ldc,
  char *dag_label,
  magma_int_t priority)
{

  QUARK_Insert_Task(quark, SCHED_ztrmm, task_flags,
    sizeof(magma_int_t),           &m,     VALUE,
    sizeof(magma_int_t),           &n,     VALUE,
    sizeof(cuDoubleComplex),         &alpha, VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, a,      INPUT,
    sizeof(magma_int_t),           &lda,   VALUE,
    sizeof(cuDoubleComplex*),        b,      INPUT,
    sizeof(magma_int_t),           &ldb,   VALUE,
    sizeof(cuDoubleComplex),         &beta,  VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, c,      INOUT | LOCALITY,
    sizeof(magma_int_t),           &ldc,   VALUE,
    sizeof(cuDoubleComplex)*ldb*ldb, NULL,   SCRATCH,
    sizeof(magma_int_t),           &priority, VALUE | TASK_PRIORITY,
    strlen(dag_label)+1,   dag_label, VALUE | TASKLABEL,
    6,                     "orange",   VALUE | TASKCOLOR,
    0);
}

// wrapper around QUARK_Insert_Task .. makes code more readable at task insertion time
void QUARK_Insert_Task_zgeqrt(Quark *quark, Quark_Task_Flags *task_flags, 
  magma_int_t m,
  magma_int_t n,
  cuDoubleComplex *a,
  magma_int_t lda,
  cuDoubleComplex *t,
  magma_int_t ldt,
  cuDoubleComplex *tau,
  char *dag_label)
{

  magma_int_t priority = 1000;

  QUARK_Insert_Task(quark, SCHED_zgeqrt, task_flags,
    sizeof(magma_int_t),           &m,        VALUE,
    sizeof(magma_int_t),           &n,        VALUE,
    sizeof(magma_int_t),           &ldt,      VALUE,
    sizeof(cuDoubleComplex)*m*n,     a,         INOUT | LOCALITY,
    sizeof(magma_int_t),           &lda,      VALUE,
    sizeof(cuDoubleComplex)*ldt*ldt, t,         OUTPUT,
    sizeof(magma_int_t),           &ldt,      VALUE,
    sizeof(cuDoubleComplex)*ldt,     tau,       OUTPUT,
    sizeof(cuDoubleComplex)*ldt*ldt, NULL,      SCRATCH,
    sizeof(magma_int_t),           &priority, VALUE | TASK_PRIORITY,
    strlen(dag_label)+1,   dag_label, VALUE | TASKLABEL,
    6,                     "green",   VALUE | TASKCOLOR,
    0);

}

// wrapper around QUARK_Insert_Task .. makes code more readable at task insertion time
void QUARK_Insert_Task_zlarfb(Quark *quark, Quark_Task_Flags *task_flags,
  magma_int_t m,
  magma_int_t n,
  magma_int_t mm,
  magma_int_t nn,
  magma_int_t ib,
  cuDoubleComplex *v,
  magma_int_t ldv,
  cuDoubleComplex *c,
  magma_int_t ldc,
  cuDoubleComplex *t,
  magma_int_t ldt,
  cuDoubleComplex **w,
  magma_int_t ldw,
  char *dag_label,
  magma_int_t priority)

{

  QUARK_Insert_Task(quark, SCHED_zlarfb, task_flags,
    sizeof(magma_int_t),         &m,        VALUE,
    sizeof(magma_int_t),         &n,        VALUE,
    sizeof(magma_int_t),         &mm,       VALUE,
    sizeof(magma_int_t),         &nn,       VALUE,
    sizeof(magma_int_t),         &ib,       VALUE,
    sizeof(cuDoubleComplex)*m*n,   v,         INPUT,
    sizeof(magma_int_t),         &ldv,      VALUE,
    sizeof(cuDoubleComplex)*m*n,   c,         INPUT,
    sizeof(magma_int_t),         &ldc,      VALUE,
    sizeof(cuDoubleComplex)*ib*ib, t,         INPUT,
    sizeof(magma_int_t),         &ldt,      VALUE,
    sizeof(cuDoubleComplex*),      w,         OUTPUT | LOCALITY,
    sizeof(magma_int_t),         &ldw,      VALUE,
    sizeof(magma_int_t),         &priority, VALUE | TASK_PRIORITY,
    strlen(dag_label)+1, dag_label, VALUE | TASKLABEL,
    6,                   "cyan",    VALUE | TASKCOLOR,
    0);

}

extern "C" magma_int_t 
magma_zgeqrf_mc( magma_context *cntxt, magma_int_t *m, magma_int_t *n,
                 cuDoubleComplex *a,    magma_int_t *lda, cuDoubleComplex *tau,
                 cuDoubleComplex *work, magma_int_t *lwork,
                 magma_int_t *info)
{
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

    Purpose   
    =======   

    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:   
    A = Q * R.   

    Arguments   
    =========   
    CNTXT   (input) MAGMA_CONTEXT
            CNTXT specifies the MAGMA hardware context for this routine.   

    M       (input) magma_int_tEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) magma_int_tEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of min(m,n) elementary reflectors (see Further   
            Details).   

    LDA     (input) magma_int_tEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) magma_int_tEGER   
            The dimension of the array WORK.  LWORK >= N*NB. 

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    INFO    (output) magma_int_tEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   
    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   
    ====================================================================    */

  if (cntxt->num_cores == 1 && cntxt->num_gpus == 1)
  {
    //magma_int_t result = magma_zgeqrf(*m, *n, a, *lda, tau, work, *lwork, info);
    //return result;
  }

  magma_int_t i,j,l;

  magma_int_t ii=-1,jj=-1,ll=-1;

  Quark* quark = cntxt->quark;

  // DAG labels
  char sgeqrt_dag_label[1000]; 
  char slarfb_dag_label[1000];
  char strmm_dag_label[1000];
  char sgemm_dag_label[1000];

  *info = 0;

  cuDoubleComplex c_one = MAGMA_Z_ONE;
  cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

  magma_int_t nb = (cntxt->nb ==-1)? magma_get_zpotrf_nb(*n): cntxt->nb;

  magma_int_t lwkopt = *n * nb;
  work[0] = MAGMA_Z_MAKE( (double)lwkopt, 0 );

  long int lquery = *lwork == -1;

  // check input arguments
  if (*m < 0) {
    *info = -1;
  } else if (*n < 0) {
    *info = -2;
  } else if (*lda < max(1,*m)) {
    *info = -4;
  } else if (*lwork < max(1,*n) && ! lquery) {
    *info = -7;
  }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
  else if (lquery)
    return 0;

  magma_int_t k = min(*m,*n);
  if (k == 0) {
    work[0] = c_one;
    return 0;
  }

  magma_int_t nt = (((*n)%nb) == 0) ? (*n)/nb : (*n)/nb + 1;
  magma_int_t mt = (((*m)%nb) == 0) ? (*m)/nb : (*m)/nb + 1;

  cuDoubleComplex **local_work = (cuDoubleComplex**) malloc(sizeof(cuDoubleComplex*)*(nt-1)*mt);
  memset(local_work, 0, sizeof(cuDoubleComplex*)*(nt-1)*mt);

  magma_int_t priority;

  // traverse diagonal blocks
  for (i = 0; i < k; i += nb) {

    ii++;

    jj = ii;

    sprintf(sgeqrt_dag_label, "GEQRT %d",ii);

    // factor diagonal block, also compute T matrix
    QUARK_Insert_Task_zgeqrt(quark, 
      0, (*m)-i, min(nb,(*n)-i), A(i,i), *lda, T(i), nb, &tau[i], sgeqrt_dag_label);

    if (i > 0) {

      priority = 100;

      // update panels in a left looking fashion
      for (j = (i-nb) + (2*nb); j < *n; j += nb) { 

        jj++;

        ll = ii-1;

        sprintf(slarfb_dag_label, "LARFB %d %d",ii-1, jj);

        // perform part of update
        QUARK_Insert_Task_zlarfb(quark, 0, 
          (*m)-(i-nb), min(nb,(*n)-(i-nb)), min(nb,(*m)-(i-nb)), min(nb,(*n)-j), nb, 
          A(i-nb,i-nb), *lda, A(i-nb,j), *lda, T(i-nb), nb, W(ii-1,jj), nb, slarfb_dag_label, priority);

        sprintf(strmm_dag_label, "TRMM %d %d",ii-1, jj);

        // perform more of update
        QUARK_Insert_Task_ztrmm(quark, 0, min(nb,(*m)-(i-nb)), min(nb,(*n)-j), c_neg_one, 
          A(i-nb,i-nb), *lda, W(ii-1,jj), nb, c_one, A(i-nb,j), *lda, strmm_dag_label, priority);

          sprintf(sgemm_dag_label, "GEMM %d %d %d",ii-1, jj, ll);

          // finish update
          QUARK_Insert_Task_zgemm(quark, 0, (*m)-i, min(nb,(*n)-j), min(nb,(*n)-(i-nb)), c_neg_one,
            A(i,i-nb), *lda, W(ii-1,jj), nb, c_one, A(i,j), *lda, A(i,j), sgemm_dag_label, priority, jj);

      }

    }

    j = i + nb;

    jj = ii;

    // handle case of short wide rectangular matrix
    if (j < (*n)) {

      priority = 0;

      jj++;

      ll = ii;

      sprintf(slarfb_dag_label, "LARFB %d %d",ii, jj);

      // perform part of update
      QUARK_Insert_Task_zlarfb(quark, 0, 
        (*m)-i, min(nb,(*n)-i), min(nb,(*m)-i), min(nb,(*n)-j), nb, 
        A(i,i), *lda, A(i,j), *lda, T(i), nb, W(ii,jj), nb, slarfb_dag_label, priority);

      sprintf(strmm_dag_label, "TRMM %d %d",ii, jj);

      // perform more of update 
      QUARK_Insert_Task_ztrmm(quark, 0, min(nb,(*m)-i), min(nb,(*n)-j), c_neg_one, 
        A(i,i), *lda, W(ii,jj), nb, c_one, A(i,j), *lda, strmm_dag_label, priority);

        sprintf(sgemm_dag_label, "GEMM %d %d %d",ii, jj, ll);

        // finish update
        QUARK_Insert_Task_zgemm(quark, 0, (*m)-i-nb, min(nb,(*n)-j), min(nb,(*n)-i), c_neg_one,
          A(i+nb,i), *lda, W(ii,jj), nb, c_one, A(i+nb,j), *lda, A(i+nb,j), sgemm_dag_label, priority, jj);

    }

  }

  // wait for all tasks to finish executing
  QUARK_Barrier(quark);
  
  // free memory
  for(k = 0 ; k < (nt-1)*mt; k++) {
    if (local_work[k] != NULL) {
      free(local_work[k]);
    }
  }
  free(local_work);
  
}

#undef A
#undef T
#undef W

