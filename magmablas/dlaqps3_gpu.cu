/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:45 2013

*/

#include "common_magma.h"
#include <cblas.h>
#include "magma.h"

#define PRECISION_d


//#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 512
//#else
//   #define BLOCK_SIZE 768
//#endif


__global__ void magma_dgemv_kernel3(int m, const double * __restrict__ V, int ldv,
                                    double *c, double *dwork,
                                    double *tau);

/* --------------------------------------------------------------------------- */

template< int n >
__device__ void sum_reduce( /*int n,*/ int i, double* x )
{
    __syncthreads();
   // if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce

/* --------------------------------------------------------------------------- */

#define BLOCK_SIZE1 192

__global__ void
magma_dswap_gemv_kernel(int m, int rk, int n, const double * __restrict__ V, int ldv,
                     const double * __restrict__ x, int ldx, double *c, double *b)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE1 * blockIdx.x;
    double lsum, tmp;

    V += j;

    lsum = MAGMA_D_ZERO;
    if (j < m){
       tmp  = b[j];
       b[j] = c[j];
       if (j>=rk) 
          for(int k=0; k<n; k++)
              lsum += MAGMA_D_MUL( V[k*ldv], MAGMA_D_CNJG(x[k*ldx]));

       c[j] = tmp - lsum;
    }
}

__global__ void
magma_dgemv_kernel(int m, int n, const double * __restrict__ V, int ldv,
                     const double * __restrict__ x, double *b, double *c)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE1 * blockIdx.x;
    double lsum;

    V += j;

    lsum = MAGMA_D_ZERO;
    if (j < m){
        for(int k=0; k<n; k++)
            lsum += MAGMA_D_MUL( V[k*ldv], x[k]);

       c[j] = b[j] - lsum;
    }
}


__global__
void magma_dscale_kernel(int n, double* dx0,
                         double *dtau, double *dxnorm, double* dAkk)
{
   const int i = threadIdx.x;
   double tmp;
   __shared__ double scale;

   /* === Compute the norm of dx0 === */
   double *dx = dx0;
   __shared__ double sum[ BLOCK_SIZE ];
   double re, lsum;

   lsum = 0;
   for( int k = i; k < n; k += BLOCK_SIZE ) {

        #if (defined(PRECISION_s) || defined(PRECISION_d))
             re = dx[k];
             lsum += re*re;
        #else
             re = MAGMA_D_REAL( dx[k] );
             double im = MAGMA_D_IMAG( dx[k] );
             lsum += re*re + im*im;
        #endif
   }
   sum[i] = lsum;
   sum_reduce< BLOCK_SIZE >( i, sum );

   /* === Compute the scaling factor === */
   if (i==0){
            double beta = sqrt(sum[0]);
            if ( beta == 0 ) {
              *dtau = MAGMA_D_ZERO;
            }
            else {
               tmp = dx0[0];
#if (defined(PRECISION_s) || defined(PRECISION_d))
               beta  = -copysign( beta, tmp );

               // todo: deal with badly scaled vectors (see lapack's larfg)
               *dtau    = (beta - tmp) / beta;
               *dAkk    = beta;

               scale = 1. / (tmp - beta);
#else
               double alphar =  MAGMA_D_REAL(tmp), alphai = MAGMA_D_IMAG(tmp);
               beta  = -copysign( beta, alphar );

               // todo: deal with badly scaled vectors (see lapack's larfg)
               *dtau = MAGMA_D_MAKE((beta - alphar)/beta, -alphai/beta);
               *dAkk = MAGMA_D_MAKE(beta, 0.);

               tmp = MAGMA_D_MAKE( alphar - beta, alphai);
               scale = MAGMA_D_DIV( MAGMA_D_ONE, tmp);
#endif
            }
   }

   __syncthreads();

   /* === Scale the vector === */
   for(int j=i; j<n; j+=BLOCK_SIZE)
      dx0[j] = MAGMA_D_MUL(dx0[j], scale);

   /* === Make temporary the first element to 1; value is stored in dAkk === */
   if (i==0)
     dx0[0] = MAGMA_D_ONE;
}


template< int n >
__device__ void zsum_reduce( /*int n,*/ int i, double* x )
{
    __syncthreads();
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}

__global__ void
magma_dgemv_kernel1(int m, double *tau, const double * __restrict__ V, int ldv,
                    const double * __restrict__ c,
                    double *dwork)
{
        const int i = threadIdx.x;
        const double *dV = V + (blockIdx.x) * ldv;

        __shared__ double sum[ BLOCK_SIZE ];
        double lsum;

        /*  lsum := v' * C  */
        lsum = MAGMA_D_ZERO;
        for( int j = i; j < m; j += BLOCK_SIZE )
           lsum += MAGMA_D_MUL( MAGMA_D_CNJG( dV[j] ), c[j] );

        sum[i] = lsum;
        zsum_reduce< BLOCK_SIZE >( i, sum );

        __syncthreads();
        if (i==0)
           dwork [blockIdx.x] = (*tau)*sum[0];
}


#define BLOCK_SIZE2 192
#if (defined(PRECISION_z) || defined(PRECISION_d))
  #define TOL 1.e-8
#else
  #define TOL 1.e-4
#endif

__global__ void
magma_dgemv_kernel_adjust(int n, int k, double * A, int lda, 
                          double *B, int ldb, double *C,
                          double *xnorm, double *xnorm2, double *Akk, int *lsticc, int *lsticcs)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE2 * blockIdx.x;
    double sum;
    double temp, oldnorm;

    if (j<n) {
      B += j;
      sum = MAGMA_D_CNJG( B[(k-1)*ldb] );
      // sum = MAGMA_D_ZERO;
      for(int m=0; m<k-1; m++) {
         sum += MAGMA_D_MUL( MAGMA_D_CNJG( B[m*ldb] ), A[m*lda] );
      }
      C[j*lda] -= sum;

      oldnorm = xnorm[j];
      temp = MAGMA_D_ABS( C[j*lda] ) / oldnorm;
      temp  = (1.0 + temp) * (1.0 - temp);
      temp  = oldnorm * sqrt(temp);

      xnorm[j] = temp;

      // Below 'j' was 'i'; was that a bug?
      double temp2 = xnorm[j] / xnorm2[j];
      temp2 = temp*(temp2 * temp2);
      if (temp2 <= TOL){
         *lsticc = 1;
         lsticcs[j] = 1;
      }
    }

   if (j==0)
       A[(k-1)*lda] = *Akk;
  
/*
    __syncthreads();
    // Check if the norm has to be recomputed 
    if (blockIdx.x==0) {
       //if (2.*temp < oldnorm) {
           //printf("recompute norm\n");
           double *dx = C+blockIdx.x*lda+1;
           __shared__ double sum[ BLOCK_SIZE2 ];
           double re, lsum;
 
           // get norm of dx
           lsum = 0;
           for( int k = i; k < n1; k += BLOCK_SIZE2 ) {

               #if (defined(PRECISION_s) || defined(PRECISION_d))
                   re = dx[k];
                   lsum += re*re;
               #else
                   re = MAGMA_D_REAL( dx[k] );
                   double im = MAGMA_D_IMAG( dx[k] );
                   lsum += re*re + im*im;
               #endif
           }
           sum[i] = lsum;
           sum_reduce< BLOCK_SIZE2 >( i, sum );

           if (i==0){
             printf("adjusted = %f recomputed = %f\n", xnorm[blockIdx.x], sqrt(sum[0])); 
             xnorm[blockIdx.x] = sqrt(sum[0]);
           }
      }
 //   }
*/
}

__global__ void
magmablas_dnrm2_check_kernel(int m, double *da, int ldda, 
                              double *dxnorm, double *dxnorm2, 
                              int *dlsticc, int *dlsticcs)
{
    const int i = threadIdx.x;
    double *dx = da + blockIdx.x * ldda;

    __shared__ double sum[ BLOCK_SIZE ];
    double re, lsum;

    if (blockIdx.x == 0 && i==0)
       *dlsticc = 0;

    // get norm of dx only if lsticc[blockIdx] != 0
    if( dlsticcs[blockIdx.x] == 0 ) 
        return;
    else
        dlsticcs[blockIdx.x] = 0;

    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {

#if (defined(PRECISION_s) || defined(PRECISION_d))
        re = dx[j];
        lsum += re*re;
#else
        re = MAGMA_D_REAL( dx[j] );
        double im = MAGMA_D_IMAG( dx[j] );
        lsum += re*re + im*im;
#endif

    }
    sum[i] = lsum;
    sum_reduce< BLOCK_SIZE >( i, sum );

    if (i==0){
      dxnorm[blockIdx.x]  = sqrt(sum[0]);
      dxnorm2[blockIdx.x] = sqrt(sum[0]);
    }
}


/* --------------------------------------------------------------------------- */



extern "C" magma_int_t
magma_dlaqps3_gpu(magma_int_t m, magma_int_t n, magma_int_t offset,
             magma_int_t nb, magma_int_t *kb,
             double *A,  magma_int_t lda,
             magma_int_t *jpvt, double *tau, 
             double *vn1, double *vn2,
             double *auxv,
             double *F,  magma_int_t ldf)
{
/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    DLAQPS computes a step of QR factorization with column pivoting
    of a real M-by-N matrix A by using Blas-3.  It tries to factorize
    NB columns from A starting from the row OFFSET+1, and updates all
    of the matrix with Blas-3 xGEMM.

    In some cases, due to catastrophic cancellations, it cannot
    factorize NB columns.  Hence, the actual number of factorized
    columns is returned in KB.

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. N >= 0

    OFFSET  (input) INTEGER
            The number of rows of A that have been factorized in
            previous steps.

    NB      (input) INTEGER
            The number of columns to factorize.

    KB      (output) INTEGER
            The number of columns actually factorized.

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    JPVT    (input/output) INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    TAU     (output) DOUBLE PRECISION array, dimension (KB)
            The scalar factors of the elementary reflectors.

    VN1     (input/output) DOUBLE PRECISION array, dimension (N)
            The vector with the partial column norms.

    VN2     (input/output) DOUBLE PRECISION array, dimension (N)
            The vector with the exact column norms.

    AUXV    (input/output) DOUBLE PRECISION array, dimension (NB)
            Auxiliar vector.

    F       (input/output) DOUBLE PRECISION array, dimension (LDF,NB)
            Matrix F' = L*Y'*A.

    LDF     (input) INTEGER
            The leading dimension of the array F. LDF >= max(1,N).

    =====================================================================    */
    
#define  A(i, j) (A  + (i) + (j)*(lda ))
#define  F(i, j) (F  + (i) + (j)*(ldf ))

    double c_zero    = MAGMA_D_MAKE( 0.,0.);
    double c_one     = MAGMA_D_MAKE( 1.,0.);
    double c_neg_one = MAGMA_D_MAKE(-1.,0.);
    magma_int_t ione = 1;
    
    magma_int_t i__1, i__2;
    
    magma_int_t k, rk;
    double tauk;
    magma_int_t pvt, itemp;

    double *dAkk = auxv;
    auxv+=1;

    int lsticc, *dlsticc, *dlsticcs;
    magma_malloc( (void**) &dlsticcs, (n+1)*sizeof(int) );
    cudaMemset( dlsticcs, 0, (n+1)*sizeof(int) );
    dlsticc = dlsticcs + n;
 
    // double tol3z = magma_dsqrt( lapackf77_dlamch("Epsilon"));

    lsticc = 0;
    k = 0;
    while( k < nb && lsticc == 0 ) {
        rk = offset + k;
        
        /* Determine ith pivot column and swap if necessary */
        pvt = k - 1 + magma_idamax( n-k, &vn1[k], ione );

        if (pvt != k) {
            magmablas_dswap( k, F(pvt,0), ldf, F(k,0), ldf);
            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            #if (defined(PRECISION_d) || defined(PRECISION_z))
                //magma_dswap( 1, &vn1[pvt], 1, &vn1[k], 1 );
                //magma_dswap( 1, &vn2[pvt], 1, &vn2[k], 1 );
                magma_dswap( 2, &vn1[pvt], n+offset, &vn1[k], n+offset);
            #else
                //magma_sswap( 1, &vn1[pvt], 1, &vn1[k], 1 );
                //magma_sswap( 1, &vn2[pvt], 1, &vn2[k], 1 );
                magma_sswap(2, &vn1[pvt], n+offset, &vn1[k], n+offset);
            #endif
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'  */
        magma_dswap_gemv_kernel<<< (m + BLOCK_SIZE1-1) / BLOCK_SIZE1, BLOCK_SIZE1, 0, magma_stream >>> 
                              ( m, rk, k, A(0, 0), lda, F(k,  0), ldf, A(0, k), A(0,pvt));
                                 
        /*  Generate elementary reflector H(k). */
        magma_dscale_kernel<<< 1, BLOCK_SIZE, 0, magma_stream >>>
               (m-rk, A(rk, k),   &tau[k], &vn1[k], dAkk);
        // printf("m-rk = %d\n", m-rk);

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1) {
            magma_dgetvector( 1, &tau[k], 1, &tauk, 1 );
            magmablas_dgemv( MagmaTrans, m-rk, n,
                         tauk,   A( rk,  0 ), lda,
                                 A( rk,  k   ), 1,
                         c_zero, auxv, 1 );
            if (k==0) 
               magmablas_dlacpy(MagmaUpperLower, n-k-1, 1, auxv+k+1, n-k-1, F( k+1, k   ), n-k-1);
        }
        
        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). 
           F(1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K)
                    := tau(K)(A(RK:M,K+1:N)' - F(1:N,1:K-1)*A(RK:M,1:K-1)') A(RK:M,K)  
           so, F is (updated A)*V */
        if (k > 0) {
            /* I think we only need stricly lower-triangular part */
            magma_dgemv_kernel<<< (n-k-1 + BLOCK_SIZE1 -1)/BLOCK_SIZE1, BLOCK_SIZE1, 0, magma_stream >>>
                       (n-k-1, k, F(k+1,0), ldf, auxv, auxv+k+1, F(k+1,k));
        }
        
        /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            /* left-looking update of rows,                     *
             * since F=A'v with original A, so no right-looking */
            magma_dgemv_kernel_adjust<<<(n-k-1 + BLOCK_SIZE2-1)/BLOCK_SIZE2, BLOCK_SIZE2, 0, magma_stream>>>
                           (n-k-1, k+1, A(rk, 0  ), lda, F(k+1,0  ), ldf, A(rk, k+1),
                           &vn1[k+1], &vn2[k+1], dAkk, dlsticc, dlsticcs);
            magma_getmatrix(1,1, sizeof(int), dlsticc, 1, &lsticc, 1); 
 
            // TTT: force not to recompute; has to be finally commented 
            if ( nb<3 )
            lsticc = 0; 

            // printf("k=%d n-k = %d\n", k, n-k);
            // forcing recompute works! - forcing it requires changing dlsticcs as well, e.g.,
            // can be done in the kernel directly (magmablas_dnrm2_check_kernel)
            // if (k==16) lsticc = 1;
        }
        
        /* Update partial column norms. */
/*
        if (rk < min(m, n+offset)-1){
           magmablas_dnrm2_row_check_adjust(n-k-1, tol3z, &vn1[k+1], 
                                             &vn2[k+1], A(rk,k+1), lda, lsticcs); 
        }

        #if defined(PRECISION_d) || defined(PRECISION_z)
            magma_dgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
        #else
            magma_sgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
        #endif
*/

        if (k>=n-1)
           magmablas_dlacpy(MagmaUpperLower, 1, 1, dAkk, 1, A(rk, k), 1);

        ++k;
    }
    // leave k as the last column done
    --k;
    *kb = k + 1;
    rk = offset + *kb - 1;

    //printf("actually factored = %d",*kb);

    /* Apply the block reflector to the rest of the matrix:
       A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - 
                                  A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'  */
    if (*kb < min(n, m - offset)-1) {
        i__1 = m - rk - 1;
        i__2 = n - *kb;
        
        magma_dgemm( MagmaNoTrans, MagmaTrans, i__1, i__2, *kb,
                     c_neg_one, A(rk+1, 0  ), lda,
                                F(*kb,  0  ), ldf,
                     c_one,     A(rk+1, *kb), lda );
    }

    /* Recomputation of difficult columns. */
    if( lsticc > 0 ) {
        printf( " -- recompute dnorms --\n" );
        //magmablas_dnrm2_check(m-rk-1, n-*kb, A(rk+1,rk+1), lda,
        //                       &vn1[rk+1], &vn2[rk+1], dlsticcs);
       
        // There is a bug when we get to recompute  
        magmablas_dnrm2_check_kernel<<< n-*kb, BLOCK_SIZE >>>
                     ( m-rk-1, A(rk+1,rk+1), lda, &vn1[rk+1], &vn2[rk+1], dlsticc, dlsticcs);
    }
    magma_free(dlsticcs);
    
    return MAGMA_SUCCESS;
} /* magma_dlaqps */
