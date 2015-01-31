/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlaqps3_gpu.cu normal z -> c, Fri Jan 30 19:00:09 2015

*/

#include "common_magma.h"
#include "commonblas_c.h"
#include "magma_templates.h"

#define PRECISION_c

#define BLOCK_SIZE 512


/* --------------------------------------------------------------------------- */

#define BLOCK_SIZE1 192

__global__ void
magma_cswap_gemv_kernel(int m, int rk, int n, const magmaFloatComplex * __restrict__ V, int ldv,
                     const magmaFloatComplex * __restrict__ x, int ldx, magmaFloatComplex *c, magmaFloatComplex *b)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE1 * blockIdx.x;
    magmaFloatComplex lsum, tmp;

    V += j;

    lsum = MAGMA_C_ZERO;
    if (j < m){
       tmp  = b[j];
       b[j] = c[j];
       if (j>=rk) 
          for(int k=0; k<n; k++)
              lsum += MAGMA_C_MUL( V[k*ldv], MAGMA_C_CNJG(x[k*ldx]));

       c[j] = tmp - lsum;
    }
}

__global__ void
magma_cgemv_kernel(int m, int n, const magmaFloatComplex * __restrict__ V, int ldv,
                     const magmaFloatComplex * __restrict__ x, magmaFloatComplex *b, magmaFloatComplex *c)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE1 * blockIdx.x;
    magmaFloatComplex lsum;

    V += j;

    lsum = MAGMA_C_ZERO;
    if (j < m){
        for(int k=0; k<n; k++)
            lsum += MAGMA_C_MUL( V[k*ldv], x[k]);

       c[j] = b[j] - lsum;
    }
}


__global__
void magma_cscale_kernel(int n, magmaFloatComplex* dx0,
                         magmaFloatComplex *dtau, float *dxnorm, magmaFloatComplex* dAkk)
{
   const int i = threadIdx.x;
   magmaFloatComplex tmp;
   __shared__ magmaFloatComplex scale;

   /* === Compute the norm of dx0 === */
   magmaFloatComplex *dx = dx0;
   __shared__ float sum[ BLOCK_SIZE ];
   float re, lsum;

   lsum = 0;
   for( int k = i; k < n; k += BLOCK_SIZE ) {

        #if (defined(PRECISION_s) || defined(PRECISION_d))
             re = dx[k];
             lsum += re*re;
        #else
             re = MAGMA_C_REAL( dx[k] );
             float im = MAGMA_C_IMAG( dx[k] );
             lsum += re*re + im*im;
        #endif
   }
   sum[i] = lsum;
   magma_sum_reduce< BLOCK_SIZE >( i, sum );

   /* === Compute the scaling factor === */
   if (i==0){
            float beta = sqrt(sum[0]);
            if ( beta == 0 ) {
              *dtau = MAGMA_C_ZERO;
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
               float alphar =  MAGMA_C_REAL(tmp), alphai = MAGMA_C_IMAG(tmp);
               beta  = -copysign( beta, alphar );

               // todo: deal with badly scaled vectors (see lapack's larfg)
               *dtau = MAGMA_C_MAKE((beta - alphar)/beta, -alphai/beta);
               *dAkk = MAGMA_C_MAKE(beta, 0.);

               tmp = MAGMA_C_MAKE( alphar - beta, alphai);
               scale = MAGMA_C_DIV( MAGMA_C_ONE, tmp);
#endif
            }
   }

   __syncthreads();

   /* === Scale the vector === */
   for(int j=i; j<n; j+=BLOCK_SIZE)
      dx0[j] = MAGMA_C_MUL(dx0[j], scale);

   /* === Make temporary the first element to 1; value is stored in dAkk === */
   if (i==0)
     dx0[0] = MAGMA_C_ONE;
}


#define BLOCK_SIZE2 192
#if (defined(PRECISION_z) || defined(PRECISION_d))
  #define TOL 1.e-8
#else
  #define TOL 1.e-4
#endif

__global__ void
magma_cgemv_kernel_adjust(int n, int k, magmaFloatComplex * A, int lda, 
                          magmaFloatComplex *B, int ldb, magmaFloatComplex *C,
                          float *xnorm, float *xnorm2, magmaFloatComplex *Akk, int *lsticc, int *lsticcs)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE2 * blockIdx.x;
    magmaFloatComplex sum;
    float temp, oldnorm;

    if (j<n) {
      B += j;
      sum = MAGMA_C_CNJG( B[(k-1)*ldb] );
      // sum = MAGMA_C_ZERO;
      for(int m=0; m<k-1; m++) {
         sum += MAGMA_C_MUL( MAGMA_C_CNJG( B[m*ldb] ), A[m*lda] );
      }
      C[j*lda] -= sum;

      oldnorm = xnorm[j];
      temp = MAGMA_C_ABS( C[j*lda] ) / oldnorm;
      temp  = (1.0 + temp) * (1.0 - temp);
      temp  = oldnorm * sqrt(temp);

      xnorm[j] = temp;

      // Below 'j' was 'i'; was that a bug?
      float temp2 = xnorm[j] / xnorm2[j];
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
           magmaFloatComplex *dx = C+blockIdx.x*lda+1;
           __shared__ float sum[ BLOCK_SIZE2 ];
           float re, lsum;
 
           // get norm of dx
           lsum = 0;
           for( int k = i; k < n1; k += BLOCK_SIZE2 ) {

               #if (defined(PRECISION_s) || defined(PRECISION_d))
                   re = dx[k];
                   lsum += re*re;
               #else
                   re = MAGMA_C_REAL( dx[k] );
                   float im = MAGMA_C_IMAG( dx[k] );
                   lsum += re*re + im*im;
               #endif
           }
           sum[i] = lsum;
           magma_sum_reduce< BLOCK_SIZE2 >( i, sum );

           if (i==0){
             printf("adjusted = %f recomputed = %f\n", xnorm[blockIdx.x], sqrt(sum[0])); 
             xnorm[blockIdx.x] = sqrt(sum[0]);
           }
      }
 //   }
*/
}

__global__ void
magmablas_scnrm2_check_kernel(int m, magmaFloatComplex *da, int ldda, 
                              float *dxnorm, float *dxnorm2, 
                              int *dlsticc, int *dlsticcs)
{
    const int i = threadIdx.x;
    magmaFloatComplex *dx = da + blockIdx.x * ldda;

    __shared__ float sum[ BLOCK_SIZE ];
    float re, lsum;

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
        re = MAGMA_C_REAL( dx[j] );
        float im = MAGMA_C_IMAG( dx[j] );
        lsum += re*re + im*im;
#endif

    }
    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    if (i==0){
      dxnorm[blockIdx.x]  = sqrt(sum[0]);
      dxnorm2[blockIdx.x] = sqrt(sum[0]);
    }
}


/* --------------------------------------------------------------------------- */



/**
    Purpose
    -------
    CLAQPS computes a step of QR factorization with column pivoting
    of a complex M-by-N matrix A by using Blas-3.  It tries to factorize
    NB columns from A starting from the row OFFSET+1, and updates all
    of the matrix with Blas-3 xGEMM.

    In some cases, due to catastrophic cancellations, it cannot
    factorize NB columns.  Hence, the actual number of factorized
    columns is returned in KB.

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. N >= 0

    @param[in]
    offset  INTEGER
            The number of rows of A that have been factorized in
            previous steps.

    @param[in]
    NB      INTEGER
            The number of columns to factorize.

    @param[out]
    kb      INTEGER
            The number of columns actually factorized.

    @param[in,out]
    A       COMPLEX array, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    @param[out]
    dtau    COMPLEX array, dimension (KB)
            The scalar factors of the elementary reflectors.

    @param[in,out]
    dvn1    REAL array, dimension (N)
            The vector with the partial column norms.

    @param[in,out]
    dvn2    REAL array, dimension (N)
            The vector with the exact column norms.

    @param[in,out]
    dauxv   COMPLEX array, dimension (NB)
            Auxiliar vector.

    @param[in,out]
    dF       COMPLEX array, dimension (LDDF,NB)
            Matrix F**H = L * Y**H * A.

    @param[in]
    lddf    INTEGER
            The leading dimension of the array F. LDDF >= max(1,N).

    @ingroup magma_cgeqp3_aux
    ********************************************************************/
extern "C" magma_int_t
magma_claqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaFloatComplex_ptr dtau, 
    magmaFloat_ptr dvn1, magmaFloat_ptr dvn2,
    magmaFloatComplex_ptr dauxv,
    magmaFloatComplex_ptr dF,  magma_int_t lddf)
{
#define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
#define dF(i_, j_) (dF + (i_) + (j_)*(lddf))

    magmaFloatComplex c_zero    = MAGMA_C_MAKE( 0.,0.);
    magmaFloatComplex c_one     = MAGMA_C_MAKE( 1.,0.);
    magmaFloatComplex c_neg_one = MAGMA_C_MAKE(-1.,0.);
    magma_int_t ione = 1;
    
    magma_int_t i__1, i__2;
    
    magma_int_t k, rk;
    magmaFloatComplex tauk;
    magma_int_t pvt, itemp;

    magmaFloatComplex_ptr dAkk = dauxv;
    dauxv += 1;

    int lsticc, *dlsticc, *dlsticcs;
    magma_malloc( (void**) &dlsticcs, (n+1)*sizeof(int) );
    cudaMemset( dlsticcs, 0, (n+1)*sizeof(int) );
    dlsticc = dlsticcs + n;
 
    // float tol3z = magma_ssqrt( lapackf77_slamch("Epsilon"));

    lsticc = 0;
    k = 0;
    while( k < nb && lsticc == 0 ) {
        rk = offset + k;
        
        /* Determine ith pivot column and swap if necessary */
        pvt = k - 1 + magma_isamax( n-k, &dvn1[k], ione );

        if (pvt != k) {
            magmablas_cswap( k, dF(pvt,0), lddf, dF(k,0), lddf);
            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            #if (defined(PRECISION_d) || defined(PRECISION_z))
                //magma_dswap( 1, &dvn1[pvt], 1, &dvn1[k], 1 );
                //magma_dswap( 1, &dvn2[pvt], 1, &dvn2[k], 1 );
                magma_dswap( 2, &dvn1[pvt], n+offset, &dvn1[k], n+offset);
            #else
                //magma_sswap( 1, &dvn1[pvt], 1, &dvn1[k], 1 );
                //magma_sswap( 1, &dvn2[pvt], 1, &dvn2[k], 1 );
                magma_sswap(2, &dvn1[pvt], n+offset, &dvn1[k], n+offset);
            #endif
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'  */
        magma_cswap_gemv_kernel<<< (m + BLOCK_SIZE1-1) / BLOCK_SIZE1, BLOCK_SIZE1, 0, magma_stream >>> 
                              ( m, rk, k, dA(0, 0), ldda, dF(k,  0), lddf, dA(0, k), dA(0,pvt));
                                 
        /*  Generate elementary reflector H(k). */
        magma_cscale_kernel<<< 1, BLOCK_SIZE, 0, magma_stream >>>
               (m-rk, dA(rk, k),   &dtau[k], &dvn1[k], dAkk);
        // printf("m-rk = %d\n", m-rk);

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1) {
            magma_cgetvector( 1, &dtau[k], 1, &tauk, 1 );
            magma_cgemv( MagmaConjTrans, m-rk, n,
                         tauk,   dA( rk,  0 ), ldda,
                                 dA( rk,  k   ), 1,
                         c_zero, dauxv, 1 );
            if (k==0) 
               magmablas_clacpy(MagmaUpperLower, n-k-1, 1, dauxv+k+1, n-k-1, dF( k+1, k   ), n-k-1);
        }
        
        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). 
           F(1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K)
                    := tau(K)(A(RK:M,K+1:N)' - F(1:N,1:K-1)*A(RK:M,1:K-1)') A(RK:M,K)  
           so, F is (updated A)*V */
        if (k > 0) {
            /* I think we only need stricly lower-triangular part */
            magma_cgemv_kernel<<< (n-k-1 + BLOCK_SIZE1 -1)/BLOCK_SIZE1, BLOCK_SIZE1, 0, magma_stream >>>
                       (n-k-1, k, dF(k+1,0), lddf, dauxv, dauxv+k+1, dF(k+1,k));
        }
        
        /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            /* left-looking update of rows,                     *
             * since F=A**H v with original A, so no right-looking */
            magma_cgemv_kernel_adjust<<<(n-k-1 + BLOCK_SIZE2-1)/BLOCK_SIZE2, BLOCK_SIZE2, 0, magma_stream>>>
                           (n-k-1, k+1, dA(rk, 0  ), ldda, dF(k+1,0  ), lddf, dA(rk, k+1),
                           &dvn1[k+1], &dvn2[k+1], dAkk, dlsticc, dlsticcs);
            magma_getmatrix(1,1, sizeof(int), dlsticc, 1, &lsticc, 1); 
 
            // TTT: force not to recompute; has to be finally commented 
            if ( nb<3 )
            lsticc = 0; 

            // printf("k=%d n-k = %d\n", k, n-k);
            // forcing recompute works! - forcing it requires changing dlsticcs as well, e.g.,
            // can be done in the kernel directly (magmablas_scnrm2_check_kernel)
            // if (k==16) lsticc = 1;
        }
        
        /* Update partial column norms. */
/*
        if (rk < min(m, n+offset)-1){
           magmablas_scnrm2_row_check_adjust(n-k-1, tol3z, &dvn1[k+1], 
                                             &dvn2[k+1], dA(rk,k+1), ldda, lsticcs); 
        }

        #if defined(PRECISION_d) || defined(PRECISION_z)
            magma_sgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
        #else
            magma_sgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
        #endif
*/

        if (k>=n-1)
           magmablas_clacpy(MagmaUpperLower, 1, 1, dAkk, 1, dA(rk, k), 1);

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
        
        magma_cgemm( MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb,
                     c_neg_one, dA(rk+1, 0  ), ldda,
                                dF(*kb,  0  ), lddf,
                     c_one,     dA(rk+1, *kb), ldda );
    }

    /* Recomputation of difficult columns. */
    if( lsticc > 0 ) {
        // printf( " -- recompute dnorms --\n" );
        //magmablas_scnrm2_check(m-rk-1, n-*kb, A(rk+1,rk+1), lda,
        //                       &dvn1[rk+1], &dvn2[rk+1], dlsticcs);
       
        // There is a bug when we get to recompute  
        magmablas_scnrm2_check_kernel<<< n-*kb, BLOCK_SIZE >>>
                     ( m-rk-1, dA(rk+1,rk+1), ldda, &dvn1[rk+1], &dvn2[rk+1], dlsticc, dlsticcs);
    }
    magma_free(dlsticcs);
    
    return MAGMA_SUCCESS;
} /* magma_claqps */
