/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from zgeqr2_batched.cu normal z -> s, Fri Jan 30 19:00:10 2015
*/

#include "common_magma.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define BLOCK_SIZE 256
#define PRECISION_s

#define dA(a_1,a_2) (dA  + (a_1) + (a_2)*(local_lda))

#define REAL

//==============================================================================
static __device__ void
slarfg_device(
    int n,
    float* dalpha, float* dx, int incx,
    float* dtau,  float* swork, float* sscale, float* scale)
{

    const int tx = threadIdx.x;

    float tmp;
    
    // find max of [dalpha, dx], to use as scaling to avoid unnecesary under- and overflow    

    if ( tx == 0 ) {
        tmp = *dalpha;
        #ifdef COMPLEX
        swork[tx] = max( fabs(real(tmp)), fabs(imag(tmp)) );
        #else
        swork[tx] = fabs(tmp);
        #endif
    }
    else {
        swork[tx] = 0;
    }
    if(tx<BLOCK_SIZE)
    {
        for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
            tmp = dx[j*incx];
            #ifdef COMPLEX
            swork[tx] = max( swork[tx], max( fabs(real(tmp)), fabs(imag(tmp)) ));
            #else
            swork[tx] = max( swork[tx], fabs(tmp) );
            #endif
         }
    }

    magma_max_reduce<BLOCK_SIZE>( tx, swork );

    if ( tx == 0 )
        *sscale = swork[0];
    __syncthreads();
    
    // sum norm^2 of dx/sscale
    // dx has length n-1
    if(tx<BLOCK_SIZE) swork[tx] = 0;
    if ( *sscale > 0 ) {
        if(tx<BLOCK_SIZE)
        {
            for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
                tmp = dx[j*incx] / *sscale;
                swork[tx] += real(tmp)*real(tmp) + imag(tmp)*imag(tmp);
            }
        }
        magma_sum_reduce<BLOCK_SIZE>( tx, swork );

    }
    
    if ( tx == 0 ) {
        float alpha = *dalpha;

        if ( swork[0] == 0 && imag(alpha) == 0 ) {
            // H = I
            *dtau = MAGMA_S_ZERO;
        }
        else {
            // beta = norm( [dalpha, dx] )
            float beta;
            tmp  = alpha / *sscale;
            beta = *sscale * sqrt( real(tmp)*real(tmp) + imag(tmp)*imag(tmp) + swork[0] );
            beta = -copysign( beta, real(alpha) );
            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau   = MAGMA_S_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );
            *dalpha = MAGMA_S_MAKE( beta, 0 );
            *scale = 1 / (alpha - beta);
        }
    }
    
    // scale x (if norm was not 0)
    __syncthreads();
    if ( swork[0] != 0 ) {
        if(tx<BLOCK_SIZE)
        {
            for( int j = tx; j < n-1; j += BLOCK_SIZE ) {
                dx[j*incx] *= *scale;
            }
        }
    }

}




//==============================================================================

static __device__
void slarfx_device( int m, int n,  float *v, float *tau,
                         float *dc, magma_int_t ldc, float* sum)
{


    if(n <=0) return ;
    if (MAGMA_S_EQUAL(*tau, MAGMA_S_ZERO) )  return; // check singularity

    const int tx = threadIdx.x;

    float lsum;
       
    for(int k=0;k<n;k++)
    {
        /* perform  w := v' * C  */
        if(tx<BLOCK_SIZE)
        {
            if (tx==0)
                lsum = dc[0+ldc*k]; //since V[0] should be one
            else
                lsum = MAGMA_S_ZERO;
            for( int j = tx+1; j < m; j += BLOCK_SIZE ){
                lsum += MAGMA_S_MUL( MAGMA_S_CNJG( v[j] ), dc[j+ldc*k] );
            }

            sum[tx] = lsum;
        }

        magma_sum_reduce< BLOCK_SIZE >( tx, sum );
        __syncthreads();

        float z__1 = - MAGMA_S_CNJG(*tau) * sum[0];
        /*  C := C - v * w  */
        if(tx<BLOCK_SIZE)
        {    
           for( int j = tx+1; j<m ; j += BLOCK_SIZE )
                 dc[j+ldc*k] += z__1 * v[j];
        }
        if(tx==0) dc[0+ldc*k] += z__1;

        __syncthreads();


    } 
}

//==============================================================================

extern __shared__ float shared_data[];


__global__
void sgeqr2_sm_kernel_batched( int m, int n, float** dA_array, magma_int_t lda,
                               float **dtau_array)
{

    float* dA = dA_array[blockIdx.z];
    float* dtau = dtau_array[blockIdx.z];

    float *sdata = (float*)shared_data;

    const int tx = threadIdx.x;

    __shared__ float scale;
    __shared__ float sum[ BLOCK_SIZE ];

    __shared__ float swork[ BLOCK_SIZE ];
    __shared__ float sscale;
    
    //load data from global to shared memory
    for(int s=0;s<n;s++)
    {
        for( int j = tx; j < m; j += BLOCK_SIZE )
        {
            sdata[j + s * m] = dA[j + s * lda] ;
        }
    }


    __syncthreads();
 
    for(int s=0; s<min(m,n); s++)
    {

       //lapack slarfg, compute the norm, scale and generate the householder vector   

       slarfg_device(m-s, &(sdata[s+s*m]), &(sdata[s+1+s*m]), 1, dtau+s, swork, &sscale, &scale); 
       __syncthreads();

       
       //update the trailing matix with the householder
       slarfx_device(m-s, n-(s+1), &(sdata[s+s*m]), dtau+s,&(sdata[s+(s+1)*m]), m, sum);

    }// end of s

    //copy back to global memory
    for(int s=0;s<n;s++)
    {
        for( int j = tx; j < m; j += BLOCK_SIZE )
        {
            dA[j + s * lda] = sdata[j + s * m];
        }
    }

}




//==============================================================================



static __device__
void sgeqr2_device( magma_int_t m, magma_int_t n, float* dA, magma_int_t lda,
                               float *dtau, 
                               float *sdata,
                               float *sum,
                               float *swork,
                               float *scale,
                               float *sscale)
{

    const int tx = threadIdx.x;


    for(int s=0; s<min(m,n); s++)
    {
       //load one vector in shared memory: sdata
       for( int j = tx; j < m-s; j += BLOCK_SIZE )
       {
           sdata[j] = dA[s + j + s * lda] ;
       }

       __syncthreads();

       //if(tx== 0) printf("m-s=%d",m-s);
       //lapack slarfg, compute the norm, scale and generate the householder vector   
       slarfg_device(m-s, sdata, &(sdata[1]), 1, dtau+s, swork, sscale, scale); 

       __syncthreads();

       //update the trailing matix with the householder
       slarfx_device(m-s, n-(s+1), sdata, dtau+s, &(dA[s+(s+1)*lda]), lda, sum);

       for( int j = tx; j < m-s; j += BLOCK_SIZE )
       {
           dA[s + j + s * lda] = sdata[j];
       }

       __syncthreads();

    }// end of s

}



//==============================================================================

__global__
void sgeqr2_kernel_batched( int m, int n, float** dA_array, magma_int_t lda,
                               float **dtau_array)
{

    float* dA = dA_array[blockIdx.z];
    float* dtau = dtau_array[blockIdx.z];

    float *sdata = (float*)shared_data;


    __shared__ float scale;
    __shared__ float sum[ BLOCK_SIZE ];

    __shared__ float swork[ BLOCK_SIZE ];
    __shared__ float sscale;

    sgeqr2_device(m, n, dA, lda, dtau, sdata, sum, swork, &scale, &sscale); 
 
}




//==============================================================================


/**
    Purpose
    -------
    SGEQR2 computes a QR factorization of a real m by n matrix A:
    A = Q * R.

    This expert routine requires two more arguments than the standard
    sgeqr2, namely, dT and ddA, explained below. The storage for A is
    also not as in the LAPACK's sgeqr2 routine (see below).

    The first is used to output the triangular
    n x n factor T of the block reflector used in the factorization.
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R.

    This version implements the right-looking QR with non-blocking.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      REAL array, dimension (LDA,N)
            On entry, the m by n matrix A.
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).
    \n
            the elements on and above the diagonal of the array
            contain the min(m,n) by n upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

    @param[in]
    lda    INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    dtau    REAL array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      REAL array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector
            used in the factorization. The lower triangular part is 0.


    @param
    dwork   (workspace) REAL array, dimension (N) * ( sizeof(float) + sizeof(float)) 

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_sgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_sgeqr2_batched(magma_int_t m, magma_int_t n, float **dA_array,
                  magma_int_t lda, float **dtau_array,
                  magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    
    magma_int_t k;

    /* Check arguments */
    magma_int_t arginfo = 0;
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (lda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }


    k = min(m,n);

    dim3 blocks(1, 1, batchCount);
    dim3 threads(BLOCK_SIZE);

    if(sizeof(float)*(m*k) <= 128 /*sizeof(float) * 128 * k*/) // there are some static shared memory besides of dynamic ones 
    {   
        //load panel in shared memory and factorize it and copy back to gloabl memory
        //intend for small panel to avoid overfill of shared memory.
        //this kernel is composed of device routine and thus clean
        sgeqr2_sm_kernel_batched<<< blocks, threads, sizeof(float)*(m*k), queue >>>
                                      (m, k, dA_array, lda, dtau_array);
    }
    else
    {
        //load one column vector in shared memory and householder it and used it to update trailing matrix which is global memory 
        // one vector is normally smaller than  48K shared memory   
        if(sizeof(float)*(m) < 42000)
            sgeqr2_kernel_batched<<< blocks, threads, sizeof(float)*(m), queue >>>
                                      (m, k, dA_array, lda, dtau_array);
        else
            printf("m is too big, kernel launching failed, shared memory is overflowed");
    }


    return arginfo;

} 



//==============================================================================


