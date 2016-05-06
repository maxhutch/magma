/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from magmablas/zgeqr2_batched.cu normal z -> s, Mon May  2 23:30:42 2016
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define BLOCK_SIZE 256


#define dA(a_1,a_2) (dA  + (a_1) + (a_2)*(local_lda))


#include "slarfg_devicesfunc.cuh"

//==============================================================================

static __device__
void slarfx_device( int m, int n,  float *v, float *tau,
                         float *dc, magma_int_t ldc, float* sum)
{
    if (n <= 0) return;
    if (MAGMA_S_EQUAL(*tau, MAGMA_S_ZERO) )  return; // check singularity

    const int tx = threadIdx.x;

    float lsum;
    
    for (int k=0; k < n; k++)
    {
        /* perform  w := v' * C  */
        if (tx < BLOCK_SIZE)
        {
            if (tx == 0)
                lsum = dc[0+ldc*k]; //since V[0] should be one
            else
                lsum = MAGMA_S_ZERO;
            for (int j = tx+1; j < m; j += BLOCK_SIZE) {
                lsum += MAGMA_S_MUL( MAGMA_S_CONJ( v[j] ), dc[j+ldc*k] );
            }

            sum[tx] = lsum;
        }

        magma_sum_reduce< BLOCK_SIZE >( tx, sum );
        __syncthreads();

        float z__1 = - MAGMA_S_CONJ(*tau) * sum[0];
        /*  C := C - v * w  */
        if (tx < BLOCK_SIZE)
        {
            for (int j = tx+1; j < m; j += BLOCK_SIZE)
                dc[j+ldc*k] += z__1 * v[j];
        }
        if (tx == 0) dc[0+ldc*k] += z__1;

        __syncthreads();
    }
}


//==============================================================================

static __device__
void sgeqr2_device( magma_int_t m, magma_int_t n,
                               float* dA, magma_int_t lda,
                               float *dtau,
                               float *dv,
                               float *sum,
                               float *swork,
                               float *scale,
                               float *sscale)
{
    //lapack slarfg, compute the norm, scale and generate the householder vector
    slarfg_device(m, dv, &(dv[1]), 1, dtau, swork, sscale, scale);
    
    __syncthreads();
    
    //update the trailing matix with the householder
    slarfx_device(m, n, dv, dtau, dA, lda, sum);
    
    __syncthreads();
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
    for (int s=0; s < n; s++)
    {
        for (int j = tx; j < m; j += BLOCK_SIZE)
        {
            sdata[j + s * m] = dA[j + s * lda];
        }
    }


    __syncthreads();
 
    for (int s=0; s < min(m,n); s++)
    {
        sgeqr2_device( m-s, n-(s+1),
                       &(sdata[s+(s+1)*m]), m,
                       dtau+s,
                       &(sdata[s+s*m]),
                       sum,
                       swork,
                       &scale,
                       &sscale);
    } // end of s

    //copy back to global memory
    for (int s=0; s < n; s++)
    {
        for (int j = tx; j < m; j += BLOCK_SIZE)
        {
            dA[j + s * lda] = sdata[j + s * m];
        }
    }
}






//==============================================================================

__global__
void sgeqr2_column_sm_kernel_batched( int m, int n, float** dA_array, magma_int_t lda,
                               float **dtau_array)
{
    float* dA = dA_array[blockIdx.z];
    float* dtau = dtau_array[blockIdx.z];

    float *sdata = (float*)shared_data;


    __shared__ float scale;
    __shared__ float sum[ BLOCK_SIZE ];

    __shared__ float swork[ BLOCK_SIZE ];
    __shared__ float sscale;

    const int tx = threadIdx.x;

    for (int s=0; s < min(m,n); s++)
    {
        //load one vector in shared memory: sdata
        for (int j = tx; j < m-s; j += BLOCK_SIZE)
        {
            sdata[j] = dA[s + j + s * lda];
        }
        
        __syncthreads();
        
        //sdata is written
        sgeqr2_device(m-s, n-(s+1),
                                &(dA[s+(s+1)*lda]), lda,
                                dtau+s,
                                sdata,
                                sum,
                                swork,
                                &scale,
                                &sscale);
        
        for (int j = tx; j < m-s; j += BLOCK_SIZE)
        {
            dA[s + j + s * lda] = sdata[j];
        }
        
        __syncthreads();
    }  
}


__global__
void sgeqr2_kernel_batched( int m, int n, float** dA_array, magma_int_t lda,
                               float **dtau_array)
{
    float* dA = dA_array[blockIdx.z];
    float* dtau = dtau_array[blockIdx.z];

    __shared__ float scale;
    __shared__ float sum[ BLOCK_SIZE ];

    __shared__ float swork[ BLOCK_SIZE ];
    __shared__ float sscale;



    for (int s=0; s < min(m,n); s++)
    {
        sgeqr2_device( m-s, n-(s+1),
                       &(dA[s+(s+1)*lda]), lda,
                       dtau+s,
                       &(dA[s+s*lda]),
                       sum,
                       swork,
                       &scale,
                       &sscale );
    }
}


//==============================================================================


/**
    Purpose
    -------
    SGEQR2 computes a QR factorization of a real m by n matrix A:
    A = Q * R.

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
    dA_array Array of pointers, dimension (batchCount).
             Each is a REAL array on the GPU, dimension (LDDA,N)
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

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a REAL array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

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

    @ingroup magma_sgeqrf_aux
    ********************************************************************/

extern "C" magma_int_t
magma_sgeqr2_batched(magma_int_t m, magma_int_t n, 
                     float **dA_array, magma_int_t ldda, 
                     float **dtau_array,
                     magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t k;

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

    k = min(m,n);

    dim3 blocks(1, 1, batchCount);
    dim3 threads(BLOCK_SIZE);

    if (sizeof(float)*(m*k) <= 42000 /*sizeof(float) * 128 * k*/) // there are some static shared memory besides of dynamic ones
    {
        //load panel in shared memory and factorize it and copy back to gloabl memory
        //intend for small panel to avoid overfill of shared memory.
        //this kernel is composed of device routine and thus clean
        sgeqr2_sm_kernel_batched<<< blocks, threads, sizeof(float)*(m*k), queue->cuda_stream() >>>
                                      (m, k, dA_array, ldda, dtau_array);
    }
    else
    {
        //load one column vector in shared memory and householder it and used it to update trailing matrix which is global memory
        // one vector is normally smaller than  48K shared memory
        if (sizeof(float)*(m) < 42000)
            sgeqr2_column_sm_kernel_batched<<< blocks, threads, sizeof(float)*(m), queue->cuda_stream() >>>
                                      (m, k, dA_array, ldda, dtau_array);
        else
            //not use dynamic shared memory at all
            sgeqr2_kernel_batched<<< blocks, threads, 0, queue->cuda_stream() >>>
                                      (m, k, dA_array, ldda, dtau_array);
    }

    return arginfo;
}



//==============================================================================
