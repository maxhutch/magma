/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmdot_shfl.cu normal z -> d, Mon May  2 23:30:46 2016
       @author Moritz Kreutzer

*/
#include "magmasparse_internal.h"

#include "magmasparse_d.h"
#define BLOCK_SIZE 512

#define PRECISION_d

#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION <= 6000)
// CUDA 6.5 adds Double precision version; here's an implementation for CUDA 6.0 and earlier.
// from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__device__ inline
real_Double_t __shfl_down(real_Double_t var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}
#endif


template<typename T>
__inline__ __device__
T warpReduceSum(T val)
{
#if __CUDA_ARCH__ >= 300
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
#endif
    return val;
}


#ifdef PRECISION_z
template<>
__inline__ __device__
double warpReduceSum<double>(double val)
{
#if __CUDA_ARCH__ >= 300
    int4 a = *reinterpret_cast<int4*>(&val);
    a.x += __shfl_down(a.x, 16);
    a.y += __shfl_down(a.y, 16);
    a.z += __shfl_down(a.z, 16);
    a.w += __shfl_down(a.w, 16);
    a.x += __shfl_down(a.x, 8);
    a.y += __shfl_down(a.y, 8);
    a.z += __shfl_down(a.z, 8);
    a.w += __shfl_down(a.w, 8);
    a.x += __shfl_down(a.x, 4);
    a.y += __shfl_down(a.y, 4);
    a.z += __shfl_down(a.z, 4);
    a.w += __shfl_down(a.w, 4);
    a.x += __shfl_down(a.x, 2);
    a.y += __shfl_down(a.y, 2);
    a.z += __shfl_down(a.z, 2);
    a.w += __shfl_down(a.w, 2);
    a.x += __shfl_down(a.x, 1);
    a.y += __shfl_down(a.y, 1);
    a.z += __shfl_down(a.z, 1);
    a.w += __shfl_down(a.w, 1);
#endif
    return val;
}
#endif // PRECISION_z


#ifdef PRECISION_c
template<>
__inline__ __device__
magmaFloatComplex warpReduceSum<magmaFloatComplex>(magmaFloatComplex val)
{
#if __CUDA_ARCH__ >= 300
    float2 a = *reinterpret_cast<float2*>(&val);
    a.x += __shfl_down(a.x, 16);
    a.y += __shfl_down(a.y, 16);
    a.x += __shfl_down(a.x, 8);
    a.y += __shfl_down(a.y, 8);
    a.x += __shfl_down(a.x, 4);
    a.y += __shfl_down(a.y, 4);
    a.x += __shfl_down(a.x, 2);
    a.y += __shfl_down(a.y, 2);
    a.x += __shfl_down(a.x, 1);
    a.y += __shfl_down(a.y, 1);
#endif
    return val;
}
#endif // PRECISION_c


template<typename T>
__inline__ __device__
T blockReduceSum_1D(T val)
{
    extern __shared__ T shared[]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum<T>(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : MAGMA_D_ZERO;
    
    if (wid == 0) val = warpReduceSum<T>(val); //Final reduce within first warp
    return val;
}


template<typename T>
__inline__ __device__
T blockReduceSum(T val)
{
    extern __shared__ T shared[]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum<T>(val);     // Each warp performs partial reduction

    if (lane == 0) shared[threadIdx.y*32+wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.y*32+lane] : MAGMA_D_ZERO;
    
    if (wid == 0) val = warpReduceSum<T>(val); //Final reduce within first warp
    return val;
}


template<typename T> 
__global__ void deviceReduceKernel(const T * __restrict__ in, T * __restrict__ out, int N)
{
    T sum = MAGMA_D_MAKE(0.0, 0.0);
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    sum = blockReduceSum<T>(sum);
    if (threadIdx.x == 0)
        out[blockIdx.x]=sum;
}


// dot product for multiple vectors using shuffle intrinsics and less shared memory
__global__ void
magma_dblockdot_kernel_shuffle( 
    int n, 
    int k,
    const double * __restrict__ v,
    const double * __restrict__ r,
    double * __restrict__ vtmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    double tmp;
    if (i < n) {
        tmp = v[i+j*n] * r[i];
    } else {
        tmp = MAGMA_D_ZERO;
    }
    tmp = blockReduceSum(tmp);
    if (threadIdx.x == 0 ){
        vtmp[ blockIdx.x+j*gridDim.x ] = tmp;
    }
}


// dot product for multiple vectors using shuffle intrinsics and less shared memory
__global__ void
magma_dblockdot_kernel_shuffle_1dblock( 
    int n, 
    int k,
    const double * __restrict__ v,
    const double * __restrict__ r,
    double * __restrict__ vtmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    for (j=0; j < k; j++) {
        double tmp;
        if (i < n) {
            tmp = v[i+j*n] * r[i];
        } else {
            tmp = MAGMA_D_ZERO;
        }
        tmp = blockReduceSum_1D(tmp);
        if (threadIdx.x == 0 ){
            vtmp[ blockIdx.x+j*gridDim.x ] = tmp;
        }
    }
}


/**
    Purpose
    -------

    Computes the scalar product of a set of vectors v_i such that

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDouble_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDouble_ptr 
                r

    @param[in]
    d1          magmaDouble_ptr 
                workspace

    @param[in]
    d2          magmaDouble_ptr 
                workspace

    @param[out]
    skp         magmaDouble_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

#define PAD(n, p) (((n) < 1 || (p) < 1)?(n):(((n) % (p)) ? ((n) + (p) - (n) % (p)) : (n)))

extern "C" magma_int_t
magma_dmdotc_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDouble_ptr v, 
    magmaDouble_ptr r,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr skp,
    magma_queue_t queue )
{
    if ( magma_getdevice_arch() < 300 ) {
        return magma_dmdotc( n, k, v, r, d1, d2, skp, queue );
    }
    else if (1) { // 1D block kernel seems to be always faster
        dim3 block( BLOCK_SIZE );
        dim3 grid( magma_ceildiv( n, block.x ) );
        magma_dblockdot_kernel_shuffle_1dblock<<< grid, block, 32*sizeof(double), queue->cuda_stream() >>>( n, k, v, r, d1 );
        int j;
        for (j=0; j < k; j++) {
            deviceReduceKernel<double> <<<1, 1024, 32*sizeof(double), queue->cuda_stream()>>>(d1+grid.x*j, skp+j, grid.x);
        }
    } else {
        dim3 block( PAD(magma_ceildiv(BLOCK_SIZE, k), 32), k );
        while (block.x*block.y > 1024) {
            block.x -= 32;
        }
        dim3 grid( magma_ceildiv( n, block.x ) );
        magma_dblockdot_kernel_shuffle<<< grid, block, 32*k*sizeof(double), queue->cuda_stream() >>>( n, k, v, r, d1 );
        int j;
        for (j=0; j < k; j++) {
            deviceReduceKernel<double> <<<1, 1024, 32*sizeof(double), queue->cuda_stream()>>>(d1+grid.x*j, skp+j, grid.x);
        }
    }
   
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    This is an extension of the merged dot product above by chunking
    the set of vectors v_i such that the data always fits into cache.
    It is equivalent to a matrix vecor product Vr where V
    contains few rows and many columns. The computation is the same:

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDouble_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDouble_ptr 
                r

    @param[in]
    d1          magmaDouble_ptr 
                workspace

    @param[in]
    d2          magmaDouble_ptr 
                workspace

    @param[out]
    skp         magmaDouble_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_d
    ********************************************************************/

extern "C" magma_int_t
magma_dgemvmdot_shfl(
    magma_int_t n, 
    magma_int_t k, 
    magmaDouble_ptr v, 
    magmaDouble_ptr r,
    magmaDouble_ptr d1,
    magmaDouble_ptr d2,
    magmaDouble_ptr skp,
    magma_queue_t queue )
{
    if (k == 1) { // call CUBLAS dotc, we will never be faster
        double res = magma_ddot( n, v, 1, r, 1, queue );
        magma_dsetvector( 1, &res, 1, skp, 1, queue );
    }
    else if ( magma_getdevice_arch() < 300 ) {
        return magma_dgemvmdot( n, k, v, r, d1, d2, skp, queue );
    }
    else {
        magma_dmdotc_shfl( n, k, v, r, d1, d2, skp, queue );
    }

    return MAGMA_SUCCESS;
}
