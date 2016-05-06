/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zgesellcmmv.cu normal z -> s, Mon May  2 23:30:44 2016

*/
#include "magmasparse_internal.h"

#define PRECISION_s

//#define TEXTURE


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning one thread to each row - 1D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_1( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    const float * __restrict__ dval, 
    const magma_index_t * __restrict__ dcolind,
    const magma_index_t * __restrict__ drowptr,
    const float *__restrict__ dx,
    float beta, 
    float * __restrict__ dy)
{
    // threads assigned to rows
    //int Idx = blockDim.x * blockIdx.x + threadIdx.x;
    //int offset = drowptr[ blockIdx.x ];
    //int border = (drowptr[ blockIdx.x+1 ]-offset)/blocksize;
    
    
    // T threads assigned to each row
    int idx = threadIdx.x;      // local row
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * 256 + idx;  // global row index
    // int lblocksize = ( row + blocksize < num_rows) ? blocksize : ( num_rows - blocksize * (row/blocksize) );
    int lrow = threadIdx.x%blocksize; // local row;
    
    if( row < num_rows ) {
        int offset = drowptr[ row/blocksize ];
        int border = (drowptr[ row/blocksize+1 ]-offset)/blocksize;
    
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        for ( int n = 0; n < border; n++) { 
            int col = dcolind [ offset+ blocksize * n + lrow ];
            float val = dval[ offset+ blocksize * n + lrow ];
            dot = dot + val * dx [ col ];
        }

        if (betazero) {
            dy[ row ] = dot * alpha;
        } else {
            dy[ row ] = dot * alpha + beta * dy [ row ];
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_4( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    float *  dx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        float x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = dx[ i1 ];   
            x2 = dx[ i2 ]; 

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = dx[ dcolind[ block*kk] ];            
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 2 ) {
            shared[ldx]+=shared[ldx+blocksize*2];              
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_8( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    float *  dx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        float x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = dx[ i1 ];   
            x2 = dx[ i2 ]; 

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = dx[ dcolind[ block*kk] ];            
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 4 ) {
            shared[ldx]+=shared[ldx+blocksize*4];              
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_16( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    float *  dx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            float val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * dx[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 8 ) {
            shared[ldx]+=shared[ldx+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_32( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    float *  dx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            float val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * dx[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 16 ) {
            shared[ldx]+=shared[ldx+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}



/************************* same but using texture mem *************************/

#if defined(PRECISION_d) && defined(TEXTURE)

__inline__ __device__ float 
read_from_tex( cudaTextureObject_t texdx, const int& i) {
  int2 temp = tex1Dfetch<int2>( texdx, i ); 
  return __hiloint2float(temp.y,temp.x);
}

// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_4_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        float x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = read_from_tex( texdx, i1 );
            x2 = read_from_tex( texdx, i2 );

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = read_from_tex( texdx, dcolind[ block*kk] );
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 2 ) {
            shared[ldx]+=shared[ldx+blocksize*2];              
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_8_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        int kk, i1, i2;
        float x1, x2, v1, v2;
        dcolind += offset + ldx;
        dval += offset + ldx;
        for ( kk = 0; kk < max_-1; kk+=2 ) {
            i1 = dcolind[ block*kk];
            i2 = dcolind[ block*kk + block];

            x1 = read_from_tex( texdx, i1 );
            x2 = read_from_tex( texdx, i2 );

            v1 = dval[ block*kk ];
            v2 = dval[ block*kk + block];

            dot += v1 * x1;
            dot += v2 * x2;
        }
  
        if (kk<max_) {
           x1 = read_from_tex( texdx, dcolind[ block*kk] );
           v1 = dval[ block*kk ];

            dot += v1 * x1;
        }

        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 4 ) {
            shared[ldx]+=shared[ldx+blocksize*4];              
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_16_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        for ( int k = 0; k < max_; k++ ) {
            float val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * read_from_tex( texdx, col );
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 8 ) {
            shared[ldx]+=shared[ldx+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
template<bool betazero>
__global__ void 
zgesellptmv2d_kernel_32_tex( 
    int num_rows, 
    int num_cols,
    int blocksize,
    int T,
    float alpha, 
    float * dval, 
    magma_index_t * dcolind,
    magma_index_t * drowptr,
    cudaTextureObject_t texdx,
    float beta, 
    float * dy)
{
   // T threads assigned to each row
    int idx = threadIdx.y;      // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

    extern __shared__ float shared[];

    if(row < num_rows ) {
        float dot = MAGMA_S_MAKE(0.0, 0.0);
        int offset = drowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (drowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_; k++ ) {
            float val = 
                        dval[ offset + ldx + block*k ];
            int col = 
                    dcolind[ offset + ldx + block*k ];

            dot += val * read_from_tex( texdx, col );
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 16 ) {
            shared[ldx]+=shared[ldx+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                if (betazero) {
                    dy[row] = (shared[ldx]+shared[ldx+blocksize*1])*alpha;
                } else {
                    dy[row] = 
                    (shared[ldx]+shared[ldx+blocksize*1])*alpha + beta*dy [row];
                }
            }
        }
    }
}

#endif

/*********************     end of texture versions   **************************/

/**
    Purpose
    -------
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is SELLP.
    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 

    @param[in]
    blocksize   magma_int_t
                number of rows in one ELL-slice

    @param[in]
    slices      magma_int_t
                number of slices in matrix

    @param[in]
    alignment   magma_int_t
                number of threads assigned to one row

    @param[in]
    alpha       float
                scalar multiplier

    @param[in]
    dval        magmaFloat_ptr
                array containing values of A in SELLP

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in SELLP

    @param[in]
    drowptr     magmaIndex_ptr
                rowpointer of SELLP

    @param[in]
    dx          magmaFloat_ptr
                input vector x

    @param[in]
    beta        float
                scalar multiplier

    @param[out]
    dy          magmaFloat_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sblas
    ********************************************************************/

extern "C" magma_int_t
magma_sgesellpmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t blocksize,
    magma_int_t slices,
    magma_int_t alignment,
    float alpha,
    magmaFloat_ptr dval,
    magmaIndex_ptr dcolind,
    magmaIndex_ptr drowptr,
    magmaFloat_ptr dx,
    float beta,
    magmaFloat_ptr dy,
    magma_queue_t queue )
{
    // using a 2D thread grid

    int num_threads = blocksize*alignment;
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 && num_threads > 256 )
        printf("error: too much shared memory requested.\n");
    
    int dimgrid1 = min( int( sqrt( float( slices ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( slices, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( slices, dimgrid1*dimgrid2 );
    int num_tx = blocksize;
    int Ms = num_threads * sizeof( float );
    
    // special case: alignment 1:
    if( alignment == 1 ){
        Ms = 0;
        num_tx = 256;
        int num_blocks = magma_ceildiv( n, 256 );
        dimgrid1 = num_blocks; //min( int( sqrt( float( num_blocks ))), 65535 );
        dimgrid2 = 1; //magma_ceildiv( num_blocks, dimgrid1 );
        dimgrid3 = 1;
        //blocksize = 256;
    }
    
    dim3 block( num_tx, alignment, 1);

    if( dimgrid3 > 65535 ){
        printf("error: too many GPU thread blocks requested.\n");
    }
        
    dim3 grid( dimgrid1, dimgrid2, 1);

    #if defined(PRECISION_d) && defined(TEXTURE)

        // Create channel.
        cudaChannelFormatDesc channel_desc;
        channel_desc = 
            cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned);

        // Create resource descriptor.
        struct cudaResourceDesc resDescdx;
        memset(&resDescdx, 0, sizeof(resDescdx));
        resDescdx.resType = cudaResourceTypeLinear;
        resDescdx.res.linear.devPtr = (void*)dx;
        resDescdx.res.linear.desc = channel_desc;
        resDescdx.res.linear.sizeInBytes = m*sizeof(float);

        // Specify texture object parameters.
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode     = cudaFilterModePoint;
        texDesc.readMode       = cudaReadModeElementType;

        // Create texture object.
        cudaTextureObject_t texdx = 0;
        cudaCreateTextureObject(&texdx, &resDescdx, &texDesc, NULL);

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        if ( alignment == 1) {
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_1<true><<< grid2, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            } else {
                zgesellptmv2d_kernel_1<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            }
        } else if ( alignment == 4){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_4_tex<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            } else {
                zgesellptmv2d_kernel_4_tex<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            }
        }

        else if ( alignment == 8){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_8_tex<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            } else {
                zgesellptmv2d_kernel_8_tex<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            }
        }

        else if ( alignment == 16){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_16_tex<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            } else {
                zgesellptmv2d_kernel_16_tex<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            }
        }

        else if ( alignment == 32){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_32_tex<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            } else {
                zgesellptmv2d_kernel_32_tex<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, texdx, beta, dy );
            }
        }

        else {
            printf("error: alignment %d not supported.\n", alignment);
            return MAGMA_ERR_NOT_SUPPORTED;
        }

        cudaDestroyTextureObject(texdx);

    #else 
        if ( alignment == 1) {
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_1<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            } else {
                zgesellptmv2d_kernel_1<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            }
        }

        else if ( alignment == 4){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_4<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            } else {
                zgesellptmv2d_kernel_4<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            }
        }

        else if ( alignment == 8){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_8<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            } else {
                zgesellptmv2d_kernel_8<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            }
        }

        else if ( alignment == 16){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_16<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            } else {
                zgesellptmv2d_kernel_16<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            }
        }

        else if ( alignment == 32){
            if (beta == MAGMA_S_ZERO) {
                zgesellptmv2d_kernel_32<true><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            } else {
                zgesellptmv2d_kernel_32<false><<< grid, block, Ms, queue->cuda_stream() >>>
                ( m, n, blocksize, alignment, alpha,
                    dval, dcolind, drowptr, dx, beta, dy );
            }
        }

        else {
            printf("error: alignment %d not supported.\n", int(alignment) );
            return MAGMA_ERR_NOT_SUPPORTED;
        }
    #endif

   return MAGMA_SUCCESS;
}
